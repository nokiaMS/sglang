# InternVL 多模态处理器模块
# 本模块实现了 InternVL 和 InternS1 模型的多模态数据处理逻辑，
# 支持图像和视频输入，包括动态图像分块预处理、视频帧提取、
# 两种提示格式（Qwen 风格和 InternLM2 风格）的处理等功能。
# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py

import logging  # 导入日志模块
from functools import lru_cache  # 导入 LRU 缓存装饰器
from typing import List  # 导入列表类型提示

import numpy as np  # 导入 NumPy 模块
import torch  # 导入 PyTorch 模块
from PIL import Image  # 导入 PIL 图像模块

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举类
    MultimodalDataItem,  # 多模态数据项类
    MultimodalProcessorOutput,  # 多模态处理器输出类
)
from sglang.srt.models.interns1 import InternS1ForConditionalGeneration  # 导入 InternS1 条件生成模型类
from sglang.srt.models.internvl import InternVLChatModel  # 导入 InternVL 聊天模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor,  # 基础多模态处理器类
    BaseMultiModalProcessorOutput,  # 基础多模态处理器输出类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)
from sglang.srt.utils import get_device  # 导入获取设备工具函数
from sglang.srt.utils.video_decoder import VideoDecoderWrapper  # 导入视频解码器包装类

logger = logging.getLogger(__name__)  # 创建模块级日志记录器


class InternVLProcessor(BaseMultimodalProcessor):  # InternVL 处理器类，继承自基础多模态处理器
    models = [InternVLChatModel, InternS1ForConditionalGeneration]  # 支持的模型列表
    gpu_image_decode = False  # InternVL HF processor does not support tensor inputs  # InternVL HuggingFace 处理器不支持张量输入

    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet 归一化均值
    IMAGENET_STD = [0.229, 0.224, 0.225]  # ImageNet 归一化标准差
    IMAGE_MAX_NUM = 12  # 单张图像最大分块数

    DEFAULT_VIDEO_NUM_FRAMES = 32  # 默认视频帧数
    VIDEO_MAX_NUM = 1  # 单个视频帧最大分块数
    VIDEO_USE_THUMBNAIL = False  # 视频是否使用缩略图

    CONTEXT_FALLBACK = 40960  # 上下文长度回退值
    CONTEXT_RESERVED = 256  # 预留的上下文长度

    # OpenAI multimodal placeholder tokens
    IMAGE_PLACEHOLDER_TOKEN = "<image>"  # OpenAI 风格图像占位符标记
    VIDEO_PLACEHOLDER_TOKEN = "<video>"  # OpenAI 风格视频占位符标记

    IMG_START = "<img>"  # 图像起始标记
    IMG_END = "</img>"  # 图像结束标记
    IMG_CONTEXT = "<IMG_CONTEXT>"  # 图像上下文标记

    @staticmethod  # 静态方法
    @lru_cache(maxsize=1)  # 使用 LRU 缓存，最多缓存 1 个结果
    def _get_normalize_tensors(device="cuda", dtype=torch.float32):  # 获取归一化张量（均值和标准差），带缓存
        mean = torch.tensor(  # 创建均值张量
            InternVLProcessor.IMAGENET_MEAN, device=device, dtype=dtype  # 使用 ImageNet 均值
        ).view(-1, 1, 1)  # 调整形状为 (3, 1, 1) 以支持广播
        std = torch.tensor(  # 创建标准差张量
            InternVLProcessor.IMAGENET_STD, device=device, dtype=dtype  # 使用 ImageNet 标准差
        ).view(-1, 1, 1)  # 调整形状为 (3, 1, 1) 以支持广播
        return mean, std  # 返回均值和标准差张量

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):  # 初始化 InternVL 处理器
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)  # 调用父类初始化方法

        image_size = (  # 获取图像尺寸
            getattr(hf_config, "force_image_size", None)  # 尝试获取强制图像尺寸
            or hf_config.vision_config.image_size  # 否则使用视觉配置中的图像尺寸
        )
        patch_size = hf_config.vision_config.patch_size  # 获取 patch 大小
        if isinstance(image_size, list):  # 如果图像尺寸是列表
            image_size = image_size[0]  # 取第一个元素
        if isinstance(patch_size, list):  # 如果 patch 大小是列表
            patch_size = patch_size[0]  # 取第一个元素

        if hasattr(self._processor, "tokenizer"):  # 如果处理器有分词器属性
            tokenizer = self._processor.tokenizer  # 获取分词器
        else:  # 否则
            tokenizer = self._processor  # 处理器本身就是分词器
        self.tokenizer = tokenizer  # 保存分词器

        # Support both InternVL (llm_config) and InternS1 (text_config).
        # Different multimodal models use different field names for the text backbone:
        # - InternVL uses: hf_config.llm_config
        # - InternS1 uses: hf_config.text_config
        # - Some store architectures at top-level
        text_cfg = (  # 获取文本配置
            getattr(hf_config, "llm_config", None)  # 尝试获取 llm_config
            or getattr(hf_config, "text_config", None)  # 尝试获取 text_config
            or hf_config  # 使用 hf_config 本身
        )
        llm_arch = (getattr(text_cfg, "architectures", []) or [None])[0]  # 获取 LLM 架构名称
        self.llm_arch = llm_arch  # 保存 LLM 架构名称
        video_token_map = {  # 不同 LLM 架构对应的视频填充标记映射
            "Qwen2ForCausalLM": "<|video_pad|>",  # Qwen2 使用 video_pad
            "Qwen3ForCausalLM": "<|video_pad|>",  # Qwen3 使用 video_pad
            "Qwen3MoeForCausalLM": "<|video_pad|>",  # Qwen3 MoE 使用 video_pad
            "GptOssForCausalLM": "<|reserved_200000|>",  # GptOss 使用 reserved_200000
        }
        self.VIDEO_CONTEXT_TOKEN = video_token_map.get(llm_arch, None)  # 获取当前架构的视频上下文标记
        self.video_token_id = (  # 获取视频标记 token ID
            tokenizer.convert_tokens_to_ids(self.VIDEO_CONTEXT_TOKEN)  # 将视频标记转换为 ID
            if self.VIDEO_CONTEXT_TOKEN  # 如果视频上下文标记存在
            else None  # 否则为 None
        )

        self.image_token_id = (  # 获取图像标记 token ID
            tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)  # 将图像上下文标记转换为 ID
            if self.IMG_CONTEXT  # 如果图像上下文标记存在
            else None  # 否则为 None
        )
        self.num_image_token = int(  # 计算每个图像分块的 token 数量
            (image_size // patch_size) ** 2 * (hf_config.downsample_ratio**2)  # 基于 patch 数和下采样率计算
        )

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START)  # 图像起始标记 token ID
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END)  # 图像结束标记 token ID

        # Placeholder token use <image>/<video>
        # Offset token id use IMG_CONTEXT / VIDEO_CONTEXT
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=self.IMAGE_PLACEHOLDER_TOKEN,  # 图像占位符标记
            image_token_id=self.image_token_id,  # 图像标记 ID
            video_token=self.VIDEO_PLACEHOLDER_TOKEN,  # 视频占位符标记
            video_token_id=self.video_token_id,  # 视频标记 ID
        ).build(_image_processor)  # 使用处理器构建标记

        # Cache token id for IMG_CONTEXT (used by both branches)
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)  # 缓存图像上下文标记 token ID

        # InternLM2 legacy multimodal tokens: use <IMG_CONTEXT> as placeholder
        self.mm_tokens_internlm2 = MultimodalSpecialTokens(  # InternLM2 遗留多模态标记
            image_token=self.IMG_CONTEXT,  # 使用 IMG_CONTEXT 作为占位符
            image_token_id=self.img_context_token_id,  # 图像上下文标记 ID
        ).build(_image_processor)  # 使用处理器构建标记

        self.max_context_len = (  # 获取最大上下文长度
            getattr(server_args, "context_length", None)  # 从服务器参数获取
            or getattr(server_args, "max_context_len", None)  # 尝试获取 max_context_len
            or getattr(hf_config, "max_position_embeddings", None)  # 从配置获取最大位置嵌入数
            or getattr(text_cfg, "max_position_embeddings", None)  # 从文本配置获取
            or self.CONTEXT_FALLBACK  # 使用回退值
        )

    @staticmethod  # 静态方法
    def dynamic_preprocess(  # 动态预处理图像，将图像分割为多个分块（tiles）
        tensor, image_size=448, max_num=IMAGE_MAX_NUM, use_thumbnail=False  # 输入张量、分块大小、最大分块数、是否使用缩略图
    ):
        # Tensor: (C,H,W) float on GPU
        C, H, W = tensor.shape  # 获取通道数、高度和宽度
        aspect_ratio = W / H  # 计算宽高比

        # Generate all possible aspect ratios
        target_ratios = set(  # 生成所有可能的目标宽高比组合
            (i, j)  # 宽高比元组
            for n in range(1, max_num + 1)  # 遍历分块总数
            for i in range(1, n + 1)  # 遍历宽度分块数
            for j in range(1, n + 1)  # 遍历高度分块数
            if i * j <= max_num  # 确保总分块数不超过最大值
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])  # 按总分块数排序

        # Find closest ratio
        best_ratio_diff = float("inf")  # 最佳宽高比差异，初始化为无穷大
        best_ratio = (1, 1)  # 最佳宽高比，初始化为 (1, 1)

        for x, y in target_ratios:  # 遍历所有目标宽高比
            target_ar = x / y  # 计算目标宽高比
            diff = abs(aspect_ratio - target_ar)  # 计算与原始宽高比的差异
            blocks = x * y  # 计算分块总数
            best_blocks = best_ratio[0] * best_ratio[1]  # 当前最佳分块总数

            if diff < best_ratio_diff:  # 如果差异更小
                best_ratio_diff = diff  # 更新最佳差异
                best_ratio = (x, y)  # 更新最佳宽高比
            elif diff == best_ratio_diff and blocks > best_blocks:  # 如果差异相同但分块更多
                best_ratio = (x, y)  # 更新最佳宽高比

        target_w, target_h = image_size * best_ratio[0], image_size * best_ratio[1]  # 计算目标宽度和高度
        blocks = best_ratio[0] * best_ratio[1]  # 计算总分块数

        # Resize on GPU
        resized = torch.nn.functional.interpolate(  # 在 GPU 上进行图像缩放
            tensor.unsqueeze(0),  # 增加批次维度
            size=(target_h, target_w),  # 目标尺寸
            mode="bicubic",  # 双三次插值
            align_corners=False,  # 不对齐角点
        ).squeeze(0)  # 去除批次维度

        # Split into tiles
        tiles = []  # 初始化分块列表
        for i in range(blocks):  # 遍历所有分块
            x = (i % best_ratio[0]) * image_size  # 计算当前分块的 x 坐标
            y = (i // best_ratio[0]) * image_size  # 计算当前分块的 y 坐标
            tile = resized[:, y : y + image_size, x : x + image_size]  # 裁剪当前分块
            tiles.append(tile)  # 添加到分块列表

        # Add thumbnail if needed
        if use_thumbnail and len(tiles) > 1:  # 如果需要缩略图且有多个分块
            thumb = torch.nn.functional.interpolate(  # 生成缩略图
                tensor.unsqueeze(0),  # 增加批次维度
                size=(image_size, image_size),  # 缩略图尺寸
                mode="bicubic",  # 双三次插值
                align_corners=False,  # 不对齐角点
            ).squeeze(0)  # 去除批次维度
            tiles.append(thumb)  # 将缩略图添加到分块列表

        return torch.stack(tiles).to(torch.bfloat16)  # 将所有分块堆叠为张量并转换为 bfloat16

    @staticmethod  # 静态方法
    def _open_video_reader(path: str):  # 打开视频读取器
        return VideoDecoderWrapper(path)  # 返回视频解码器包装对象

    def _ensure_placeholders_before_assistant(  # 确保在 assistant 标记之前有足够的占位符
        self, prompt: str, placeholder: str, want: int  # 提示文本、占位符标记、需要的数量
    ) -> str:
        if want <= 0:  # 如果不需要占位符
            return prompt  # 直接返回原始提示
        have = (prompt or "").count(placeholder)  # 计算当前占位符数量
        missing = want - have  # 计算缺少的占位符数量
        if missing <= 0:  # 如果不缺少
            return prompt  # 直接返回原始提示

        insert = "\n" + "\n".join([placeholder] * missing) + "\n"  # 构建要插入的占位符字符串

        marker = "<|im_start|>assistant"  # assistant 标记
        idx = (prompt or "").rfind(marker)  # 查找 assistant 标记的位置
        if idx != -1:  # 如果找到了
            return (prompt or "")[:idx] + insert + (prompt or "")[idx:]  # 在 assistant 标记前插入占位符
        return (prompt or "") + insert  # 否则在末尾添加占位符

    def _token_len(self, text: str) -> int:  # 计算文本的 token 长度
        try:  # 尝试计算
            ids = self.tokenizer(text, return_tensors="pt")["input_ids"].flatten()  # 分词并获取 ID
            return int(ids.numel())  # 返回 token 数量
        except Exception:  # 如果出错
            return 0  # 返回 0

    def _resolve_video_num_frames(  # 解析视频帧数，根据上下文长度预算计算
        self, *, requested: int, num_videos: int, text_len: int, image_tile_cnt: int
    ) -> int:
        if num_videos <= 0:  # 如果没有视频
            return 0  # 返回 0 帧
        if not self.VIDEO_CONTEXT_TOKEN or not self.video_token_id:  # 如果不支持视频
            return 0  # 返回 0 帧
        image_tokens = image_tile_cnt * self.num_image_token  # 计算图像占用的 token 数
        budget = (  # 计算剩余的 token 预算
            int(self.max_context_len)  # 最大上下文长度
            - int(text_len)  # 减去文本长度
            - int(image_tokens)  # 减去图像 token 数
            - int(self.CONTEXT_RESERVED)  # 减去预留长度
        )
        if budget <= 0:  # 如果预算不足
            return 1  # 至少返回 1 帧
        max_total_frames = max(1, budget // self.num_image_token)  # 计算最大总帧数
        frames_per_video = max(1, max_total_frames // max(num_videos, 1))  # 计算每个视频的帧数
        return max(1, min(int(requested), int(frames_per_video)))  # 返回限制后的帧数

    @staticmethod  # 静态方法
    def _has_special_format(image_data, video_data):  # 检查是否有使用特殊格式的输入项
        """Check if any input items use processor_output or precomputed_embedding format."""
        """检查是否有输入项使用 processor_output 或 precomputed_embedding 格式。"""
        for data in list(image_data or []) + list(video_data or []):  # 遍历所有图像和视频数据
            if isinstance(data, dict) and data.get("format") in (  # 如果数据是字典且格式为特殊格式
                "processor_output",  # 处理器输出格式
                "precomputed_embedding",  # 预计算嵌入格式
            ):
                return True  # 返回 True
        return False  # 没有特殊格式，返回 False

    async def _process_special_format(  # 处理特殊格式的多模态数据（processor_output 和 precomputed_embedding）
        self, image_data, video_data, input_text, request_obj, **kwargs
    ):
        """Handle processor_output and precomputed_embedding input formats.
        处理 processor_output 和 precomputed_embedding 输入格式。

        Delegates to the base class process_and_combine_mm_data which has
        built-in support for these formats.
        委托给基类的 process_and_combine_mm_data 方法，该方法内置支持这些格式。
        """
        # When user provides input_ids directly, input_text may be a list of ints
        if isinstance(input_text, list):  # 如果输入文本是列表（即用户直接提供了 input_ids）
            user_input_ids = input_text  # 保存用户提供的 input_ids
            prompt = ""  # 提示文本设为空
        else:  # 否则
            user_input_ids = None  # 没有用户提供的 input_ids
            prompt = input_text or ""  # 使用输入文本

        # When the prompt is empty (user provided input_ids directly),
        # load_mm_data can't match multimodal tokens to data items.
        # Build BaseMultiModalProcessorOutput directly from the dict items.
        if not prompt and (image_data or video_data):  # 如果提示为空但有图像或视频数据
            images = [d for d in (image_data or []) if isinstance(d, dict)]  # 筛选字典类型的图像数据
            videos = [d for d in (video_data or []) if isinstance(d, dict)]  # 筛选字典类型的视频数据

            # Raise if raw (non-dict) images/videos were silently filtered out.
            # InternVL cannot process raw images without a text prompt because
            # dynamic tiling and placeholder expansion require the prompt string.
            raw_img_dropped = len(image_data or []) - len(images)  # 计算被过滤的原始图像数
            raw_vid_dropped = len(video_data or []) - len(videos)  # 计算被过滤的原始视频数
            if raw_img_dropped > 0 or raw_vid_dropped > 0:  # 如果有被过滤的原始数据
                raise ValueError(  # 抛出异常
                    f"[internvl] Cannot process raw images/videos with pre-tokenized "  # 无法使用预分词的 input_ids 处理原始图像/视频
                    f"input_ids. Provide multimodal data in 'processor_output' or "  # 请以 processor_output 格式提供多模态数据
                    f"'precomputed_embedding' format, or use a text prompt instead. "  # 或使用文本提示
                    f"(raw images dropped: {raw_img_dropped}, "  # 被过滤的原始图像数
                    f"raw videos dropped: {raw_vid_dropped})"  # 被过滤的原始视频数
                )

            base_output = BaseMultiModalProcessorOutput(  # 直接构建基础多模态处理器输出
                input_text=prompt,  # 输入文本
                images=images,  # 图像数据
                videos=videos,  # 视频数据
            )
        else:  # 否则正常加载
            base_output = await self.load_mm_data(  # 异步加载多模态数据
                prompt=prompt,  # 输入提示文本
                image_data=image_data,  # 图像数据
                video_data=video_data,  # 视频数据
                multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
                discard_alpha_channel=True,  # 丢弃透明通道
            )

        mm_items, input_ids_tensor, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )

        # If user provided input_ids directly, use those and recompute offsets
        if user_input_ids is not None:  # 如果用户直接提供了 input_ids
            input_ids_tensor = torch.tensor(user_input_ids, dtype=torch.long)  # 将用户 input_ids 转为张量
            for mm_item in mm_items:  # 遍历多模态项
                if (  # 如果是视频模态且有视频标记 ID
                    mm_item.modality == Modality.VIDEO  # 检查是否为视频模态
                    and self.video_token_id is not None  # 检查视频标记 ID 是否存在
                ):
                    mm_token_id = self.video_token_id  # 使用视频标记 ID
                else:  # 否则
                    mm_token_id = self.img_context_token_id  # 使用图像上下文标记 ID
                mm_item.offsets = self.get_mm_items_offset(  # 重新计算多模态项偏移量
                    input_ids=input_ids_tensor,  # 输入 ID 张量
                    mm_token_id=mm_token_id,  # 多模态标记 ID
                )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids_tensor.flatten().tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_start_id=self.img_start_token_id,  # 图像起始标记 ID
            im_end_id=self.img_end_token_id,  # 图像结束标记 ID
            im_token_id=self.img_context_token_id,  # 图像上下文标记 ID
            video_token_id=self.video_token_id,  # 视频标记 ID
        )

    async def process_mm_data_async(  # 异步处理多模态数据，根据架构选择不同的处理分支
        self, image_data, input_text, request_obj, **kwargs
    ):
        video_data = getattr(request_obj, "video_data", None) or []  # 获取视频数据

        # Handle processor_output and precomputed_embedding formats
        if isinstance(input_text, list) or self._has_special_format(  # 如果输入文本是列表或有特殊格式
            image_data, video_data
        ):
            return await self._process_special_format(  # 使用特殊格式处理
                image_data=image_data,  # 图像数据
                video_data=video_data,  # 视频数据
                input_text=input_text,  # 输入文本
                request_obj=request_obj,  # 请求对象
                **kwargs,  # 其他参数
            )

        is_internlm2 = self.llm_arch == "InternLM2ForCausalLM"  # 判断是否为 InternLM2 架构

        if is_internlm2:  # 如果是 InternLM2 架构
            return await self.process_internlm2_mm_data_async(  # 使用 InternLM2 分支处理
                image_data=image_data,  # 图像数据
                input_text=input_text,  # 输入文本
                request_obj=request_obj,  # 请求对象
                **kwargs,  # 其他参数
            )
        else:  # 否则
            # Default branch uses OpenAI-style placeholders
            return await self.process_qwen_mm_data_async(  # 使用 Qwen 风格（OpenAI 占位符）分支处理
                image_data=image_data,  # 图像数据
                input_text=input_text,  # 输入文本
                request_obj=request_obj,  # 请求对象
                **kwargs,  # 其他参数
            )

    async def process_qwen_mm_data_async(  # Qwen 风格的多模态数据处理，使用 OpenAI 风格占位符
        self, image_data, input_text, request_obj, **kwargs
    ):

        img_max_num = (  # 获取图像最大分块数
            getattr(request_obj, "image_max_dynamic_patch", None)  # 从请求获取
            or getattr(request_obj, "max_dynamic_patch", None)  # 尝试获取 max_dynamic_patch
            or kwargs.get("image_max_dynamic_patch")  # 从 kwargs 获取
            or kwargs.get("max_dynamic_patch")  # 尝试从 kwargs 获取
            or self.IMAGE_MAX_NUM  # 使用类默认值
        )
        img_max_num = max(1, int(img_max_num))  # 确保至少为 1

        vid_max_num = (  # 获取视频帧最大分块数
            getattr(request_obj, "video_max_dynamic_patch", None)  # 从请求获取
            or getattr(request_obj, "max_dynamic_patch", None)  # 尝试获取 max_dynamic_patch
            or kwargs.get("video_max_dynamic_patch")  # 从 kwargs 获取
            or kwargs.get("max_dynamic_patch")  # 尝试从 kwargs 获取
            or self.VIDEO_MAX_NUM  # 使用类默认值
        )
        vid_max_num = max(1, int(vid_max_num))  # 确保至少为 1

        # Qwen/Qwen3 branch: OpenAI-style placeholders <image>/<video>
        prompt = input_text or ""  # 获取输入文本
        video_data = getattr(request_obj, "video_data", None) or []  # 获取视频数据

        if image_data:  # 如果有图像数据
            prompt = self._ensure_placeholders_before_assistant(  # 确保图像占位符数量足够
                prompt, self.IMAGE_PLACEHOLDER_TOKEN, len(image_data)  # 提示文本、图像占位符、图像数量
            )
        if video_data:  # 如果有视频数据
            prompt = self._ensure_placeholders_before_assistant(  # 确保视频占位符数量足够
                prompt, self.VIDEO_PLACEHOLDER_TOKEN, len(video_data)  # 提示文本、视频占位符、视频数量
            )

        logger.info(  # 输出占位符数量日志
            "[internvl][qwen] placeholders image=%d video=%d",  # 图像和视频占位符数量
            prompt.count(self.IMAGE_PLACEHOLDER_TOKEN),  # 图像占位符计数
            prompt.count(self.VIDEO_PLACEHOLDER_TOKEN),  # 视频占位符计数
        )

        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=prompt,  # 输入提示文本
            image_data=image_data,  # 图像数据
            video_data=video_data,  # 视频数据
            multimodal_tokens=self.mm_tokens,  # expects <image>/<video>  # 期望 <image>/<video> 占位符
            discard_alpha_channel=True,  # 丢弃透明通道
        )

        logger.info(  # 输出加载结果日志
            "[internvl][qwen] loaded images=%d videos=%d",  # 加载的图像和视频数量
            len(base_output.images),  # 图像数量
            len(base_output.videos),  # 视频数量
        )

        mean, std = self._get_normalize_tensors(device=get_device())  # 获取归一化张量

        # ----- Images -> tiles -----
        num_patches_list: List[int] = []  # 每张图像的分块数列表
        pixel_values_list: List[torch.Tensor] = []  # 像素值列表

        for image in base_output.images:  # 遍历所有图像
            if isinstance(image, Image.Image):  # 如果是 PIL Image 对象
                img_np = np.array(image.convert("RGB"))  # 转换为 RGB 格式的 numpy 数组
                tensor = (  # 转换为 GPU 上的浮点张量
                    torch.from_numpy(img_np).permute(2, 0, 1).to(get_device()).float()
                    / 255.0  # 归一化到 [0, 1]
                )
            else:  # 否则假设已经是张量
                tensor = image.to(get_device())  # 移动到目标设备

            tensor = (tensor - mean) / std  # 使用 ImageNet 均值和标准差进行归一化
            tiles = self.dynamic_preprocess(  # 动态预处理图像，生成分块
                tensor, image_size=448, max_num=img_max_num, use_thumbnail=True  # 使用缩略图
            )
            pixel_values_list.append(tiles)  # 添加分块像素值
            num_patches_list.append(int(tiles.shape[0]))  # 添加分块数量

        if image_data and not pixel_values_list:  # 如果有图像数据但没有解析到图像
            raise ValueError(  # 抛出异常
                "[internvl][qwen] image_data provided but no images parsed from prompt placeholders"  # 提供了图像数据但从提示占位符中未解析到图像
            )

        image_tensor = (  # 拼接所有图像的分块像素值
            torch.cat(pixel_values_list, dim=0) if pixel_values_list else None  # 如果有像素值则拼接，否则为 None
        )

        # ----- Videos -> frame tiles (optional) -----
        video_tensor = None  # 初始化视频张量
        video_patch_lists = []  # 每个视频每帧的分块数列表
        video_pixel_values = []  # 视频像素值列表

        requested_frames = int(  # 获取请求的帧数
            kwargs.get("video_num_frames", self.DEFAULT_VIDEO_NUM_FRAMES)  # 从参数获取或使用默认值
        )
        num_frames = self._resolve_video_num_frames(  # 解析实际使用的帧数
            requested=requested_frames,  # 请求的帧数
            num_videos=len(base_output.videos),  # 视频数量
            text_len=self._token_len(base_output.input_text or prompt),  # 文本 token 长度
            image_tile_cnt=int(sum(num_patches_list)) if num_patches_list else 0,  # 图像分块总数
        )

        if base_output.videos and num_frames > 0 and self.video_token_id is not None:  # 如果有视频且帧数大于 0
            for video in base_output.videos:  # 遍历所有视频
                is_video_obj = isinstance(video, VideoDecoderWrapper)  # 检查是否为视频解码器对象
                vr = video if is_video_obj else self._open_video_reader(str(video))  # 获取视频读取器
                max_frame = len(vr) - 1  # 最大帧索引
                frame_indices = (  # 计算帧索引
                    [0]  # 如果只取 1 帧，取第一帧
                    if num_frames == 1  # 帧数为 1
                    else np.linspace(0, max_frame, num=num_frames, dtype=int).tolist()  # 否则均匀采样
                )

                per_video_tiles = []  # 当前视频的每帧分块
                per_video_patch_cnt = []  # 当前视频每帧的分块数
                for fi in frame_indices:  # 遍历帧索引
                    img_np = vr[int(fi)]  # 获取指定帧的 numpy 数组
                    frame_t = (  # 转换为 GPU 上的浮点张量
                        torch.from_numpy(img_np)
                        .permute(2, 0, 1)
                        .to(get_device())
                        .float()
                        / 255.0  # 归一化到 [0, 1]
                    )
                    frame_t = (frame_t - mean) / std  # 归一化

                    tiles = self.dynamic_preprocess(  # 动态预处理帧
                        frame_t,  # 帧张量
                        image_size=448,  # 分块大小
                        max_num=vid_max_num,  # 最大分块数
                        use_thumbnail=self.VIDEO_USE_THUMBNAIL,  # 是否使用缩略图
                    )
                    per_video_tiles.append(tiles)  # 添加帧分块
                    per_video_patch_cnt.append(int(tiles.shape[0]))  # 添加分块数

                pv = torch.cat(per_video_tiles, dim=0)  # 拼接当前视频所有帧的分块
                video_pixel_values.append(pv)  # 添加到视频像素值列表
                video_patch_lists.append(per_video_patch_cnt)  # 添加分块数列表

            video_tensor = (  # 拼接所有视频的像素值
                torch.cat(video_pixel_values, dim=0) if video_pixel_values else None  # 如果有则拼接，否则为 None
            )

        # ----- Build prompt text with <img> + CONTEXT*n + </img> -----
        img_ph = "<<<__IMG_PLACEHOLDER__>>>"  # 临时图像占位符
        vid_ph = "<<<__VID_PLACEHOLDER__>>>"  # 临时视频占位符

        input_text_mid = base_output.input_text or prompt  # 获取中间文本
        input_text_mid = input_text_mid.replace(self.IMAGE_PLACEHOLDER_TOKEN, img_ph)  # 替换图像占位符
        input_text_mid = input_text_mid.replace(self.IMG_CONTEXT, img_ph)  # 替换图像上下文标记

        if self.VIDEO_CONTEXT_TOKEN and self.video_token_id is not None:  # 如果支持视频
            input_text_mid = input_text_mid.replace(  # 替换视频占位符
                self.VIDEO_PLACEHOLDER_TOKEN, vid_ph  # 替换为临时视频占位符
            )
        else:  # 否则不支持视频
            input_text_mid = input_text_mid.replace(self.VIDEO_PLACEHOLDER_TOKEN, "")  # 移除视频占位符

        input_text_updated = input_text_mid  # 初始化更新后的文本

        # Expand images
        for num_patches in num_patches_list:  # 遍历每张图像的分块数
            image_tokens = (  # 构建展开后的图像标记
                self.IMG_START  # 图像起始标记
                + (self.IMG_CONTEXT * (self.num_image_token * int(num_patches)))  # 图像上下文标记 × token 数
                + self.IMG_END  # 图像结束标记
            )
            input_text_updated = input_text_updated.replace(img_ph, image_tokens, 1)  # 替换第一个图像占位符

        # Expand videos (each frame is one <img>...</img>)
        if video_patch_lists and self.VIDEO_CONTEXT_TOKEN:  # 如果有视频分块且支持视频标记
            for frame_patch_list in video_patch_lists:  # 遍历每个视频的分块列表
                frame_lines = []  # 初始化帧行列表
                for i, patch_cnt in enumerate(frame_patch_list):  # 遍历每帧的分块数
                    ctx_cnt = int(self.num_image_token) * int(patch_cnt)  # 计算上下文标记总数
                    frame_tokens = (  # 构建帧标记
                        self.IMG_START  # 图像起始标记
                        + (self.VIDEO_CONTEXT_TOKEN * ctx_cnt)  # 视频上下文标记 × 数量
                        + self.IMG_END  # 图像结束标记
                    )
                    frame_lines.append(f"Frame {i+1}: {frame_tokens}")  # 添加帧行
                video_tokens = "\n".join(frame_lines) + "\n"  # 用换行符连接所有帧行
                input_text_updated = input_text_updated.replace(vid_ph, video_tokens, 1)  # 替换第一个视频占位符

        # Tokenize
        input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")[  # 分词并获取 input_ids
            "input_ids"
        ].flatten()  # 展平为一维
        input_ids = input_ids_tensor.tolist()  # 转换为列表

        # Offsets
        image_offsets = []  # 初始化图像偏移量列表
        if image_tensor is not None:  # 如果有图像张量
            image_offsets = self.get_mm_items_offset(  # 获取图像项偏移量
                input_ids=input_ids_tensor.to(get_device()),  # 输入 ID 张量
                mm_token_id=self.img_context_token_id,  # 图像上下文标记 ID
            )

        video_offsets = []  # 初始化视频偏移量列表
        if video_tensor is not None and self.video_token_id is not None:  # 如果有视频张量且有视频标记 ID
            video_offsets = self.get_mm_items_offset(  # 获取视频项偏移量
                input_ids=input_ids_tensor.to(get_device()),  # 输入 ID 张量
                mm_token_id=self.video_token_id,  # 视频标记 ID
            )

        items = []  # 初始化多模态数据项列表
        if image_tensor is not None:  # 如果有图像张量
            # Split per-image for better cache granularity
            assert len(num_patches_list) == len(image_offsets), (  # 断言图像分块数和偏移量数量一致
                f"InternVL: num_patches_list ({len(num_patches_list)}) != "
                f"image_offsets ({len(image_offsets)})"
            )
            cumulative = 0  # 累计分块数
            for i, num_patches in enumerate(num_patches_list):  # 遍历每张图像
                items.append(  # 添加多模态数据项
                    MultimodalDataItem(  # 创建数据项
                        feature=image_tensor[cumulative : cumulative + num_patches],  # 当前图像的特征
                        modality=Modality.IMAGE,  # 图像模态
                        offsets=[image_offsets[i]],  # 偏移量
                    )
                )
                cumulative += num_patches  # 更新累计分块数
        if video_tensor is not None:  # 如果有视频张量
            items.append(  # 添加视频数据项
                MultimodalDataItem(
                    feature=video_tensor, modality=Modality.VIDEO, offsets=video_offsets  # 视频特征  # 视频模态  # 视频偏移量
                )
            )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids,  # 输入 ID 列表
            mm_items=items,  # 多模态数据项
            im_start_id=self.img_start_token_id,  # 图像起始标记 ID
            im_end_id=self.img_end_token_id,  # 图像结束标记 ID
            im_token_id=self.img_context_token_id,  # 图像上下文标记 ID
            video_token_id=self.video_token_id,  # 视频标记 ID
        )

    async def process_internlm2_mm_data_async(  # InternLM2 风格的多模态数据处理，使用 IMG_CONTEXT 占位符
        self, image_data, input_text, request_obj, **kwargs
    ):
        # InternLM2 branch: legacy placeholder <IMG_CONTEXT> (stable for InternLM2 prompt behavior)
        prompt = input_text or ""  # 获取输入文本
        video_data = getattr(request_obj, "video_data", None) or []  # 获取视频数据
        if video_data:  # 如果有视频数据
            logger.warning(  # 输出警告日志
                "[internvl][internlm2] video input ignored for InternLM2 branch"  # InternLM2 分支忽略视频输入
            )

        # Convert any OpenAI-style <image> into <IMG_CONTEXT>
        prompt = prompt.replace(self.IMAGE_PLACEHOLDER_TOKEN, self.IMG_CONTEXT)  # 将 OpenAI 风格占位符转换为 IMG_CONTEXT

        if image_data:  # 如果有图像数据
            prompt = self._ensure_placeholders_before_assistant(  # 确保 IMG_CONTEXT 占位符数量足够
                prompt, self.IMG_CONTEXT, len(image_data)  # 提示文本、图像上下文标记、图像数量
            )

        logger.info(  # 输出占位符数量日志
            "[internvl][internlm2] placeholders img_context=%d",  # 图像上下文占位符数量
            prompt.count(self.IMG_CONTEXT),  # IMG_CONTEXT 计数
        )

        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=prompt,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens_internlm2,  # expects <IMG_CONTEXT>  # 期望 IMG_CONTEXT 占位符
            discard_alpha_channel=True,  # 丢弃透明通道
        )

        mean, std = self._get_normalize_tensors(device=get_device())  # 获取归一化张量

        num_patches_list: List[int] = []  # 每张图像的分块数列表
        pixel_values_list: List[torch.Tensor] = []  # 像素值列表

        for image in base_output.images:  # 遍历所有图像
            if isinstance(image, Image.Image):  # 如果是 PIL Image 对象
                img_np = np.array(image.convert("RGB"))  # 转换为 RGB 格式的 numpy 数组
                tensor = (  # 转换为 GPU 上的浮点张量
                    torch.from_numpy(img_np).permute(2, 0, 1).to(get_device()).float()
                    / 255.0  # 归一化到 [0, 1]
                )
            else:  # 否则假设已经是张量
                tensor = image.to(get_device())  # 移动到目标设备

            tensor = (tensor - mean) / std  # 使用 ImageNet 均值和标准差进行归一化
            tiles = self.dynamic_preprocess(  # 动态预处理图像，生成分块
                tensor, image_size=448, max_num=12, use_thumbnail=True  # 使用缩略图
            )
            pixel_values_list.append(tiles)  # 添加分块像素值
            num_patches_list.append(int(tiles.shape[0]))  # 添加分块数量

        if image_data and not pixel_values_list:  # 如果有图像数据但没有解析到图像
            raise ValueError(  # 抛出异常
                "[internvl][internlm2] image_data provided but no images parsed from prompt placeholders"  # 提供了图像数据但从提示占位符中未解析到图像
            )

        pixel_values = (  # 拼接所有图像的分块像素值
            torch.cat(pixel_values_list, dim=0) if pixel_values_list else None  # 如果有像素值则拼接，否则为 None
        )

        # Expand each <IMG_CONTEXT> into <img> + <IMG_CONTEXT>*N + </img>
        ph = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"  # 临时占位符
        input_text_base = (base_output.input_text or prompt).replace(  # 替换 IMG_CONTEXT 为临时占位符
            self.IMG_CONTEXT, ph  # 替换
        )

        input_text_updated = input_text_base  # 初始化更新后的文本
        for num_patches in num_patches_list:  # 遍历每张图像的分块数
            image_tokens = (  # 构建展开后的图像标记
                self.IMG_START  # 图像起始标记
                + (self.IMG_CONTEXT * (self.num_image_token * int(num_patches)))  # 图像上下文标记 × token 数
                + self.IMG_END  # 图像结束标记
            )
            input_text_updated = input_text_updated.replace(ph, image_tokens, 1)  # 替换第一个占位符

        # Tokenize
        input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")[  # 分词并获取 input_ids
            "input_ids"
        ].flatten()  # 展平为一维
        input_ids = input_ids_tensor.tolist()  # 转换为列表

        # Offsets
        image_offsets = []  # 初始化图像偏移量列表
        if pixel_values is not None:  # 如果有像素值
            image_offsets = self.get_mm_items_offset(  # 获取图像项偏移量
                input_ids=input_ids_tensor.to(get_device()),  # 输入 ID 张量
                mm_token_id=self.img_context_token_id,  # 图像上下文标记 ID
            )

        items = []  # 初始化多模态数据项列表
        if pixel_values is not None:  # 如果有像素值
            # Split per-image for better cache granularity
            assert len(num_patches_list) == len(image_offsets), (  # 断言图像分块数和偏移量数量一致
                f"InternVL: num_patches_list ({len(num_patches_list)}) != "
                f"image_offsets ({len(image_offsets)})"
            )
            cumulative = 0  # 累计分块数
            for i, num_patches in enumerate(num_patches_list):  # 遍历每张图像
                items.append(  # 添加多模态数据项
                    MultimodalDataItem(  # 创建数据项
                        feature=pixel_values[cumulative : cumulative + num_patches],  # 当前图像的特征
                        modality=Modality.IMAGE,  # 图像模态
                        offsets=[image_offsets[i]],  # 偏移量
                    )
                )
                cumulative += num_patches  # 更新累计分块数

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids,  # 输入 ID 列表
            mm_items=items,  # 多模态数据项
            im_start_id=self.img_start_token_id,  # 图像起始标记 ID
            im_end_id=self.img_end_token_id,  # 图像结束标记 ID
            im_token_id=self.img_context_token_id,  # 图像上下文标记 ID
            video_token_id=self.video_token_id,  # 视频标记 ID
        )
