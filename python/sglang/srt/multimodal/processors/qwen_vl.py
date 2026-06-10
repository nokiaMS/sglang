# Qwen-VL多模态处理器模块
# 本模块为Qwen视觉语言系列模型提供图像、视频和音频数据处理功能
# 支持Qwen2-VL、Qwen2.5-VL、Qwen3-VL等多种模型
# 包含智能图像缩放、视频预处理和M-RoPE位置编码计算

import math  # 导入数学模块
import os  # 导入操作系统模块
import re  # 导入正则表达式模块
import time  # 导入时间模块
from typing import List, Optional, Union  # 导入类型提示

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import torchvision  # 导入torchvision
from PIL import Image  # 导入PIL图像处理库
from torchvision.transforms import InterpolationMode  # 导入插值模式枚举

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding  # 导入M-RoPE旋转嵌入
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalProcessorOutput,  # 多模态处理器输出
)
from sglang.srt.models.interns2preview import InternS2PreviewForConditionalGeneration  # 导入InternS2模型
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # 导入Qwen2.5-VL模型
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration  # 导入Qwen2-VL模型
from sglang.srt.models.qwen3_5 import (  # 导入Qwen3.5模型
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP  # 导入Qwen3.5-MTP模型
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration  # 导入Qwen3-Omni-MoE模型
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration  # 导入Qwen3-VL模型
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration  # 导入Qwen3-VL-MoE模型
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入多模态特殊标记类
    MultimodalSpecialTokens,
)
from sglang.srt.utils import cpu_has_amx_support, is_cpu  # 导入CPU检测工具
from sglang.srt.utils.video_decoder import VideoDecoderWrapper  # 导入视频解码器包装器
from sglang.utils import logger  # 导入日志记录器

IMAGE_FACTOR = 28  # 图像因子，用于图像尺寸对齐
MIN_PIXELS = 4 * 28 * 28  # 最小像素数
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()  # 最大像素数，从环境变量获取
MAX_RATIO = 200  # 最大宽高比
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)  # 获取重采样方法
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:  # 如果设置了但获取不到
    logger.warning(  # 输出警告
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "  # 无效的重采样值
        f"Ignoring and using default."  # 忽略并使用默认值
    )
VIDEO_TOTAL_PIXELS = int(  # 视频总像素数
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))  # 从环境变量获取或使用默认值
)

VIDEO_MIN_PIXELS = 128 * 28 * 28  # 视频最小像素数
VIDEO_MAX_PIXELS = 768 * 28 * 28  # 视频最大像素数
FRAME_FACTOR = 2  # 帧因子
FPS = 2.0  # 默认帧率
FPS_MIN_FRAMES = 4  # 最小帧数
FPS_MAX_FRAMES = 768  # 最大帧数


_is_cpu_amx_available = cpu_has_amx_support()  # 检查CPU是否支持AMX
_is_cpu = is_cpu()  # 检查是否为CPU
if _is_cpu and _is_cpu_amx_available:  # 如果是CPU且支持AMX
    try:  # 尝试加载AMX优化
        import transformers  # 导入transformers

        from sglang.srt.layers.amx_utils import fast_preprocess_cpu  # 导入CPU快速预处理

        transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess = (  # 替换Qwen2-VL图像处理器的预处理方法
            fast_preprocess_cpu  # 使用CPU快速预处理
        )
    except Exception as e:  # 捕获异常
        logger.warning(  # 输出警告
            f"Failed to hack Qwen2VLImageProcessorFast with AMX optimization: {e}"  # AMX优化替换失败
        )


def smart_resize(  # 智能缩放图像尺寸，使其满足条件
    height: int,  # 图像高度
    width: int,  # 图像宽度
    factor: int = IMAGE_FACTOR,  # 对齐因子
    min_pixels: int = MIN_PIXELS,  # 最小像素数
    max_pixels: int = MAX_PIXELS,  # 最大像素数
) -> tuple[int, int]:  # 返回缩放后的宽高
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:  # 检查宽高比是否超过限制
        raise ValueError(  # 抛出异常
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"  # 宽高比超过限制
        )
    h_bar = max(factor, round_by_factor(height, factor))  # 将高度对齐到因子
    w_bar = max(factor, round_by_factor(width, factor))  # 将宽度对齐到因子
    if h_bar * w_bar > max_pixels:  # 如果超过最大像素数
        beta = math.sqrt((height * width) / max_pixels)  # 计算缩放因子
        h_bar = floor_by_factor(height / beta, factor)  # 向下对齐高度
        w_bar = floor_by_factor(width / beta, factor)  # 向下对齐宽度
    elif h_bar * w_bar < min_pixels:  # 如果低于最小像素数
        beta = math.sqrt(min_pixels / (height * width))  # 计算缩放因子
        h_bar = ceil_by_factor(height * beta, factor)  # 向上对齐高度
        w_bar = ceil_by_factor(width * beta, factor)  # 向上对齐宽度
    return h_bar, w_bar  # 返回缩放后的尺寸


def round_by_factor(number: int, factor: int) -> int:  # 按因子四舍五入
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor  # 四舍五入到最接近的因子倍数


def ceil_by_factor(number: int, factor: int) -> int:  # 按因子向上取整
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor  # 向上取整到因子倍数


def floor_by_factor(number: int, factor: int) -> int:  # 按因子向下取整
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor  # 向下取整到因子倍数


def smart_nframes(  # 智能计算视频帧数
    ele: dict,  # 视频配置字典
    total_frames: int,  # 视频总帧数
    video_fps: int | float,  # 视频帧率
) -> int:  # 返回提取的帧数
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not (  # 断言不能同时指定fps和nframes
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"  # 只接受fps或nframes其中之一
    if "nframes" in ele:  # 如果指定了帧数
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)  # 按因子对齐帧数
    else:  # 否则按帧率计算
        fps = ele.get("fps", FPS)  # 获取帧率，默认为2.0
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)  # 获取最小帧数
        max_frames = floor_by_factor(  # 获取最大帧数
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR  # 最大帧数不超过视频总帧数
        )
        nframes = total_frames / video_fps * fps  # 计算提取帧数
        if nframes > total_frames:  # 如果计算的帧数超过总帧数
            logger.warning(  # 输出警告
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"  # 提取帧数超过总帧数
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)  # 限制帧数在有效范围内
        nframes = floor_by_factor(nframes, FRAME_FACTOR)  # 按因子向下对齐帧数
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):  # 检查帧数是否在有效范围内
        raise ValueError(  # 抛出异常
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."  # 帧数不在有效范围
        )
    return nframes  # 返回帧数


# process video, qwen-specific
async def preprocess_video(  # 异步预处理视频，Qwen专用
    vr,  # 视频读取器或张量
    image_factor: int = IMAGE_FACTOR,  # 图像因子
    video_config: dict = {},  # 视频配置
) -> torch.Tensor:  # 返回视频张量和元数据
    # preprocessed video
    is_video_obj = isinstance(vr, VideoDecoderWrapper)  # 检查是否为视频解码器对象
    if not is_video_obj:  # 如果不是视频对象
        return vr, None  # 直接返回原始数据
    entry_time = time.perf_counter()  # 记录开始时间

    total_frames, video_fps = len(vr), vr.avg_fps  # 获取总帧数和帧率

    nframes = smart_nframes(  # 智能计算提取帧数
        video_config, total_frames=total_frames, video_fps=video_fps  # 传入配置和视频信息
    )
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)  # 均匀采样帧索引
    idx = np.unique(idx)  # 去重

    video = vr.get_frames_as_tensor(idx.tolist())  # 获取指定帧的张量

    video = video.permute(0, 3, 1, 2)  # NHWC -> TCHW，调整维度顺序

    nframes, _, height, width = video.shape  # 获取视频尺寸
    min_pixels = video_config.get("min_pixels", VIDEO_MIN_PIXELS)  # 获取最小像素数
    total_pixels = video_config.get("total_pixels", VIDEO_TOTAL_PIXELS)  # 获取总像素数
    max_pixels = max(  # 计算最大像素数
        min(  # 取两者较小值
            video_config.get("max_pixels", VIDEO_MAX_PIXELS),  # 配置中的最大像素数
            total_pixels / nframes * FRAME_FACTOR,  # 按帧数分配的像素数
        ),
        int(min_pixels * 1.05),  # 至少为最小像素数的1.05倍
    )

    get_batch_time = time.perf_counter()  # 记录批次获取时间

    max_pixels_supposed = video_config.get("max_pixels", max_pixels)  # 获取配置的最大像素数

    if max_pixels_supposed > max_pixels:  # 如果配置值超过计算值
        logger.warning(  # 输出警告
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."  # 最大像素数超过限制
        )
    max_pixels = min(max_pixels_supposed, max_pixels)  # 取两者较小值
    if "resized_height" in video_config and "resized_width" in video_config:  # 如果指定了缩放尺寸
        resized_height, resized_width = smart_resize(  # 使用指定尺寸进行智能缩放
            video_config["resized_height"],  # 指定高度
            video_config["resized_width"],  # 指定宽度
            factor=image_factor,  # 图像因子
        )
    else:  # 否则自动计算
        resized_height, resized_width = smart_resize(  # 智能计算缩放尺寸
            height,  # 原始高度
            width,  # 原始宽度
            factor=image_factor,  # 图像因子
            min_pixels=min_pixels,  # 最小像素数
            max_pixels=max_pixels,  # 最大像素数
        )
    smart_resize_time = time.perf_counter()  # 记录智能缩放完成时间
    video = torchvision.transforms.functional.resize(  # 缩放视频
        video,  # 视频张量
        [resized_height, resized_width],  # 目标尺寸
        interpolation=InterpolationMode.BILINEAR,  # 双线性插值
    )
    video = video.pin_memory()  # 固定内存以提高传输效率
    video_metadata = {  # 创建视频元数据
        "fps": video_fps,  # 帧率
        "duration": total_frames / video_fps,  # 时长
        "total_num_frames": total_frames,  # 总帧数
        "frames_indices": idx,  # 采样的帧索引
        "video_backend": "torchvision",  # 视频后端
    }
    torchvision_resize_time = time.perf_counter()  # 记录torchvision缩放完成时间
    logger.debug(  # 输出调试信息
        f"[preprocess_video Perf], "  # 视频预处理性能
        f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms, "  # 获取批次耗时
        f"smart_resize_time: {(smart_resize_time - get_batch_time) * 1000:.2f} ms, "  # 智能缩放耗时
        f"torchvision_resize_time: {(torchvision_resize_time - smart_resize_time) * 1000:.2f} ms, "  # torchvision缩放耗时
        f"total_time: {(torchvision_resize_time - entry_time) * 1000:.2f} ms"  # 总耗时
    )
    return video, video_metadata  # 返回视频张量和元数据


# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(SGLangBaseProcessor):  # Qwen-VL图像处理器，兼容Qwen-VL和Qwen-Omni系列
    supports_transformers_backend = True  # 支持transformers后端
    models = [  # 关联的模型列表
        Qwen2VLForConditionalGeneration,  # Qwen2-VL模型
        Qwen2_5_VLForConditionalGeneration,  # Qwen2.5-VL模型
        Qwen3VLForConditionalGeneration,  # Qwen3-VL模型
        Qwen3VLMoeForConditionalGeneration,  # Qwen3-VL-MoE模型
        Qwen3_5ForConditionalGeneration,  # Qwen3.5模型
        Qwen3_5MoeForConditionalGeneration,  # Qwen3.5-MoE模型
        Qwen3_5ForCausalLMMTP,  # Qwen3.5-MTP模型
        InternS2PreviewForConditionalGeneration,  # InternS2预览模型
        Qwen3OmniMoeForConditionalGeneration,  # Qwen3-Omni-MoE模型
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Qwen-VL图像处理器
        self.model_type = hf_config.model_type  # 获取模型类型
        if hf_config.model_type == "qwen3_omni_moe":  # 如果是Qwen3-Omni-MoE
            hf_config = hf_config.thinker_config  # 使用思考器配置

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id  # 获取图像开始标记ID
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id  # 获取图像结束标记ID
        self.IM_TOKEN_ID = hf_config.image_token_id  # 获取图像标记ID
        self.VIDEO_TOKEN_ID = hf_config.video_token_id  # 获取视频标记ID

        self.vision_start_token_id = hf_config.vision_start_token_id  # 视觉起始标记ID
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)  # 视觉结束标记ID

        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)  # 音频起始标记ID
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)  # 音频标记ID

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token="<|vision_start|><|image_pad|><|vision_end|>",  # 图像标记
            image_token_id=hf_config.image_token_id,  # 图像标记ID
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(  # 图像标记正则表达式
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"  # 匹配扩展的图像标记
            ),
            video_token_id=self.VIDEO_TOKEN_ID,  # 视频标记ID
            audio_token_id=self.audio_token_id,  # 音频标记ID
        ).build(_processor)  # 构建标记对象

    def build_input_ids_with_timestamps(  # 构建带时间戳的输入ID
        self, prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps  # 提示、嵌入、图像网格、视频网格、视频时间戳
    ):
        """
        Build input_ids with timestamps for qwen3_vl models.
        """
        if not isinstance(prompt, list):  # 如果提示不是列表
            prompt = self._processor.tokenizer.encode(prompt)  # 编码为标记ID列表

        img_token_id = getattr(self, "IM_TOKEN_ID", None)  # 获取图像标记ID
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)  # 获取视频标记ID
        spatial_merge_size = getattr(self, "spatial_merge_size", 1)  # 获取空间合并尺寸
        vision_start_token_id = getattr(self, "vision_start_token_id", None)  # 获取视觉起始标记ID
        vision_end_token_id = getattr(self, "vision_end_token_id", None)  # 获取视觉结束标记ID

        input_ids = []  # 输入ID列表
        offsets = []  # 偏移量列表
        modality_list = []  # 模态列表
        cur_idx = 0  # 当前索引

        vision_start_indices = []  # 视觉起始索引列表
        for i in range(len(prompt) - 1):  # 遍历提示标记
            if img_token_id is not None and prompt[i + 1] == img_token_id:  # 如果下一个标记是图像标记
                vision_start_indices.append((i, Modality.IMAGE))  # 添加图像模态索引
            elif video_token_id is not None and prompt[i + 1] == video_token_id:  # 如果下一个标记是视频标记
                vision_start_indices.append((i, Modality.VIDEO))  # 添加视频模态索引

        img_idx = 0  # 图像索引
        video_idx = 0  # 视频索引
        for mm_start_idx, modality in vision_start_indices:  # 遍历视觉起始索引
            modality_list.append(modality)  # 添加模态
            video_tokens = None  # 视频标记初始化
            if modality == Modality.IMAGE:  # 如果是图像模态
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)  # 计算图像标记数
                mm_token_id = img_token_id  # 使用图像标记ID
                img_idx += 1  # 递增图像索引
            elif modality == Modality.VIDEO:  # 如果是视频模态
                curr_timestamps = video_timestamps[video_idx]  # 获取当前视频时间戳
                num_frames = video_grid_thw[video_idx][0]  # 获取视频帧数
                frame_seqlen = video_grid_thw[video_idx][1:].prod().item() // (  # 计算每帧标记数
                    spatial_merge_size**2
                )
                video_tokens = []  # 视频标记列表
                _current_offset = len(input_ids) + mm_start_idx + 1 - cur_idx  # 计算当前偏移
                # take single frame as one mm_item
                for frame_idx in range(num_frames):  # 遍历每帧
                    if frame_idx > 0:  # 如果不是第一帧
                        modality_list.append(Modality.VIDEO)  # 添加视频模态
                    curr_time = curr_timestamps[frame_idx]  # 获取当前帧时间戳
                    timestamp_text = f"<{curr_time:.1f} seconds>"  # 生成时间戳文本
                    timestamp_tokens = self._processor.tokenizer.encode(  # 编码时间戳文本
                        timestamp_text, add_special_tokens=False  # 不添加特殊标记
                    )
                    video_tokens.extend(timestamp_tokens)  # 添加时间戳标记
                    _current_offset += len(timestamp_tokens)  # 更新偏移
                    if vision_start_token_id is not None:  # 如果存在视觉起始标记
                        video_tokens.append(vision_start_token_id)  # 添加视觉起始标记
                        _current_offset += 1  # 更新偏移
                    video_tokens.extend([video_token_id] * frame_seqlen)  # 添加视频帧标记
                    if vision_end_token_id is not None:  # 如果存在视觉结束标记
                        video_tokens.append(vision_end_token_id)  # 添加视觉结束标记
                    offsets.append(  # 添加偏移量
                        (_current_offset, _current_offset + frame_seqlen - 1)  # 帧标记的起止位置
                    )
                    _current_offset += (  # 更新偏移
                        frame_seqlen + 1  # 帧标记数加1
                        if vision_end_token_id is not None  # 如果有视觉结束标记
                        else frame_seqlen  # 否则只加帧标记数
                    )  # for vision_end_token_id
                mm_token_num = len(video_tokens)  # 视频标记总数
                mm_token_id = None  # 不使用单一标记ID
                video_idx += 1  # 递增视频索引
            else:  # 其他模态
                logger.warning(  # 输出警告
                    f"{modality} modality is not supported for qwen3_vl models with timestamps."  # 不支持该模态
                )
                continue  # 跳过
            assert cur_idx <= mm_start_idx  # 断言当前索引不超过多模态起始索引
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])  # 添加提示标记
            if modality == Modality.VIDEO:  # 如果是视频模态
                input_ids.extend(video_tokens)  # 添加视频标记
            else:  # 图像模态
                mm_offset_start = len(input_ids)  # 记录标记起始位置
                input_ids.extend([mm_token_id] * mm_token_num)  # 添加图像标记
                offsets.append((mm_offset_start, len(input_ids) - 1))  # 添加偏移量
            cur_idx = mm_start_idx + 2  # jump to vision_end_id，跳到视觉结束标记后
        else:  # 遍历完成后
            input_ids.extend(prompt[cur_idx:])  # 添加剩余提示标记

        return input_ids, offsets, modality_list  # 返回输入ID、偏移量和模态列表

    def compute_mrope_positions(self, input_ids, mm_items):  # 计算M-RoPE位置编码
        image_grid_thw = self._concat_mm_item_grid(  # 拼接图像网格
            mm_items, "image_grid_thw", Modality.IMAGE  # 多模态数据项和键名
        )
        video_grid_thw = self._concat_mm_item_grid(  # 拼接视频网格
            mm_items, "video_grid_thw", Modality.VIDEO  # 多模态数据项和键名
        )

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # 转换输入ID为张量
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(  # 计算M-RoPE位置索引
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,  # 空间合并尺寸
            image_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记ID
            vision_start_token_id=self.vision_start_token_id,  # 视觉起始标记ID
            model_type=self.model_type,  # 模型类型
            tokens_per_second=getattr(  # 每秒标记数
                self.hf_config.vision_config, "tokens_per_second", None  # 从配置获取
            ),
            input_ids=input_ids_tensor,  # 输入ID张量
            image_grid_thw=image_grid_thw,  # 图像网格
            video_grid_thw=video_grid_thw,  # 视频网格
        )
        return mrope_positions.squeeze(1), mrope_position_delta  # 返回位置和增量

    @staticmethod
    def _get_processor_output_value(ret, key):  # 从处理器输出中获取指定键的值
        if ret is None:  # 如果输出为None
            return None  # 返回None
        return ret.get(key) if hasattr(ret, "get") else getattr(ret, key, None)  # 尝试字典或属性方式获取

    def _get_precomputed_mrope_from_output(self, ret):  # 从处理器输出中获取预计算的M-RoPE位置
        mrope_positions = self._get_processor_output_value(ret, "mrope_positions")  # 获取M-RoPE位置
        mrope_position_delta = self._get_processor_output_value(  # 获取M-RoPE位置增量
            ret, "mrope_position_delta"
        )
        if mrope_positions is None or mrope_position_delta is None:  # 如果任一为None
            return None  # 返回None

        mrope_positions = torch.as_tensor(mrope_positions)  # 转换为张量
        if mrope_positions.ndim == 3:  # 如果是3维
            if mrope_positions.shape[1] != 1:  # 如果第2维不为1
                return None  # 返回None
            mrope_positions = mrope_positions.squeeze(1)  # 压缩第2维
        if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:  # 检查形状
            return None  # 返回None

        mrope_position_delta = torch.as_tensor(mrope_position_delta)  # 转换为张量
        if mrope_position_delta.ndim <= 1:  # 如果维度不大于1
            mrope_position_delta = mrope_position_delta.reshape(-1, 1)  # 重塑为列向量
        return mrope_positions, mrope_position_delta  # 返回位置和增量

    @staticmethod
    def _as_grid_batch(value):  # 将网格值转换为批次格式
        if value is None:  # 如果值为None
            return None  # 返回None
        if isinstance(value, torch.Tensor):  # 如果是张量
            return value.unsqueeze(0) if value.ndim == 1 else value  # 1维则增加批次维度
        tensor = torch.as_tensor(value, dtype=torch.long)  # 转换为长整型张量
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor  # 1维则增加批次维度

    def _compute_image_only_mrope_positions_from_offsets(  # 从偏移量计算仅图像的M-RoPE位置
        self,
        input_len: int,  # 输入长度
        mm_items: List[MultimodalDataItem],  # 多模态数据项列表
        dtype: torch.dtype,  # 数据类型
        device: torch.device,  # 设备
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:  # 返回位置和增量，或None
        """instead of calling get_rope_index, build mrope position from mm_items.offsets and image_grid_thw of each image
        basically a simplified version of get_rope_index for image-only reqs
        """
        if self.model_type not in (  # 检查模型类型是否支持
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
        ):
            return None  # 不支持则返回None

        image_items = [item for item in mm_items if item.is_image()]  # 筛选图像数据项
        if not image_items or len(image_items) != len(mm_items):  # 如果没有图像或包含其他模态
            return None  # 返回None

        spatial_merge_size = self.hf_config.vision_config.spatial_merge_size  # 获取空间合并尺寸
        sorted_items = sorted(image_items, key=lambda item: item.offsets[0][0])  # 按偏移量排序
        position_segments = []  # 位置段列表
        st = 0  # 起始位置
        next_pos = 0  # 下一个位置

        for item in sorted_items:  # 遍历排序后的图像项
            if item.offsets is None or len(item.offsets) != 1:  # 检查偏移量有效性
                return None  # 返回None

            start, end = item.offsets[0]  # 获取起止位置
            if start < st or end >= input_len:  # 检查位置有效性
                return None  # 返回None

            text_len = start - st  # 计算文本长度
            if text_len > 0:  # 如果存在文本
                position_segments.append(  # 添加文本位置段
                    torch.arange(text_len, dtype=dtype, device=device)  # 生成位置序列
                    .view(1, -1)  # 重塑为2维
                    .expand(3, -1)  # 扩展到3维
                    + next_pos  # 加上偏移
                )
                next_pos += text_len  # 更新下一个位置

            grid = self._as_grid_batch(item.model_specific_data.get("image_grid_thw"))  # 获取图像网格
            if grid is None or grid.shape[0] != 1:  # 检查网格有效性
                return None  # 返回None
            t, h, w = [int(x) for x in grid[0].tolist()]  # 解析网格维度
            llm_grid_t = t  # LLM时间维度
            llm_grid_h = h // spatial_merge_size  # LLM高度维度
            llm_grid_w = w // spatial_merge_size  # LLM宽度维度
            num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w  # 计算图像标记数
            if num_image_tokens != end - start + 1:  # 检查标记数是否匹配
                return None  # 返回None

            t_index = (  # 生成时间索引
                torch.arange(llm_grid_t, dtype=dtype, device=device)  # 时间范围
                .view(-1, 1)  # 重塑
                .expand(llm_grid_t, llm_grid_h * llm_grid_w)  # 扩展
                .reshape(-1)  # 展平
            )
            h_index = (  # 生成高度索引
                torch.arange(llm_grid_h, dtype=dtype, device=device)  # 高度范围
                .view(1, -1, 1)  # 重塑
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)  # 扩展
                .reshape(-1)  # 展平
            )
            w_index = (  # 生成宽度索引
                torch.arange(llm_grid_w, dtype=dtype, device=device)  # 宽度范围
                .view(1, 1, -1)  # 重塑
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)  # 扩展
                .reshape(-1)  # 展平
            )
            position_segments.append(  # 添加图像位置段
                torch.stack([t_index, h_index, w_index]) + next_pos  # 堆叠索引并加上偏移
            )
            next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)  # 更新下一个位置
            st = end + 1  # 更新起始位置

        if st < input_len:  # 如果还有剩余文本
            text_len = input_len - st  # 计算剩余文本长度
            position_segments.append(  # 添加剩余文本位置段
                torch.arange(text_len, dtype=dtype, device=device)  # 生成位置序列
                .view(1, -1)  # 重塑
                .expand(3, -1)  # 扩展到3维
                + next_pos  # 加上偏移
            )

        mrope_positions = torch.cat(position_segments, dim=1).unsqueeze(1)  # 拼接所有位置段
        mrope_position_delta = (mrope_positions.max() + 1 - input_len).reshape(1, 1)  # 计算位置增量
        return mrope_positions, mrope_position_delta  # 返回位置和增量

    @classmethod
    def _concat_mm_item_grid(cls, mm_items: list[MultimodalDataItem], key, modality):  # 拼接多模态数据项的网格信息
        grids = []  # 网格列表
        for item in mm_items:  # 遍历数据项
            if not item.is_modality(modality):  # 如果模态不匹配
                continue  # 跳过
            grid = cls._as_grid_batch(item.model_specific_data.get(key))  # 获取网格数据
            if grid is not None:  # 如果网格数据不为空
                grids.append(grid)  # 添加到列表
        if not grids:  # 如果没有网格数据
            return None  # 返回None
        if len(grids) == 1:  # 如果只有一个网格
            return grids[0]  # 直接返回
        return torch.cat(grids, dim=0)  # 拼接所有网格

    @classmethod
    def _get_grid_from_output_or_items(  # 从输出或数据项中获取网格信息
        cls, ret, mm_items, key, modality, input_data=None  # 输出、数据项、键名、模态、输入数据
    ):
        grid = cls._get_processor_output_value(ret, key)  # 从输出中获取
        if grid is None:  # 如果输出中没有
            grid = cls._concat_mm_item_grid(mm_items, key, modality)  # 从数据项中获取
        if grid is None and input_data and isinstance(input_data[0], dict):  # 如果还没找到且输入数据是字典
            grid = input_data[0].get(key)  # 从输入数据中获取
        return grid  # 返回网格

    def get_mm_data(self, prompt, embeddings, **kwargs):  # 获取多模态数据（用于transformers后端）
        img_grid_thw = kwargs.get("img_grid_thw", None)  # 获取图像网格
        video_grid_thw = kwargs.get("video_grid_thw", None)  # 获取视频网格
        audio_feature_lens = kwargs.get("audio_feature_lens", None)  # 获取音频特征长度
        video_timestamps = kwargs.get("video_timestamps", None)  # 获取视频时间戳
        second_per_grid_ts = kwargs.get("second_per_grid_ts", None)  # 获取每网格秒数

        audio_seq_lens = None  # 音频序列长度
        if audio_feature_lens is not None:  # 如果有音频特征长度
            if self.model_type == "qwen3_omni_moe":  # Qwen3-Omni-MoE模型
                # apply _get_feat_extract_lengths to get seq_lens
                input_lengths_leave = audio_feature_lens % 100  # 计算输入长度余数
                feat_lengths = (input_lengths_leave - 1) // 2 + 1  # 计算特征长度
                audio_seq_lens = (  # 计算音频序列长度
                    ((feat_lengths - 1) // 2 + 1 - 1) // 2
                    + 1
                    + (audio_feature_lens // 100) * 13
                )
            elif self.model_type == "qwen2_5_omni":  # Qwen2.5-Omni模型
                audio_seq_lens = (audio_feature_lens - 1) // 2 + 1  # 计算音频序列长度
                audio_seq_lens = (audio_seq_lens - 2) // 2 + 1  # 二次转换

        if (  # 检查是否需要使用时间戳构建
            self.model_type
            in [
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
                "intern_s2_preview",
            ]
            and video_timestamps is not None  # 且有视频时间戳
        ):
            input_ids, offsets, modality_list = self.build_input_ids_with_timestamps(  # 使用时间戳构建输入ID
                prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps  # 传入参数
            )
        else:  # 不使用时间戳
            input_ids, offsets, modality_list = self.build_input_ids(  # 构建输入ID
                prompt, img_grid_thw, video_grid_thw, audio_seq_lens=audio_seq_lens  # 传入参数
            )
        assert all(isinstance(modality, Modality) for modality in modality_list)  # 断言所有模态都是Modality类型

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(  # 计算M-RoPE位置
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,  # 空间合并尺寸
            image_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记ID
            vision_start_token_id=self.vision_start_token_id,  # 视觉起始标记ID
            model_type=self.model_type,  # 模型类型
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),  # 输入ID张量
            image_grid_thw=img_grid_thw,  # 图像网格
            video_grid_thw=video_grid_thw,  # 视频网格
            second_per_grid_ts=second_per_grid_ts,  # 每网格秒数
            use_audio_in_video=False,  # 不使用音频在视频中
            audio_seqlens=(  # 音频序列长度
                audio_feature_lens if self.model_type == "qwen3_omni_moe" else None  # 仅Qwen3-Omni-MoE
            ),
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),  # 音频标记ID
            audio_start_token_id=self.audio_start_token_id,  # 音频起始标记ID
            position_id_per_seconds=getattr(  # 每秒位置ID
                self.hf_config, "position_id_per_seconds", None
            ),
            tokens_per_second=getattr(  # 每秒标记数
                self.hf_config.vision_config, "tokens_per_second", None
            ),
        )
        mrope_positions = mrope_positions.squeeze(1)  # 压缩维度

        mm_items = []  # 多模态数据项列表
        consumed_per_modality = {}  # 每种模态已消耗的嵌入数量

        for modality, offset in zip(modality_list, offsets):  # 遍历模态和偏移量
            num_tokens = offset[1] - offset[0] + 1  # 计算标记数量
            embedding_start = consumed_per_modality.get(modality, 0)  # 获取嵌入起始位置
            embedding_slice = embeddings[modality][  # 获取嵌入切片
                embedding_start : embedding_start + num_tokens  # 范围切片
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens  # 更新已消耗数量
            mm_items.append(  # 添加多模态数据项
                MultimodalDataItem(  # 创建数据项
                    modality=modality,  # 模态类型
                    offsets=[offset],  # 偏移量
                    precomputed_embeddings=embedding_slice,  # 预计算嵌入
                )
            )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids,  # 输入ID
            mm_items=mm_items,  # 多模态数据项
            im_start_id=self.IM_START_TOKEN_ID,  # 图像起始标记ID
            im_end_id=self.IM_END_TOKEN_ID,  # 图像结束标记ID
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记ID
            audio_token_id=self.mm_tokens.audio_token_id,  # 音频标记ID
            mrope_positions=mrope_positions,  # M-RoPE位置
            mrope_position_delta=mrope_position_delta,  # M-RoPE位置增量
        )

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        entry_time = time.perf_counter()  # 记录入口时间
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            video_data=request_obj.video_data,  # 视频数据
            audio_data=request_obj.audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
        )
        load_time = time.perf_counter()  # 记录加载完成时间
        rid = getattr(request_obj, "rid", "anonymous_rid")  # 获取请求ID

        video_metadata = None  # 视频元数据
        if base_output.videos and not isinstance(base_output.videos[0], dict):  # 如果有视频且不是字典格式
            videos_processed = [  # 预处理所有视频
                await preprocess_video(video, video_config=self.video_config)  # 异步预处理每个视频
                for video in base_output.videos  # 遍历视频列表
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))  # 分离视频和元数据

        preprocess_time = time.perf_counter()  # 记录预处理完成时间

        # NOTE: for qwen3-vl, video_meta need to be passed in, since do_sample_frames is already done in preprocess_video
        if self.hf_config.model_type in (  # 如果是qwen3-vl系列模型
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
        ):
            mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
                base_output,  # 基础输出
                self.mm_tokens,  # 多模态标记
                video_metadata=video_metadata,  # 视频元数据
                do_sample_frames=False,  # 不再采样帧（已在预处理中完成）
            )
        else:  # 其他模型
            mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
                base_output, self.mm_tokens  # 基础输出和多模态标记
            )

        audio_feature_lengths = None  # 音频特征长度初始化

        if self.model_type == "qwen3_omni_moe":  # Qwen3-Omni-MoE模型
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)  # 找到音频数据项
            if audio_item:  # 如果存在音频数据项
                audio_feature_lengths = torch.sum(  # 计算音频特征长度
                    audio_item.feature_attention_mask, dim=1  # 按维度1求和
                )

        second_per_grid_ts = self._get_processor_output_value(ret, "second_per_grid_ts")  # 获取每网格秒数
        if second_per_grid_ts is None:  # 如果没找到
            second_per_grid_ts = self._get_processor_output_value(  # 尝试备用键名
                ret, "video_second_per_grid"
            )

        process_time = time.perf_counter()  # 记录处理完成时间

        input_ids = input_ids.flatten()  # 展平输入ID
        base_input_ids = getattr(base_output, "input_ids", None)  # 获取基础输入ID
        if (  # 如果基础输入ID可用
            isinstance(base_input_ids, list)
            and len(base_input_ids) == input_ids.numel()
        ):
            # reuse preprocess input if it already carries list of input_ids
            input_ids_list = base_input_ids  # 复用预处理输入
        else:  # 否则转换
            input_ids_list = input_ids.tolist()  # 转换为列表

        # look for if padded_input_ids already exists before computing
        padded_input_ids = self._get_processor_output_value(ret, "padded_input_ids")  # 获取填充输入ID
        if padded_input_ids is None:  # 如果不存在
            padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(  # 构建填充输入ID
                input_ids_list, mm_items  # 输入ID列表和多模态数据项
            )
        elif isinstance(padded_input_ids, torch.Tensor):  # 如果是张量
            # reuse existing padded_input_ids
            padded_input_ids = padded_input_ids.flatten().tolist()  # 展平并转为列表
        else:  # 其他类型
            padded_input_ids = list(padded_input_ids)  # 转为列表

        image_grid_thw = self._get_grid_from_output_or_items(  # 获取图像网格
            ret, mm_items, "image_grid_thw", Modality.IMAGE, image_data  # 输出、数据项、键名、模态、输入
        )
        video_grid_thw = self._get_grid_from_output_or_items(  # 获取视频网格
            ret,
            mm_items,
            "video_grid_thw",
            Modality.VIDEO,
            request_obj.video_data,  # 视频输入数据
        )

        mrope_result = self._get_precomputed_mrope_from_output(ret)  # 尝试获取预计算的M-RoPE
        if mrope_result is None:  # 如果没有预计算结果
            if (  # 如果只有图像没有视频和音频
                video_grid_thw is None
                and second_per_grid_ts is None
                and audio_feature_lengths is None
            ):
                mrope_result = self._compute_image_only_mrope_positions_from_offsets(  # 使用简化方法计算
                    input_len=input_ids.numel(),  # 输入长度
                    mm_items=mm_items,  # 多模态数据项
                    dtype=input_ids.dtype,  # 数据类型
                    device=input_ids.device,  # 设备
                )
        if mrope_result is None:  # 如果仍未获得结果
            mrope_result = MRotaryEmbedding.get_rope_index(  # 使用完整方法计算
                spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,  # 空间合并尺寸
                image_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
                video_token_id=self.mm_tokens.video_token_id,  # 视频标记ID
                vision_start_token_id=self.vision_start_token_id,  # 视觉起始标记ID
                model_type=self.model_type,  # 模型类型
                tokens_per_second=getattr(  # 每秒标记数
                    self.hf_config.vision_config, "tokens_per_second", None
                ),
                # use the expanded token ids
                input_ids=input_ids.unsqueeze(0),  # 使用扩展的标记ID
                image_grid_thw=image_grid_thw,  # 图像网格
                video_grid_thw=video_grid_thw,  # 视频网格
                second_per_grid_ts=second_per_grid_ts,  # 每网格秒数
                use_audio_in_video=False,  # 不使用音频在视频中
                audio_seqlens=audio_feature_lengths,  # 音频序列长度
                audio_token_id=getattr(self.hf_config, "audio_token_id", None),  # 音频标记ID
                audio_start_token_id=self.audio_start_token_id,  # 音频起始标记ID
                position_id_per_seconds=getattr(  # 每秒位置ID
                    self.hf_config, "position_id_per_seconds", None
                ),
            )

        mrope_positions, mrope_position_delta = mrope_result  # 解包结果
        if mrope_positions.ndim == 3:  # 如果是3维
            mrope_positions = mrope_positions.squeeze(1)  # 压缩维度
        get_rope_index_time = time.perf_counter()  # 记录RoPE计算完成时间
        logger.debug(  # 输出调试信息
            f"[QwenVLProcessor Perf] {rid=}, "  # 请求ID
            f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "  # 加载耗时
            f"preprocess_time: {(preprocess_time - load_time) * 1000:.2f} ms, "  # 预处理耗时
            f"process_time: {(process_time - preprocess_time) * 1000:.2f} ms, "  # 处理耗时
            f"get_rope_index_time: {(get_rope_index_time - process_time) * 1000:.2f} ms, "  # RoPE计算耗时
            f"total_time: {(get_rope_index_time - entry_time) * 1000:.2f} ms"  # 总耗时
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids_list,  # 输入ID列表
            padded_input_ids=padded_input_ids,  # 填充输入ID
            mm_items=mm_items,  # 多模态数据项
            im_start_id=self.vision_start_token_id,  # 视觉起始标记ID
            im_end_id=self.vision_end_token_id,  # 视觉结束标记ID
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记ID
            audio_token_id=self.mm_tokens.audio_token_id,  # 音频标记ID
            mrope_positions=mrope_positions,  # M-RoPE位置
            mrope_position_delta=mrope_position_delta,  # M-RoPE位置增量
        )
