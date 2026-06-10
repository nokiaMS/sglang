# Ernie 4.5 VL 多模态处理器模块
# 本模块实现了文心 4.5 视觉语言模型的多模态数据处理逻辑，
# 包括图像和视频的智能缩放、帧数计算、预处理等功能，
# 并支持 MRotaryEmbedding 位置编码的计算。
import math  # 导入数学模块
import os  # 导入操作系统模块
from typing import List, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 模块
import torch  # 导入 PyTorch 模块
import torchvision  # 导入 Torchvision 模块
from PIL import Image  # 导入 PIL 图像模块
from torchvision.transforms import InterpolationMode  # 导入插值模式枚举
from transformers import BaseImageProcessor  # 导入 HuggingFace 基础图像处理器类

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding  # 导入多维旋转位置编码模块
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.ernie45_vl import Ernie4_5_VLMoeForConditionalGeneration  # 导入 Ernie4.5 VL MoE 模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模odal处理器
    BaseMultimodalProcessor as SGLangBaseProcessor,  # 将基础处理器重命名为 SGLangBaseProcessor
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入多模态特殊标记类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)
from sglang.srt.utils import get_bool_env_var, is_npu, logger  # 导入工具函数：布尔环境变量获取、NPU检测、日志器

_is_npu = is_npu()  # 检测当前是否为 NPU 环境

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")  # 获取是否使用 CUDA IPC 传输的环境变量


IMAGE_FACTOR = 28  # 图像因子，用于对齐像素
MIN_PIXELS = 4 * 28 * 28  # 最小像素数
# MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_PIXELS = 16384 * 28 * 28  # 最大像素数
MAX_RATIO = 200  # 最大宽高比
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)  # 获取图像缩放重采样方法
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:  # 如果环境变量已设置但获取的方法为空
    logger.warning(  # 输出警告日志
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "  # 无效的重采样值
        f"Ignoring and using default."  # 忽略并使用默认值
    )
VIDEO_TOTAL_PIXELS = int(  # 视频总像素数
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))  # 从环境变量获取或使用默认值
)

VIDEO_MIN_PIXELS = 299 * 28 * 28  # 视频最小像素数
VIDEO_MAX_PIXELS = 1196 * 28 * 28  # 视频最大像素数
FRAME_FACTOR = 2  # 帧数因子，帧数必须是该值的倍数
FPS = 2.0  # 默认帧率
FPS_MIN_FRAMES = 16  # 最小帧数
FPS_MAX_FRAMES = 180  # 最大帧数


def smart_resize(  # 智能调整图像尺寸，确保尺寸满足因子约束和像素范围
    height: int,  # 原始高度
    width: int,  # 原始宽度
    factor: int = IMAGE_FACTOR,  # 对齐因子，默认为图像因子
    min_pixels: int = MIN_PIXELS,  # 最小像素数
    max_pixels: int = MAX_PIXELS,  # 最大像素数
):
    if max(height, width) / min(height, width) > MAX_RATIO:  # 如果宽高比超过最大限制
        if height > width:  # 如果高度大于宽度
            new_width = max(factor, round_by_factor(width, factor))  # 新宽度按因子取整
            new_height = floor_by_factor(new_width * MAX_RATIO, factor)  # 新高度按最大宽高比和因子向下取整
        else:  # 如果宽度大于等于高度
            new_height = max(factor, round_by_factor(height, factor))  # 新高度按因子取整
            new_width = floor_by_factor(new_height * MAX_RATIO, factor)  # 新宽度按最大宽高比和因子向下取整

        height = new_height  # 更新高度
        width = new_width  # 更新宽度

    h_bar = max(factor, round_by_factor(height, factor))  # 高度按因子向上取整，至少为因子值
    w_bar = max(factor, round_by_factor(width, factor))  # 宽度按因子向上取整，至少为因子值
    if h_bar * w_bar > max_pixels:  # 如果像素数超过最大限制
        beta = math.sqrt((height * width) / max_pixels)  # 计算缩放因子
        h_bar = floor_by_factor(height / beta, factor)  # 高度按缩放因子和因子向下取整
        w_bar = floor_by_factor(width / beta, factor)  # 宽度按缩放因子和因子向下取整
    elif h_bar * w_bar < min_pixels:  # 如果像素数低于最小限制
        beta = math.sqrt(min_pixels / (height * width))  # 计算放大因子
        h_bar = ceil_by_factor(height * beta, factor)  # 高度按放大因子和因子向上取整
        w_bar = ceil_by_factor(width * beta, factor)  # 宽度按放大因子和因子向上取整

    if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:  # 验证最终像素数是否在范围内
        raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")  # 抛出异常

    return h_bar, w_bar  # 返回调整后的高度和宽度


def resize_image(  # 调整图像大小，使其满足像素和因子约束
    image,  # 输入图像
    min_pixels: int = MIN_PIXELS,  # 最小像素数
    max_pixels: int = MAX_PIXELS,  # 最大像素数
    size_factor: int = IMAGE_FACTOR,  # 尺寸对齐因子
) -> Image.Image:
    width, height = image.size  # 获取图像的宽度和高度
    min_pixels = min_pixels  # 最小像素数
    max_pixels = max_pixels  # 最大像素数
    resized_height, resized_width = smart_resize(  # 调用智能调整函数计算目标尺寸
        height,  # 高度
        width,  # 宽度
        factor=size_factor,  # 对齐因子
        min_pixels=min_pixels,  # 最小像素数
        max_pixels=max_pixels,  # 最大像素数
    )
    image = image.resize((resized_width, resized_height), resample=RESIZE_RESAMPLE)  # 使用指定重采样方法调整图像尺寸
    return image  # 返回调整后的图像


def round_by_factor(number: int | float, factor: int) -> int:  # 将数值按因子四舍五入取整
    return round(number / factor) * factor  # 返回四舍五入后的结果


def ceil_by_factor(number: int | float, factor: int) -> int:  # 将数值按因子向上取整
    return math.ceil(number / factor) * factor  # 返回向上取整后的结果


def floor_by_factor(number: int | float, factor: int) -> int:  # 将数值按因子向下取整
    return math.floor(number / factor) * factor  # 返回向下取整后的结果


async def resize_image_async(  # 异步调整图像大小，直接调用同步版本
    image,  # 输入图像
    min_pixels: int = MIN_PIXELS,  # 最小像素数
    max_pixels: int = MAX_PIXELS,  # 最大像素数
    size_factor: int = IMAGE_FACTOR,  # 尺寸对齐因子
):
    return resize_image(image, min_pixels, max_pixels, size_factor)  # 调用同步版本的 resize_image 并返回结果


def smart_nframes(  # 智能计算视频帧数，支持按帧率或指定帧数提取
    ele: dict,  # 视频配置字典
    total_frames: int,  # 视频原始总帧数
    video_fps: int | float,  # 视频原始帧率
) -> int:
    """calculate the number of frames for video used for model inputs.
    计算模型输入所需的视频帧数。

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
        返回模型输入所需的视频帧数。
    """
    assert not (  # 断言不能同时指定 fps 和 nframes
        "fps" in ele and "nframes" in ele  # 检查配置中是否同时包含 fps 和 nframes
    ), "Only accept either `fps` or `nframes`"  # 只接受 fps 或 nframes 之一
    if "nframes" in ele:  # 如果指定了帧数
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)  # 将帧数按因子四舍五入
    else:  # 否则根据帧率计算
        fps = ele.get("fps", FPS)  # 获取目标帧率，默认为 FPS
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)  # 计算最小帧数
        max_frames = floor_by_factor(  # 计算最大帧数
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR  # 取最大帧数和总帧数的较小值
        )
        nframes = total_frames / video_fps * fps  # 根据帧率比例计算目标帧数
        if nframes > total_frames:  # 如果计算出的帧数超过总帧数
            logger.warning(  # 输出警告日志
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"  # 帧数超出总帧数的警告
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)  # 将帧数限制在合理范围内
        nframes = floor_by_factor(nframes, FRAME_FACTOR)  # 按因子向下取整
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):  # 验证帧数是否在有效范围内
        raise ValueError(  # 抛出异常
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."  # 帧数不在有效范围内的错误信息
        )
    return nframes  # 返回计算出的帧数


# process video, qwen-specific
async def preprocess_video(  # 异步预处理视频，提取帧并调整尺寸
    vr,  # 视频读取器对象
    image_factor: int = IMAGE_FACTOR,  # 图像对齐因子
) -> torch.Tensor:

    total_frames, video_fps = len(vr), vr.get_avg_fps()  # 获取视频总帧数和平均帧率
    nframes = smart_nframes({}, total_frames=total_frames, video_fps=video_fps)  # 计算目标帧数
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)  # 均匀采样帧索引
    idx = np.unique(idx)  # 去除重复的帧索引
    video_np = vr.get_batch(idx).asnumpy()  # 获取指定帧的 numpy 数组
    video = torch.from_numpy(video_np).pin_memory()  # 转换为 PyTorch 张量并固定内存
    video = video.permute(0, 3, 1, 2)  # Convert to TCHW format  # 转换为 TCHW 格式
    nframes, _, height, width = video.shape  # 获取视频帧数、通道数、高度和宽度
    min_pixels = VIDEO_MIN_PIXELS  # 设置最小像素数
    total_pixels = VIDEO_TOTAL_PIXELS  # 设置总像素数
    max_pixels = max(  # 计算每帧最大像素数
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),  # 取视频最大像素数和按帧均分像素数的较小值
        int(min_pixels * 1.05),  # 至少为最小像素数的 1.05 倍
    )

    resized_height, resized_width = smart_resize(  # 智能计算调整后的尺寸
        height,  # 原始高度
        width,  # 原始宽度
        factor=image_factor,  # 对齐因子
        min_pixels=min_pixels,  # 最小像素数
        max_pixels=max_pixels,  # 最大像素数
    )
    video = torchvision.transforms.functional.resize(  # 使用 torchvision 调整视频帧尺寸
        video,  # 输入视频张量
        [resized_height, resized_width],  # 目标尺寸
        interpolation=InterpolationMode.BILINEAR,  # 使用双线性插值
    )

    video = video.permute(0, 2, 3, 1)  # 将 TCHW 格式转回 THWC 格式
    video = video.pin_memory()  # 固定内存以加速 CPU-GPU 传输
    video_metadata = {  # 构建视频元数据字典
        "fps": video_fps,  # 原始帧率
        "duration": total_frames / video_fps,  # 视频时长（秒）
        "total_num_frames": total_frames,  # 原始总帧数
        "frames_indices": idx,  # 采样的帧索引
        "video_backend": "torchvision",  # 视频后端类型
    }

    return video, video_metadata  # 返回处理后的视频张量和元数据


# Compatible with Ernie-VL Series
class Ernie4_5_VLImageProcessor(SGLangBaseProcessor):  # Ernie 4.5 VL 图像处理器类，兼容文心 VL 系列
    models = [Ernie4_5_VLMoeForConditionalGeneration]  # 支持的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 Ernie 4.5 VL 图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法
        self.hf_config = hf_config  # 保存 HuggingFace 配置
        self.model_type = hf_config.model_type  # 获取模型类型
        self.image_start_token_id = hf_config.image_start_token_id  # 图像起始标记 ID
        self.image_end_token_id = hf_config.image_end_token_id  # 图像结束标记 ID
        self.video_start_token_id = hf_config.video_start_token_id  # 视频起始标记 ID
        self.video_end_token_id = hf_config.video_end_token_id  # 视频结束标记 ID

        self.IMAGE_FACTOR = 28  # 图像因子
        self.MIN_PIXELS = 4 * 28 * 28  # 最小像素数
        self.MAX_PIXELS = 16384 * 28 * 28  # 最大像素数
        self.MAX_RATIO = 200  # 最大宽高比
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token="<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>",  # 图像标记模板
            video_token="<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>",  # 视频标记模板
            image_token_id=hf_config.im_patch_id,  # 图像标记 ID
            video_token_id=hf_config.im_patch_id,  # image and video use the same token_id  # 图像和视频使用相同的 token ID
        ).build(_processor)  # 使用处理器构建标记

        self.tokenizer = self._processor.tokenizer  # 保存分词器
        self.image_processor = self._processor.image_processor  # 保存图像处理器

    def _pixel_values_norm(  # 对像素值进行归一化处理，应用均值和标准差
        self,
        pixel_values: torch.Tensor,  # 输入像素值张量
        mm_kwargs: object,  # 多模态参数对象
    ) -> torch.Tensor:
        hf_config = self.hf_config  # 获取 HuggingFace 配置
        vision_config = hf_config.vision_config  # 获取视觉编码器配置
        image_processor = self.image_processor  # 获取图像处理器
        image_mean_tensor = torch.tensor(  # 创建图像均值张量
            image_processor.image_mean, dtype=torch.float32  # 使用处理器的均值参数
        ).reshape([1, 3, 1, 1])  # 调整形状为 [1, 3, 1, 1] 以支持广播
        image_std_tensor = torch.tensor(  # 创建图像标准差张量
            image_processor.image_std, dtype=torch.float32  # 使用处理器的标准差参数
        ).reshape([1, 3, 1, 1])  # 调整形状为 [1, 3, 1, 1] 以支持广播
        rescale_factor = torch.tensor(  # 创建重缩放因子张量
            image_processor.rescale_factor, dtype=torch.float32  # 使用处理器的重缩放因子
        )
        patch_size_squared = vision_config.patch_size**2  # 计算 patch 大小的平方

        image_mean_tensor = image_mean_tensor.squeeze([-2, -1]).repeat_interleave(  # 将均值张量按 patch 大小扩展
            patch_size_squared, -1  # 在通道维度上重复 patch_size_squared 次
        )
        image_std_tensor = image_std_tensor.squeeze([-2, -1]).repeat_interleave(  # 将标准差张量按 patch 大小扩展
            patch_size_squared, -1  # 在通道维度上重复 patch_size_squared 次
        )

        if not image_mean_tensor.is_contiguous():  # 如果均值张量不连续
            image_mean_tensor = image_mean_tensor.contiguous()  # 使其连续
        if not image_std_tensor.is_contiguous():  # 如果标准差张量不连续
            image_std_tensor = image_std_tensor.contiguous()  # 使其连续

        pixel_values = (  # 计算归一化后的像素值
            rescale_factor * pixel_values.to(torch.float32) - image_mean_tensor  # 先缩放再减去均值
        ) / image_std_tensor  # 除以标准差
        pixel_values = pixel_values.to(hf_config.dtype)  # 转换为模型所需的数据类型
        return pixel_values  # 返回归一化后的像素值

    def process_mm_data(  # 使用 transformers AutoProcessor 处理多模态数据
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor
        使用 transformers AutoProcessor 处理多模态数据
        """
        if images:  # 如果有图像数据
            kwargs["images"] = images  # 将图像添加到关键字参数中
            if self.image_config:  # 如果有图像配置
                kwargs.setdefault("images_kwargs", {}).update(self.image_config)  # 设置图像处理参数
        if videos:  # 如果有视频数据
            kwargs["videos"] = videos  # 将视频添加到关键字参数中
            if self.video_config:  # 如果有视频配置
                kwargs.setdefault("videos_kwargs", {}).update(self.video_config)  # 设置视频处理参数

        processor = self._processor  # 获取处理器
        if (  # 如果处理器有图像处理器且是 BaseImageProcessor 类型
            hasattr(processor, "image_processor")  # 检查是否有 image_processor 属性
            and isinstance(processor.image_processor, BaseImageProcessor)  # 检查是否为 BaseImageProcessor 实例
            and not self.server_args.disable_fast_image_processor  # 检查是否未禁用快速图像处理器
        ):
            if not _is_npu:  # 如果不是 NPU 环境
                kwargs["device"] = "cuda"  # 设置设备为 CUDA

        result = processor.__call__(  # 调用处理器处理数据
            text=[input_text],  # 输入文本
            padding=True,  # 启用填充
            return_tensors="pt",  # 返回 PyTorch 张量
            **kwargs,  # 其他参数
        )

        # Divide the processor_output into two modalities: image and video.
        if result is not None:  # 如果处理结果不为空
            pixel_values = result["images"]  # 获取像素值
            if pixel_values is not None:  # 如果像素值不为空
                result["images"] = self._pixel_values_norm(pixel_values, kwargs)  # 对像素值进行归一化
            for key in list(result.keys()):  # 遍历结果的所有键
                if result[key] is None:  # 如果值为空
                    del result[key]  # 删除该键
                    continue  # 继续下一个键
                if key == "grid_thw":  # 如果键为 grid_thw（网格时间-高度-宽度信息）
                    grid_thw = result["grid_thw"]  # 获取网格信息
                    pixel_values_all = result["images"]  # 获取所有像素值
                    # Identify elements where the first
                    # dimension is greater than 1 and
                    # treat them as the video modality
                    mask = grid_thw[:, 0] > 1  # 识别第一维大于1的元素，视为视频模态
                    result["video_grid_thw"] = grid_thw[mask]  # 设置视频网格信息
                    result["image_grid_thw"] = grid_thw[~mask]  # 设置图像网格信息
                    image_patch_num = result["image_grid_thw"].prod(dim=1).sum()  # 计算图像 patch 总数
                    result["pixel_values"] = pixel_values_all[:image_patch_num]  # 设置图像像素值
                    result["pixel_values_videos"] = pixel_values_all[image_patch_num:]  # 设置视频像素值
                    del result["images"]  # 删除原始图像键
                    del result["grid_thw"]  # 删除原始网格键

                    # del empty result
                    if result["image_grid_thw"].numel() == 0:  # 如果图像网格信息为空
                        del result["image_grid_thw"]  # 删除图像网格键
                    if result["pixel_values"].numel() == 0:  # 如果图像像素值为空
                        del result["pixel_values"]  # 删除图像像素值键
                    if result["video_grid_thw"].numel() == 0:  # 如果视频网格信息为空
                        del result["video_grid_thw"]  # 删除视频网格键
                    if result["pixel_values_videos"].numel() == 0:  # 如果视频像素值为空
                        del result["pixel_values_videos"]  # 删除视频像素值键

        if not self.server_args.keep_mm_feature_on_device:  # 如果不需要在设备上保留多模态特征
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:  # 遍历特征名称列表
                if SGL_USE_CUDA_IPC:  # 如果使用 CUDA IPC 传输
                    pass  # 不需要移动到 CPU
                else:  # 否则
                    if feature_name in result and isinstance(  # 如果特征在结果中且为张量
                        result[feature_name], torch.Tensor
                    ):
                        result[feature_name] = result[feature_name].to("cpu")  # 将特征移动到 CPU

        return result  # 返回处理结果

    def compute_mrope_positions(self, input_ids, mm_items):  # 计算多维旋转位置编码
        image_grid_thw = None  # 初始化图像网格信息
        video_grid_thw = None  # 初始化视频网格信息
        for item in mm_items:  # 遍历多模态项
            if "image_grid_thw" in item.model_specific_data:  # 如果项中包含图像网格信息
                image_grid_thw = item.model_specific_data["image_grid_thw"]  # 获取图像网格信息
            if "video_grid_thw" in item.model_specific_data:  # 如果项中包含视频网格信息
                video_grid_thw = item.model_specific_data["video_grid_thw"]  # 获取视频网格信息

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # 将输入 ID 转为张量并增加批次维度
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_ernie45(  # 调用 Ernie45 专用的旋转位置编码计算
            input_ids=input_ids_tensor,  # 输入 ID 张量
            hf_config=self.hf_config,  # HuggingFace 配置
            image_grid_thw=image_grid_thw,  # 图像网格信息
            video_grid_thw=video_grid_thw,  # 视频网格信息
        )
        return mrope_positions.squeeze(1), mrope_position_delta  # 返回位置编码和增量，去除多余维度

    async def process_mm_data_async(  # 异步处理多模态数据，支持图像、视频和音频
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            video_data=request_obj.video_data,  # 视频数据
            audio_data=request_obj.audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )

        # resize images if they are raw Image objects
        resized_images = []  # 初始化调整后的图像列表
        if base_output.images and isinstance(base_output.images[0], Image.Image):  # 如果图像是 PIL Image 对象
            for image in base_output.images:  # 遍历所有图像
                resized_image = resize_image(image)  # 调整图像大小
                resized_images.append(resized_image)  # 添加到调整后的图像列表
            base_output.images = resized_images  # 替换原始图像列表

        if base_output.videos:  # 如果有视频数据
            videos_processed = [  # 预处理所有视频
                await preprocess_video(video) for video in base_output.videos  # 异步预处理每个视频
            ]
            base_output.videos, _ = map(list, zip(*videos_processed))  # 分离视频数据和元数据，只保留视频数据

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )

        input_ids = input_ids.flatten()  # 将输入 ID 展平为一维

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_ernie45(  # 计算 Ernie45 的旋转位置编码
            input_ids=input_ids.unsqueeze(0),  # 增加批次维度
            hf_config=self.hf_config,  # HuggingFace 配置
            image_grid_thw=getattr(ret, "image_grid_thw", None),  # 获取图像网格信息
            video_grid_thw=getattr(ret, "video_grid_thw", None),  # 获取视频网格信息
        )
        mrope_positions = mrope_positions.squeeze(1)  # 去除多余的中间维度

        assert (  # 断言输入 ID 和位置编码长度一致
            input_ids.shape[0] == mrope_positions.shape[-1]
        ), "input_ids and mrope_positions should have the same length"  # 输入 ID 和位置编码应有相同长度

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_start_id=self.image_start_token_id,  # 图像起始标记 ID
            im_end_id=self.image_end_token_id,  # 图像结束标记 ID
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记 ID
            mrope_positions=mrope_positions,  # 多维旋转位置编码
            mrope_position_delta=mrope_position_delta,  # 位置编码增量
        )
