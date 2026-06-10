# 本文件实现多模态处理器基类 BaseMultimodalProcessor，提供图像/视频/音频数据的加载、
# 预处理、组合、CUDA IPC 传输等核心功能，是所有具体多模态处理器的父类
import asyncio  # 导入异步 IO 模块
import concurrent  # 导入并发模块
import concurrent.futures  # 导入并发未来对象
import dataclasses  # 导入数据类模块
import multiprocessing as mp  # 导入多进程模块
import os  # 导入操作系统模块
import re  # 导入正则表达式模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 numpy 数值计算库
import torch  # 导入 PyTorch 深度学习框架
from PIL import Image  # 导入 PIL 图像处理库
from transformers import BaseImageProcessor  # 导入 Hugging Face 基础图像处理器

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import (  # 导入工具函数
    envs,
    is_cpu,
    is_npu,
    is_xpu,
    load_audio,
    load_image,
    load_video,
    logger,
)
from sglang.srt.utils.cuda_ipc_transport_utils import (  # 导入 CUDA IPC 传输相关工具
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
)

_is_cpu = is_cpu()  # 判断是否为 CPU 设备
_is_npu = is_npu()  # 判断是否为 NPU 设备
_is_xpu = is_xpu()  # 判断是否为 XPU 设备

SGL_USE_CUDA_IPC = envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get()  # 获取是否使用 CUDA IPC 传输的环境变量
_IPC_POOL_HANDLE_CACHE = envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()  # 获取是否使用 IPC 池句柄缓存


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:  # 多模态处理器输出的基类
    # input_text with all multimodality placeholder token expanded
    input_text: str  # 包含所有多模态占位符 token 的扩展文本

    # original pre-tokenized ids, useful for processor_output/precomputed inputs,
    # when they already carry the input ids
    input_ids: Optional[Union[List[int], torch.Tensor]] = None  # 原始预分词 ID

    # frames loaded from image, in given order
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )  # 加载的图像帧列表

    # videos
    videos: Optional[list[Union[torch.Tensor, dict]]] = dataclasses.field(
        default_factory=list
    )  # 加载的视频列表

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = dataclasses.field(
        default_factory=list
    )  # 加载的音频列表

    def organize_results(self) -> List[Tuple[Modality, Any]]:  # 将结果按模态组织为列表
        """将加载的多模态数据按模态分类返回

        :return: a list of results, with their corresponding modalities
        """
        return (
            [(Modality.IMAGE, data) for data in self.images]  # 图像模态
            + [(Modality.VIDEO, data) for data in self.videos]  # 视频模态
            + [(Modality.AUDIO, data) for data in self.audios]  # 音频模态
        )


@dataclasses.dataclass
class MultimodalSpecialTokens:  # 多模态特殊 token 配置类
    image_token: Optional[Union[str, List[str]]] = None  # 图像特殊 token
    video_token: Optional[Union[str, List[str]]] = None  # 视频特殊 token
    audio_token: Optional[Union[str, List[str]]] = None  # 音频特殊 token

    image_token_id: Optional[int] = None  # 图像特殊 token ID
    video_token_id: Optional[int] = None  # 视频特殊 token ID
    audio_token_id: Optional[int] = None  # 音频特殊 token ID

    image_token_regex: Optional[re.Pattern] = None  # 图像 token 正则表达式
    video_token_regex: Optional[re.Pattern] = None  # 视频 token 正则表达式
    audio_token_regex: Optional[re.Pattern] = None  # 音频 token 正则表达式

    combined_regex: Optional[re.Pattern] = None  # 组合正则表达式

    def build(self, processor):  # 构建并初始化特殊 token 配置
        """构建特殊 token 的字符串、正则表达式和组合正则"""
        self.convert_to_strs(processor)  # 将 token ID 转换为字符串
        self.parse_regex()  # 解析正则表达式
        self.get_combined_regex()  # 获取组合正则
        return self  # 返回自身

    def convert_to_str(self, token: Union[str, int], processor) -> str:  # 将 token 转换为字符串
        """将单个 token（ID 或字符串）转换为字符串表示"""
        if token is None:  # 如果 token 为 None
            return token  # 直接返回 None
        if isinstance(token, str):  # 如果已经是字符串
            return token  # 直接返回
        return processor.tokenizer.convert_ids_to_tokens([token])[0]  # 通过分词器将 ID 转为 token 字符串

    def convert_to_strs(self, processor):  # 将所有 token ID 转换为字符串
        """将所有未设置的 token 字符串通过 ID 从分词器转换"""
        if not self.image_token:  # 如果图像 token 字符串未设置
            self.image_token = self.convert_to_str(self.image_token_id, processor)  # 通过 ID 转换
        if not self.video_token:  # 如果视频 token 字符串未设置
            self.video_token = self.convert_to_str(self.video_token_id, processor)  # 通过 ID 转换
        if not self.audio_token:  # 如果音频 token 字符串未设置
            self.audio_token = self.convert_to_str(self.audio_token_id, processor)  # 通过 ID 转换

    def get_modality_of_token(self, token: str) -> Optional[Modality]:  # 根据 token 获取模态类型
        """返回给定 token 对应的模态类型

        :return: the modality associated with the given token, if the token is a special_token or matches with the multimodal token regex
        """
        modality = {  # 直接匹配 token 字符串
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }.get(token)
        if modality:  # 如果找到匹配
            return modality  # 返回模态

        for regex, modality in [  # 遍历正则表达式匹配
            (self.image_token_regex, Modality.IMAGE),
            (self.video_token_regex, Modality.VIDEO),
            (self.audio_token_regex, Modality.AUDIO),
        ]:
            if regex and regex.match(token):  # 如果正则匹配成功
                return modality  # 返回模态

        return None  # 未匹配则返回 None

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:  # 根据模态获取 token ID
        """根据模态类型获取对应的特殊 token ID"""
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)

    def parse_regex(self):  # 解析并编译正则表达式
        """为每个特殊 token 字符串编译正则表达式"""
        if self.image_token_regex is None and self.image_token is not None:  # 如果图像正则未编译
            self.image_token_regex = re.compile(re.escape(self.image_token))  # 编译图像 token 正则
        if self.video_token_regex is None and self.video_token is not None:  # 如果视频正则未编译
            self.video_token_regex = re.compile(re.escape(self.video_token))  # 编译视频 token 正则
        if self.audio_token_regex is None and self.audio_token is not None:  # 如果音频正则未编译
            self.audio_token_regex = re.compile(re.escape(self.audio_token))  # 编译音频 token 正则

    def get_combined_regex(self) -> re.Pattern:  # 获取组合正则表达式
        """
        Builds and returns a regex, used to split input str into tokens (with mm special tokens)
        构建并返回组合正则表达式，用于将输入字符串按多模态特殊 token 分割。
        """
        if self.combined_regex:  # 如果已有组合正则
            return self.combined_regex  # 直接返回
        tokens = [  # 收集所有 token 正则
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []  # 存储非空正则模式
        flags = 0  # 正则标志位
        for t in tokens:  # 遍历所有 token 正则
            if t is not None:  # 如果正则不为空
                patterns.append(t.pattern)  # 添加模式字符串
                flags |= t.flags  # 合并标志位
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"  # 构建组合正则
        self.combined_regex = re.compile(combined, flags)  # 编译组合正则
        return self.combined_regex  # 返回组合正则


class BaseMultimodalProcessor(ABC):  # 多模态处理器基类
    models = []  # 支持的模型列表，由子类覆写
    gpu_image_decode = True  # Enable GPU decoding by default 默认启用 GPU 解码

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):  # 初始化多模态处理器
        self.hf_config = hf_config  # 保存 Hugging Face 配置
        self._processor = _processor  # 保存处理器实例
        self.server_args = server_args  # 保存服务器参数
        self.transport_mode = transport_mode  # 保存传输模式

        mm_process_config = self.server_args.mm_process_config  # 获取多模态处理配置
        self.image_config = mm_process_config.get("image", {})  # 获取图像配置
        self.video_config = mm_process_config.get("video", {})  # 获取视频配置
        self.audio_config = mm_process_config.get("audio", {})  # 获取音频配置

        # Resolve tokenizer: some processors (e.g. InternVL) pass a tokenizer
        # directly as _processor rather than a processor that wraps a tokenizer.
        if hasattr(self._processor, "tokenizer"):  # 如果处理器包含分词器
            self._tokenizer = self._processor.tokenizer  # 使用处理器的分词器
        else:  # 否则
            self._tokenizer = self._processor  # 处理器本身就是分词器

        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330  # 每帧的默认 token 数（不精确）

        self.io_executor = concurrent.futures.ThreadPoolExecutor(  # IO 线程池
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))  # 最大工作线程数
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(  # CPU 进程池
            mp_context=mp.get_context("fork"),  # 使用 fork 上下文
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),  # 最大工作进程数
        )

        # Mapping from attribute names to modality types
        self.ATTR_NAME_TO_MODALITY = {  # 属性名到模态类型的映射
            # Image-related attributes
            "pixel_values": Modality.IMAGE,  # 像素值
            "image_sizes": Modality.IMAGE,  # 图像尺寸
            "image_grid_thw": Modality.IMAGE,  # 图像网格 (时间, 高, 宽)
            "image_attention_mask": Modality.IMAGE,  # 图像注意力掩码
            "image_emb_mask": Modality.IMAGE,  # 图像嵌入掩码
            "images_spatial_crop": Modality.IMAGE,  # 图像空间裁剪
            "images_crop": Modality.IMAGE,  # 图像裁剪
            "has_local_crops": Modality.IMAGE,  # 是否有局部裁剪
            "has_images": Modality.IMAGE,  # 是否有图像
            "tgt_size": Modality.IMAGE,  # 目标尺寸
            "image_grid_hws": Modality.IMAGE,  # 图像网格高宽
            "aspect_ratio_ids": Modality.IMAGE,  # 宽高比 ID
            "aspect_ratio_mask": Modality.IMAGE,  # 宽高比掩码
            "num_patches": Modality.IMAGE,  # 补丁数
            "patch_pixel_values": Modality.IMAGE,  # 补丁像素值
            "block_sizes": Modality.IMAGE,  # 块尺寸
            "grid_thws": Modality.IMAGE,  # for kimi k2.5 网格宽高（用于 kimi k2.5）
            # Audio-related attributes
            "audio_features": Modality.AUDIO,  # 音频特征
            "audio_feature_lens": Modality.AUDIO,  # 音频特征长度
            "input_features": Modality.AUDIO,  # 输入特征
            "input_features_mask": Modality.AUDIO,  # 输入特征掩码
            "audio_attention_mask": Modality.AUDIO,  # 音频注意力掩码
            "feature_attention_mask": Modality.AUDIO,  # 特征注意力掩码
            # Video-related attributes
            "pixel_values_videos": Modality.VIDEO,  # 视频像素值
            "second_per_grid_ts": Modality.VIDEO,  # 每网格时间戳
            "video_grid_thw": Modality.VIDEO,  # 视频网格 (时间, 高, 宽)
            # Generic attributes that could apply to multiple modalities
            # "precomputed_embeddings" - handled specially as it can be any modality
        }

        # name of the feature filed
        # TODO: pass from processors
        self.FEATURE_NAMES = [  # 特征字段名列表
            "pixel_values",  # 像素值
            "pixel_values_videos",  # 视频像素值
            "audio_features",  # 音频特征
            "input_features",  # 输入特征
        ]

        skip_mm_pool = kwargs.get("skip_mm_pool", False)  # 是否跳过多模态内存池

        if SGL_USE_CUDA_IPC and not skip_mm_pool:  # 如果使用 CUDA IPC 且不跳过内存池
            # SGLANG_MM_FEATURE_CACHE_MB is the total pool budget across all
            # tokenizer workers. Each worker gets an equal share so that adding
            # workers doesn't multiply the GPU-side footprint.
            worker_num = self.server_args.tokenizer_worker_num  # 获取分词工作器数量
            per_worker_pool_size = max(  # 计算每个工作器的池大小
                MM_FEATURE_CACHE_SIZE // worker_num,  # 平均分配
                128 * 1024 * 1024,  # 最小 128 MiB
            )
            logger.info(
                "MmItemMemoryPool size per tokenizer worker: %.0f MiB "
                "(budget %.0f MiB / %d worker(s))",
                per_worker_pool_size / (1024 * 1024),
                MM_FEATURE_CACHE_SIZE / (1024 * 1024),
                worker_num,
            )  # 记录内存池大小信息
            self.cudaipc_mmfeature_pool = MmItemMemoryPool(  # 创建多模态内存池
                per_worker_pool_size,
                MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
            )

    def compute_mrope_positions(self, input_ids, mm_items):  # 计算 M-RoPE 位置
        """Compute M-RoPE positions from expanded input_ids and multimodal items.
        从扩展的 input_ids 和多模态项计算 M-RoPE 位置。

        Returns (mrope_positions, mrope_position_delta) or (None, None) if the
        model does not use M-RoPE.
        """
        return None, None  # 默认不使用 M-RoPE

    @property
    def spatial_merge_size(self):  # 空间合并尺寸属性
        return self.hf_config.vision_config.spatial_merge_size  # 返回视觉配置中的空间合并尺寸

    def build_input_ids(
        self, prompt, img_grid_thw=None, video_grid_thw=None, audio_seq_lens=None
    ):  # 构建 input_ids
        """
        Use prompt, img_grid_thw, video_grid_thw, and audio_seq_lens to build input_ids.
        Supports image, video, and audio tokens.
        使用提示词和网格尺寸信息构建包含多模态 token 的 input_ids。
        """
        if not isinstance(prompt, list):  # 如果提示词不是列表
            prompt = self._tokenizer.encode(prompt)  # 对提示词进行分词

        img_token_id = getattr(self, "IM_TOKEN_ID", None)  # 获取图像 token ID
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)  # 获取视频 token ID
        audio_token_id = getattr(self, "audio_token_id", None)  # 获取音频 token ID
        spatial_merge_size = getattr(self, "spatial_merge_size", 1)  # 获取空间合并尺寸

        input_ids = []  # 存储构建的 input_ids
        offsets = []  # 存储多模态 token 的偏移区间

        cur_idx = 0  # 当前处理位置

        # Use img_token_id instead of im_start_id, because a dummy im_start_id
        # may be generated by the tokenizer.
        vision_start_indices = []  # 存储视觉 token 的起始位置
        for i in range(len(prompt) - 1):  # 遍历提示词
            if img_token_id is not None and prompt[i + 1] == img_token_id:  # 如果下一个是图像 token
                vision_start_indices.append((i, Modality.IMAGE))  # 记录位置和模态
            elif video_token_id is not None and prompt[i + 1] == video_token_id:  # 如果下一个是视频 token
                vision_start_indices.append((i, Modality.VIDEO))  # 记录位置和模态
            elif audio_token_id is not None and prompt[i + 1] == audio_token_id:  # 如果下一个是音频 token
                vision_start_indices.append((i, Modality.AUDIO))  # 记录位置和模态
        # get modality list with order preserved
        modality_list = [modality for _, modality in vision_start_indices]  # 提取模态列表（保持顺序）

        img_idx = 0  # 图像索引
        video_idx = 0  # 视频索引
        audio_idx = 0  # 音频索引
        for mm_start_idx, modality in vision_start_indices:  # 遍历视觉 token 位置
            if modality == Modality.IMAGE:  # 图像模态
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)  # 计算图像 token 数
                mm_token_id = img_token_id  # 设置图像 token ID
                img_idx += 1  # 递增图像索引
            elif modality == Modality.VIDEO:  # 视频模态
                mm_token_num = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2
                )  # 计算视频 token 数
                mm_token_id = video_token_id  # 设置视频 token ID
                video_idx += 1  # 递增视频索引
            elif modality == Modality.AUDIO:  # 音频模态
                mm_token_num = int(audio_seq_lens[audio_idx].item())  # 计算音频 token 数
                mm_token_id = audio_token_id  # 设置音频 token ID
                audio_idx += 1  # 递增音频索引
            else:  # 未知模态
                raise ValueError(f"Invalid modality: {modality}")  # 抛出异常
            assert cur_idx <= mm_start_idx  # 确保游标不超过起始位置

            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])  # 添加前缀 token
            mm_offset_start = len(input_ids)  # 记录多模态 token 起始偏移
            input_ids.extend([mm_token_id] * mm_token_num)  # 添加多模态占位符 token
            cur_idx = (
                mm_start_idx + 2
            )  # jump to img_end_id, video_end_id, or audio_end_id 跳到结束 token 之后
            offsets.append((mm_offset_start, len(input_ids) - 1))  # 记录偏移区间
        else:
            input_ids.extend(prompt[cur_idx:])  # 添加剩余的 token

        return input_ids, offsets, modality_list  # 返回 input_ids、偏移和模态列表

    def get_mm_data(self, prompt, embeddings, **kwargs):  # 获取多模态数据
        """从提示词和嵌入中构建多模态数据项"""
        img_grid_thw = kwargs.get("img_grid_thw", None)  # 获取图像网格尺寸
        video_grid_thw = kwargs.get("video_grid_thw", None)  # 获取视频网格尺寸
        audio_feature_lens = kwargs.get("audio_feature_lens", None)  # 获取音频特征长度

        input_ids, offsets, modality_list = self.build_input_ids(  # 构建 input_ids
            prompt,
            img_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_seq_lens=audio_feature_lens,
        )
        assert all(isinstance(modality, Modality) for modality in modality_list)  # 验证模态类型

        mm_items = []  # 存储多模态数据项
        consumed_per_modality = {}  # 记录每个模态已消费的嵌入数量

        for modality, offset in zip(modality_list, offsets):  # 遍历模态和偏移
            num_tokens = offset[1] - offset[0] + 1  # 计算 token 数
            embedding_start = consumed_per_modality.get(modality, 0)  # 获取该模态的嵌入起始位置
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]  # 切片嵌入
            consumed_per_modality[modality] = embedding_start + num_tokens  # 更新消费位置
            mm_items.append(  # 创建多模态数据项
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(  # 返回处理器输出
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=getattr(self, "VIDEO_TOKEN_ID", None),
        )

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:  # 使用 transformers AutoProcessor 处理多模态数据
        """
        process multimodal data with transformers AutoProcessor
        使用 transformers AutoProcessor 处理多模态数据
        """
        if images:  # 如果有图像
            kwargs["images"] = images  # 添加到参数
            if self.image_config:  # 如果有图像配置
                kwargs.setdefault("images_kwargs", {}).update(self.image_config)  # 更新图像参数
        if videos:  # 如果有视频
            kwargs["videos"] = videos  # 添加到参数
            if self.video_config:  # 如果有视频配置
                kwargs.setdefault("videos_kwargs", {}).update(self.video_config)  # 更新视频参数
        if audios:  # 如果有音频
            if self._processor.__class__.__name__ in {  # 特定处理器需要使用 "audio" 而非 "audios"
                "Gemma3nProcessor",
                "Gemma4Processor",
                "GlmAsrProcessor",
                "Qwen2AudioProcessor",
                "Qwen3ASRProcessor",
                "Qwen3OmniMoeProcessor",
            }:
                # Note(Xinyuan): for gemma3n, ref: https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/processing_gemma3n.py#L107
                kwargs["audio"] = audios  # 使用 "audio" 键
                kwargs.setdefault("audio_kwargs", {})  # 初始化音频参数
                kwargs["audio_kwargs"].setdefault("truncation", False)  # 默认不截断
            else:  # 其他处理器
                kwargs["audios"] = audios  # 使用 "audios" 键
            if self.audio_config:  # 如果有音频配置
                kwargs.setdefault("audio_kwargs", {}).update(self.audio_config)  # 更新音频参数

        processor = self._processor  # 获取处理器
        if (  # 检查是否使用 GPU 图像处理
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessor)
            and not self.server_args.disable_fast_image_processor
        ):
            if _is_cpu or get_global_server_args().rl_on_policy_target is not None:  # CPU 或 RL 模式
                kwargs["device"] = "cpu"  # 使用 CPU
            elif _is_xpu:  # XPU 设备
                kwargs["device"] = "xpu"  # 使用 XPU
            elif not _is_npu:  # 非 NPU 设备
                base_gpu_id = get_global_server_args().base_gpu_id  # 获取基础 GPU ID
                kwargs["device"] = f"cuda:{base_gpu_id}"  # 使用 CUDA 设备
            elif processor.__class__.__name__ not in {  # NPU 上不支持的处理器
                "Glm4vProcessor",
                "Glm46VProcessor",
            }:
                # Note: for qwen-vl, processor has some reshape issue because of dims restriction on Ascend.
                from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
                    npu_apply_qwen_image_preprocess_patch,
                )

                npu_apply_qwen_image_preprocess_patch()  # 应用 NPU 图像预处理补丁
                kwargs["device"] = "npu"  # 使用 NPU
            elif processor.__class__.__name__ == "Glm46VProcessor":  # GLM4.6V 处理器
                from sglang.srt.hardware_backend.npu.modules.glm46v_processor import (
                    npu_apply_glm46v_image_preprocess_patch,
                )

                npu_apply_glm46v_image_preprocess_patch()  # 应用 GLM4.6V 图像预处理补丁
                kwargs["device"] = "npu"  # 使用 NPU

        result = processor.__call__(  # 调用处理器
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        if not self.server_args.keep_mm_feature_on_device:  # 如果不需要保留特征在设备上
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:  # 遍历特征名
                if SGL_USE_CUDA_IPC:  # 如果使用 CUDA IPC
                    pass  # 不移动
                else:  # 否则
                    if feature_name in result and isinstance(
                        result[feature_name], torch.Tensor
                    ):  # 如果特征存在且为张量
                        result[feature_name] = result[feature_name].to("cpu")  # 移动到 CPU

        return result  # 返回处理结果

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:  # 异步处理多模态数据的抽象方法
        pass  # 子类必须实现

    def get_estimated_frames_list(self, image_data):  # 获取所有视觉输入的估计帧数
        """
        estimate the total frame count from all visual input
        估计所有视觉输入的总帧数
        """
        from sglang.srt.utils.video_decoder import VideoDecoderWrapper  # 导入视频解码器

        # Before processing inputs
        if not image_data or len(image_data) == 0:  # 如果没有数据
            return []  # 返回空列表
        estimated_frames_list = []  # 存储估计帧数
        for image in image_data:  # 遍历所有输入
            if isinstance(image, str) and image.startswith("video:"):  # 如果是视频路径
                path = image[len("video:") :]  # 提取路径
                decoder = VideoDecoderWrapper(path)  # 创建解码器
                num_frames = len(decoder)  # 获取帧数
            else:  # 如果是图像
                # For images, each contributes one frame
                num_frames = 1  # 图像贡献一帧
            estimated_frames_list.append(num_frames)  # 添加到列表

        return estimated_frames_list  # 返回估计帧数列表

    @classmethod
    def _load_single_item(
        cls,
        data,  # 数据
        modality: Modality,  # 模态类型
        frame_count_limit=None,  # 帧数限制
        audio_sample_rate: Optional[int] = None,  # 音频采样率
        discard_alpha_channel=True,  # 是否丢弃 alpha 通道
    ):  # 加载单个多模态数据
        """
        Load a single multimodal data.
        加载单个多模态数据。

        If data is processor_output or precomputed embedding, return directly.

        Class method that can be pickled for multiprocessing
        可被 pickle 序列化的类方法，用于多进程加载。
        """
        if cls._is_preprocessed_input(data):  # 如果数据已预处理
            return data  # 直接返回
        try:
            if modality == Modality.IMAGE:  # 图像模态
                img, _ = load_image(data, cls.gpu_image_decode)  # 加载图像
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):  # 如果需要丢弃 alpha 通道且图像不是 RGB
                    # Needed only when `img` is a PIL image
                    img = img.convert("RGB")  # 转换为 RGB
                return img  # 返回图像
            elif modality == Modality.VIDEO:  # 视频模态
                return load_video(data, frame_count_limit)  # 加载视频
            elif modality == Modality.AUDIO:  # 音频模态
                return load_audio(data, audio_sample_rate)  # 加载音频

        except Exception as e:  # 捕获异常
            raise RuntimeError(f"Error while loading data {data}: {e}")  # 抛出运行时异常

    @staticmethod
    def _get_preprocessed_input_format(data):  # 获取预处理输入格式
        """returns the detailed format if the provided data is already preprocessed.
        returns none if the provided data is not preprocessed
        如果数据已预处理则返回详细格式，否则返回 None。
        """
        if not isinstance(data, dict):  # 如果不是字典
            return None  # 返回 None
        data_format = data.get("format")  # 获取格式字段
        if isinstance(data_format, MultimodalInputFormat):  # 如果是枚举类型
            return data_format  # 直接返回
        if data_format in (
            MultimodalInputFormat.PROCESSOR_OUTPUT.name,
            "processor_output",
        ):  # 如果是处理器输出格式
            return MultimodalInputFormat.PROCESSOR_OUTPUT  # 返回对应枚举
        if data_format in (
            MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
            "precomputed_embedding",
        ):  # 如果是预计算嵌入格式
            return MultimodalInputFormat.PRECOMPUTED_EMBEDDING  # 返回对应枚举
        return None  # 未识别格式返回 None

    @classmethod
    def _is_preprocessed_input(cls, data):  # 判断数据是否已预处理
        """returns if the data is already preprocessed (by the vlm processor)
        判断数据是否已被 VLM 处理器预处理。"""
        return cls._get_preprocessed_input_format(data) is not None  # 格式非 None 即已预处理

    @classmethod
    def _all_mm_data_is_preprocessed(cls, *data_lists):  # 判断所有多模态数据是否都已预处理
        """检查所有多模态数据是否都已预处理"""
        has_mm_data = False  # 是否有多模态数据
        for data_list in data_lists:  # 遍历数据列表
            if not data_list:  # 如果列表为空
                continue  # 跳过
            if not isinstance(data_list, list):  # 如果不是列表
                data_list = [data_list]  # 包装为列表
            for item in data_list:  # 遍历每个数据项
                if item is None:  # 如果为 None
                    continue  # 跳过
                has_mm_data = True  # 标记有多模态数据
                if not cls._is_preprocessed_input(item):  # 如果某项未预处理
                    return False  # 返回 False
        return has_mm_data  # 返回是否有多模态数据（全部已预处理）

    def _submit_mm_data_loading_tasks_simple(
        self,
        data_list: Optional[list],  # 数据列表
        modality: Modality,  # 模态类型
        audio_sample_rate: Optional[int],  # 音频采样率
        discard_alpha_channel: bool,  # 是否丢弃 alpha 通道
    ) -> List[Tuple[Modality, int, concurrent.futures.Future]]:  # 返回任务元组列表
        """
        Simple version: For one modal data submit IO load task.
        简化版：为单一模态数据提交 IO 加载任务。

        Return:
            List[(modality, index_in_that_modality, future)]
        """
        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []  # 存储任务列表

        if not data_list:  # 如果数据列表为空
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] no data for modality=%s",
                modality.name,
            )  # 记录调试日志
            return futures  # 返回空列表

        for idx, data in enumerate(data_list):  # 遍历数据列表
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] submit load task: "
                "modality=%s, index=%d, data_type=%s",
                modality.name,
                idx,
                type(data),
            )  # 记录调试日志
            future = self.io_executor.submit(  # 提交加载任务到 IO 线程池
                self.__class__._load_single_item,
                data,
                modality,
                None,  # frame_count_limit: no consider for fast path 帧数限制：快速路径不考虑
                audio_sample_rate,
                discard_alpha_channel,
            )
            futures.append((modality, idx, future))  # 添加到任务列表

        return futures  # 返回任务列表

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],  # 文本部分列表
        multimodal_tokens: MultimodalSpecialTokens,  # 多模态特殊 token
        data_iterators: dict[Modality, Iterator[Any]],  # 数据迭代器字典
        discard_alpha_channel: bool = True,  # 是否丢弃 alpha 通道
        image_estimated_frames_iter: Optional[iter] = None,  # 图像估计帧数迭代器
        image_scaling_factor: float = 1.0,  # 图像缩放因子
        max_image_frames: int = 30,  # 最大图像帧数
        audio_sample_rate: Optional[int] = None,  # 音频采样率
    ) -> Tuple[List, List]:  # 返回 (future列表, 任务信息列表)
        """
        load multimodal data parallelly using iterators.
        使用迭代器并行加载多模态数据。
        """
        futures = []  # 存储加载任务的 future
        task_info = []  # 存储任务信息

        for text_part in text_parts:  # 遍历文本部分
            modality = multimodal_tokens.get_modality_of_token(text_part)  # 获取模态
            if modality is not None:  # 如果是多模态 token
                data_iterator = data_iterators.get(modality)  # 获取对应的数据迭代器
                if data_iterator is None:  # 如果迭代器不存在
                    raise ValueError(f"No data iterator found for token: {text_part}")  # 抛出异常

                try:
                    data = next(data_iterator)  # 获取下一个数据
                except StopIteration:  # 数据不足
                    logger.warning(
                        f"Mismatch: More '{modality.name}' tokens found than corresponding data provided."
                    )  # 记录警告
                    return futures, task_info  # 返回已收集的任务

                frame_count_limit = None  # 帧数限制初始化
                if modality == Modality.IMAGE and image_estimated_frames_iter:  # 图像模态且有帧估计
                    try:
                        estimated_frames = next(image_estimated_frames_iter)  # 获取估计帧数
                        # Use the pre-calculated scaling factor and max frames
                        frame_count_limit = max(
                            1, int(estimated_frames * image_scaling_factor)
                        )  # 计算帧数限制
                        # Ensure we don't exceed the absolute max (redundant if scaling_factor handles it)
                        # frame_count_limit = min(frame_count_limit, max_image_frames)
                    except StopIteration:  # 估计帧数不足
                        raise ValueError(
                            "Mismatch between image tokens and estimated frame counts."
                        )  # 抛出异常

                futures.append(
                    self.io_executor.submit(  # 提交加载任务
                        self.__class__._load_single_item,
                        data,
                        modality,
                        frame_count_limit,
                        audio_sample_rate,
                        discard_alpha_channel,
                    )
                )
                task_info.append((modality, data, frame_count_limit))  # 记录任务信息

        for modality, iterator in data_iterators.items():  # 检查是否有未消费的数据
            try:
                next(iterator)  # 尝试获取下一个
                logger.warning(
                    f"Warning: More {modality.name.lower()} data items provided than corresponding tokens found in the prompt."
                )  # 记录数据多于 token 的警告
            except StopIteration:  # 正常结束
                pass
            except Exception:  # 其他异常
                pass

        return futures, task_info  # 返回任务列表和任务信息

    @staticmethod
    def _validate_one_modality(modality: Modality, data_list: Optional[list]):  # 验证单模态数据
        """验证单个模态的数据列表合法性"""
        if data_list is None:  # 如果数据为 None
            return  # 直接返回
        if not isinstance(data_list, list):  # 如果不是列表
            raise TypeError(
                f"{modality.name} must be a list or None, got {type(data_list)}"
            )  # 抛出类型错误

        formatted_indices = []  # 存储预处理项的索引
        for idx, item in enumerate(data_list):  # 遍历数据
            if BaseMultimodalProcessor._is_preprocessed_input(item):  # 如果已预处理
                formatted_indices.append(idx)  # 记录索引

        if formatted_indices:  # 如果有预处理项
            if len(data_list) != 1:  # 且列表不只一项
                raise ValueError(
                    f"For {modality}, when providing a 'processor_output' or "
                    f"'precomputed_embedding', you must pass exactly one item; "
                    f"received {len(data_list)} items (formatted at indices {formatted_indices})."
                )  # 抛出值错误

    @staticmethod
    def validate_mm_data(
        image_data: Optional[list] = None,  # 图像数据
        video_data: Optional[list] = None,  # 视频数据
        audio_data: Optional[list] = None,  # 音频数据
    ):  # 验证多模态数据
        """
        Validate multimodal input lists per modality.
        验证每种模态的输入列表。

        Rule per modality (image/video/audio):
        - Either the list has exactly one item and that single item is a dict with
          format in {"processor_output", "precomputed_embedding"};
        - Or, the list contains only "normal" items (i.e., does not include any
          item whose format is one of the two above).

        Empty or None lists are considered valid.
        """

        BaseMultimodalProcessor._validate_one_modality(Modality.IMAGE, image_data)  # 验证图像数据
        BaseMultimodalProcessor._validate_one_modality(Modality.VIDEO, video_data)  # 验证视频数据
        BaseMultimodalProcessor._validate_one_modality(Modality.AUDIO, audio_data)  # 验证音频数据

    def _process_loaded_mm_data(self, modality, raw_data, result):  # 处理已加载的多模态数据
        """将加载的数据按模态分类"""
        images, videos, audios = [], [], []  # 初始化分类列表

        is_precomputed = self._is_preprocessed_input(raw_data)  # 判断是否预处理

        if modality == Modality.IMAGE:  # 图像模态
            if is_precomputed:  # 已预处理
                images.append(result)  # 直接添加
            else:  # 未预处理
                if isinstance(result, list):  # 如果是列表
                    images.extend(result)  # 扩展列表
                else:  # 单个图像
                    images.append(result)  # 添加
        elif modality == Modality.VIDEO:  # 视频模态
            videos.append(result)  # 添加视频
        elif modality == Modality.AUDIO:  # 音频模态
            audios.append(result)  # 添加音频

        return is_precomputed, images, videos, audios  # 返回分类结果

    async def load_mm_data(
        self,
        prompt: str,  # 提示词
        multimodal_tokens: MultimodalSpecialTokens,  # 多模态特殊 token
        image_data: Optional[list] = None,  # 图像数据
        video_data: Optional[list] = None,  # 视频数据
        audio_data: Optional[list] = None,  # 音频数据
        return_text: Optional[bool] = True,  # 是否返回文本
        discard_alpha_channel: bool = True,  # 是否丢弃 alpha 通道
        audio_sample_rate: Optional[int] = None,  # 音频采样率
    ) -> BaseMultiModalProcessorOutput:  # 返回多模态处理器输出
        """加载多模态数据，根据 token 与数据对齐情况选择快速路径或传统路径"""

        BaseMultimodalProcessor.validate_mm_data(image_data, video_data, audio_data)  # 验证数据

        input_ids = prompt if isinstance(prompt, list) else None  # 如果提示词是列表则作为 input_ids
        if input_ids is not None and self._all_mm_data_is_preprocessed(
            image_data, video_data, audio_data
        ):  # 如果所有数据都已预处理
            # fast path for preprocessed data: early return
            return BaseMultiModalProcessorOutput(  # 快速返回
                input_text="",
                input_ids=input_ids,
                images=list(image_data or []),
                videos=list(video_data or []),
                audios=list(audio_data or []),
            )

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()  # 获取组合正则
        if isinstance(prompt, list) and return_text:  # 如果提示词是列表且需要返回文本
            assert len(prompt) and isinstance(prompt[0], int)  # 验证列表非空且为整数
            prompt = self._tokenizer.decode(prompt)  # 解码为文本
        else:
            prompt = prompt  # 保持原样

        assert isinstance(prompt, str)  # 确保提示词为字符串
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)  # 按特殊 token 分割文本

        cnt = {Modality.IMAGE: 0, Modality.VIDEO: 0, Modality.AUDIO: 0}  # 统计各模态 token 数
        for text_part in text_parts:  # 遍历文本部分
            modality = multimodal_tokens.get_modality_of_token(text_part)  # 获取模态
            if modality is not None:  # 如果是多模态 token
                cnt[modality] += 1  # 计数加一

        n_image = len(image_data) if image_data else 0  # 图像数据数
        n_video = len(video_data) if video_data else 0  # 视频数据数
        n_audio = len(audio_data) if audio_data else 0  # 音频数据数

        # For MiniCPMO and MiniCPMV or multimodal_tokens not totally align, legacy show path
        if (  # 如果 token 与数据不对齐，使用传统路径
            self.server_args.skip_tokenizer_init
            or cnt[Modality.IMAGE] != n_image
            or cnt[Modality.VIDEO] != n_video
            or cnt[Modality.AUDIO] != n_audio
            or getattr(self, "support_dynamic_frame_expansion", False)
        ):
            return await self.legacy_load_mm_data(  # 使用传统加载路径
                prompt=prompt,
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                return_text=return_text,
                discard_alpha_channel=discard_alpha_channel,
                audio_sample_rate=audio_sample_rate,
                input_ids=input_ids,
            )
        # For models other than MiniCPMO and MiniCPMV,
        # totally align multimodal_tokens, fast path
        return await self.fast_load_mm_data(  # 使用快速加载路径
            prompt=prompt,
            multimodal_tokens=multimodal_tokens,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            return_text=return_text,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
            input_ids=input_ids,
        )

    async def fast_load_mm_data(
        self,
        prompt: str,  # 提示词
        multimodal_tokens: MultimodalSpecialTokens,  # 多模态特殊 token
        image_data: Optional[list] = None,  # 图像数据
        video_data: Optional[list] = None,  # 视频数据
        audio_data: Optional[list] = None,  # 音频数据
        return_text: Optional[bool] = True,  # 是否返回文本
        discard_alpha_channel: bool = True,  # 是否丢弃 alpha 通道
        audio_sample_rate: Optional[int] = None,  # 音频采样率
        input_ids: Optional[Union[List[int], torch.Tensor]] = None,  # 输入 ID
    ) -> BaseMultiModalProcessorOutput:  # 返回多模态处理器输出
        """
        A fast version of `load_mm_data` that loads multimodal data directly.
        `load_mm_data` 的快速版本，直接加载多模态数据。

        This version does not scan the prompt to recognize tokens. It assumes
        that the caller has already aligned the tokens and data in a 1:1 manner.
        The behavior is as follows:
          1. It runs `_load_single_item` for all input data concurrently.
          2. It returns the loaded images, videos, and audios in their original order.
          3. It returns the input prompt as a string.
        """

        # Convert prompt into str
        if isinstance(prompt, list) and return_text:  # 如果提示词是列表且需要文本
            assert len(prompt) and isinstance(prompt[0], int)  # 验证列表非空且为整数
            prompt_str = self._tokenizer.decode(prompt)  # 解码为文本
        else:
            assert isinstance(prompt, str)  # 确保为字符串
            prompt_str = prompt  # 直接使用

        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []  # 任务列表

        modalities_data = [  # 模态数据列表
            (image_data, Modality.IMAGE),
            (video_data, Modality.VIDEO),
            (audio_data, Modality.AUDIO),
        ]

        for data_list, modality in modalities_data:  # 遍历每种模态
            futures.extend(
                self._submit_mm_data_loading_tasks_simple(
                    data_list, modality, audio_sample_rate, discard_alpha_channel
                )
            )  # 提交加载任务

        logger.debug("[load_mm_data(simple)] total futures submitted: %d", len(futures))  # 记录调试日志

        images: List[Any] = [None] * len(image_data) if image_data else []  # 初始化图像列表
        videos: List[Any] = [None] * len(video_data) if video_data else []  # 初始化视频列表
        audios: List[Any] = [None] * len(audio_data) if audio_data else []  # 初始化音频列表

        for modality, idx, future in futures:  # 遍历所有任务
            try:
                result = await asyncio.wrap_future(future)  # 等待任务完成
            except Exception as e:  # 捕获异常
                logger.exception(
                    "[load_mm_data(simple)] error loading %s data at index=%d",
                    modality.name,
                    idx,
                )  # 记录异常日志
                raise RuntimeError(
                    f"An exception occurred while loading {modality.name} data at index {idx}: {e}"
                )  # 抛出运行时异常

            if modality == Modality.IMAGE:  # 图像模态
                images[idx] = result  # 按索引放置
            elif modality == Modality.VIDEO:  # 视频模态
                videos[idx] = result  # 按索引放置
            elif modality == Modality.AUDIO:  # 音频模态
                audios[idx] = result  # 按索引放置

        logger.debug(
            "[load_mm_data(simple)] loaded counts: images=%d, videos=%d, audios=%d",
            len(images),
            len(videos),
            len(audios),
        )  # 记录加载统计

        return BaseMultiModalProcessorOutput(  # 返回输出
            images=images,
            audios=audios,
            videos=videos,
            input_text=prompt_str,
            input_ids=input_ids,
        )

    async def legacy_load_mm_data(
        self,
        prompt: str,  # 提示词
        multimodal_tokens: MultimodalSpecialTokens,  # 多模态特殊 token
        image_data: Optional[list] = None,  # 图像数据
        video_data: Optional[list] = None,  # 视频数据
        audio_data: Optional[list] = None,  # 音频数据
        return_text: Optional[bool] = True,  # 是否返回文本
        discard_alpha_channel: bool = True,  # 是否丢弃 alpha 通道
        audio_sample_rate: Optional[int] = None,  # 音频采样率
        input_ids: Optional[Union[List[int], torch.Tensor]] = None,  # 输入 ID
    ) -> BaseMultiModalProcessorOutput:  # 返回多模态处理器输出
        """
        Each frame of video/image will be replaced by a single image token
        每个视频/图像帧将被替换为单个图像 token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()  # 获取组合正则
        if isinstance(prompt, list) and return_text:  # 如果提示词是列表且需要文本
            assert len(prompt) and isinstance(prompt[0], int)  # 验证列表
            prompt = self._tokenizer.decode(prompt)  # 解码为文本
        else:
            prompt = prompt  # 保持原样

        assert isinstance(prompt, str)  # 确保为字符串
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)  # 按特殊 token 分割文本
        # collect all data
        data_iterators = {}  # 数据迭代器字典
        if multimodal_tokens.image_token and image_data:  # 如果有图像 token 和数据
            data_iterators[Modality.IMAGE] = iter(image_data)  # 创建图像迭代器
        if multimodal_tokens.video_token and video_data:  # 如果有视频 token 和数据
            data_iterators[Modality.VIDEO] = iter(video_data)  # 创建视频迭代器
        if multimodal_tokens.audio_token and audio_data:  # 如果有音频 token 和数据
            data_iterators[Modality.AUDIO] = iter(audio_data)  # 创建音频迭代器

        # futures: the futures of loaded data
        # task_info: modality, raw_data, and other metadata of each data
        futures, task_info = self.submit_data_loading_tasks(  # 提交数据加载任务
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            data_iterators=data_iterators,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
        )
        task_info_iter = iter(task_info)  # 创建任务信息迭代器
        futures_iter = iter(futures)  # 创建 future 迭代器

        # Process results
        images, videos, audios = [], [], []  # 初始化分类列表
        new_text_parts = []  # 存储新的文本部分
        has_precomputed_input = False  # 是否有预处理输入
        for text_part in text_parts:  # 遍历文本部分
            try:
                if multimodal_tokens_pattern.match(text_part):  # 如果匹配多模态 token
                    modality, raw_data, frame_limit = next(task_info_iter)  # 获取任务信息
                    result = await asyncio.wrap_future(next(futures_iter))  # 等待加载完成

                    is_precomputed, new_imgs, new_vids, new_auds = (
                        self._process_loaded_mm_data(modality, raw_data, result)
                    )  # 处理加载的数据

                    has_precomputed_input |= is_precomputed  # 更新预处理标志
                    images.extend(new_imgs)  # 扩展图像列表
                    videos.extend(new_vids)  # 扩展视频列表
                    audios.extend(new_auds)  # 扩展音频列表

                    if modality == Modality.IMAGE:  # 图像模态
                        if is_precomputed:  # 已预处理
                            new_text_parts += [text_part]  # 保留原始 token
                        else:  # 未预处理
                            count = len(new_imgs)  # 获取图像数量
                            if count > 0:  # 如果有图像
                                new_text_parts += [
                                    multimodal_tokens.image_token
                                ] * count  # 用图像 token 替换
                    elif modality == Modality.VIDEO:  # 视频模态
                        # load as video
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.video_token
                        )  # 选择 token
                        new_text_parts += mm_tokens  # 添加到文本
                    elif modality == Modality.AUDIO:  # 音频模态
                        # audio
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.audio_token
                        )  # 选择 token
                        new_text_parts += mm_tokens  # 添加到文本
                else:  # 普通文本
                    # normal text
                    new_text_parts += [text_part]  # 直接添加

            except StopIteration as e:  # 迭代器耗尽
                # when precomputed_input is presented with multi-images, StopIteration is expected
                if has_precomputed_input:  # 如果有预处理输入
                    new_text_parts += [text_part]  # 保留原始文本
                    continue  # 继续
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )  # 抛出异常
            except Exception as e:  # 其他异常
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )  # 抛出异常
        return BaseMultiModalProcessorOutput(  # 返回输出
            images=images,
            audios=audios,
            videos=videos,
            input_text="".join(new_text_parts),
            input_ids=input_ids,
        )

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:  # 返回偏移区间列表
        """
        Get a set of range for mm_items from input_ids
        从 input_ids 中获取多模态项的偏移区间。

        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        mask = input_ids == mm_token_id  # 创建布尔掩码
        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]  # 找到连续区间的起始位置
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]  # 找到连续区间的结束位置
        return list(zip(start_positions.tolist(), end_positions.tolist()))  # 返回偏移区间列表

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:  # 返回偏移区间列表
        """通过起始和结束 token ID 获取多模态项的偏移区间"""
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1  # 起始位置（跳过起始 token）
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1  # 结束位置（跳过结束 token）

        return list(zip(indices_start.tolist(), indices_end.tolist()))  # 返回偏移区间列表

    def collect_mm_items_from_processor_output(
        self, data_dict: dict, modality: Modality = None
    ) -> List[MultimodalDataItem]:  # 返回多模态数据项列表
        """
        Create mm_items from processor output.
        从处理器输出创建多模态数据项。

        Initially creates one item per modality; these are later split into per-image/video items by get_new_expanded_mm_items.

        Note that the data_dict can be hf processor output, or passed via offline engine api

        Args:
            modality: if provided, force the data into a single MultimodalDataItem of that modality
        """

        # universal getter for data_dict
        get_data_value = (  # 通用数据获取函数
            data_dict.get
            if hasattr(data_dict, "get")
            else lambda name, default=None: getattr(data_dict, name, default)
        )

        # decide explicitly-set modality
        explicit_modality = modality  # 显式指定的模态
        modality_value = get_data_value("modality")  # 获取数据中的模态值
        if explicit_modality is None and modality_value is not None:  # 如果未显式指定但有模态值
            explicit_modality = (
                modality_value
                if isinstance(modality_value, Modality)
                else Modality.from_str(str(modality_value))
            )  # 转换模态值

        items: dict[Modality, MultimodalDataItem] = {}  # 存储各模态的数据项
        for attr_name, value in data_dict.items():  # 遍历数据字典
            if attr_name in (  # 跳过元数据字段
                "input_ids",
                "format",
                "modality",
                "hash",
                "pad_value",
                "offsets",
            ):
                # metadata fields need explicit handling, skip generic item.set
                continue

            # Get modality for this attribute
            current_modality = explicit_modality or self.ATTR_NAME_TO_MODALITY.get(
                attr_name
            )  # 获取当前属性的模态

            if attr_name == "precomputed_embeddings":  # 预计算嵌入
                current_modality = current_modality or Modality.IMAGE  # 默认为图像模态

            if current_modality:  # 如果确定了模态
                # Create item if needed
                if current_modality not in items:  # 如果该模态尚无数据项
                    items[current_modality] = MultimodalDataItem(
                        modality=current_modality
                    )  # 创建数据项

                if attr_name in self.FEATURE_NAMES:  # 如果是特征属性
                    attr_name = "feature"  # 统一重命名为 feature

                items[current_modality].set(attr_name, value)  # 设置属性值

        # deal with metadata fields when data_dict is preprocessed input: convert from tensor to expected python types
        # the attribution of the metadata fields is only clear when number of MultimodalDataItem is 1
        if len(items) == 1:  # 如果只有一个数据项
            item = next(iter(items.values()))  # 获取数据项

            # adjust offset
            offsets = get_data_value("offsets")  # 获取偏移
            if offsets is not None:  # 如果偏移存在
                if isinstance(offsets, torch.Tensor):  # 如果是张量
                    offsets = offsets.detach().cpu().tolist()  # 转为列表
                item.offsets = [(int(start), int(end)) for start, end in offsets]  # 转换偏移格式

            # adjust hash_value
            hash_value = get_data_value("hash")  # 获取哈希值
            if hash_value is not None:  # 如果哈希值存在
                if isinstance(hash_value, torch.Tensor):  # 如果是张量
                    hash_value = hash_value.item()  # 转为标量
                item.hash = int(hash_value)  # 设置哈希值
                pad_value = get_data_value("pad_value")  # 获取填充值
                if pad_value is not None:  # 如果填充值存在
                    if isinstance(pad_value, torch.Tensor):  # 如果是张量
                        pad_value = pad_value.item()  # 转为标量
                    item.pad_value = int(pad_value)  # 设置填充值

        return list(items.values())  # 返回数据项列表

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:  # 返回 (数据项, input_ids, 原始结果)
        """
        Helper method to process multimodal data and create mm_items in one step.
        辅助方法，一步完成多模态数据处理和数据项创建。

        Returns:
            Tuple of (created mm_items, input_ids)
        """
        ret = self.process_mm_data(  # 处理多模态数据
            input_text=input_text, images=images, audios=audios, videos=videos, **kwargs
        )

        input_ids = ret["input_ids"].flatten()  # 展平 input_ids
        collected_items = self.collect_mm_items_from_processor_output(ret)  # 收集数据项

        return collected_items, input_ids, ret  # 返回数据项、input_ids 和原始结果

    @staticmethod
    def _ensure_input_ids_is_tensor(input_ids) -> Optional[torch.Tensor]:  # 确保 input_ids 为展平的张量
        """make sure the input_ids is a flattened tensor
        确保 input_ids 是展平的长整型张量。"""
        if input_ids is None:  # 如果为 None
            return None  # 返回 None
        if isinstance(input_ids, torch.Tensor):  # 如果已是张量
            return input_ids.flatten().to(dtype=torch.long)  # 展平并转为长整型
        return torch.tensor(input_ids, dtype=torch.long).flatten()  # 从列表创建并展平

    def _wrap_tensor_for_cuda_ipc(self, tensor: torch.Tensor):  # 将张量包装为 CUDA IPC 可传输对象
        """helper function to turn a tensor into a cuda-ipc tensor
        辅助函数，将张量转换为 CUDA IPC 可传输对象。"""
        if not tensor.is_cuda:  # 如果不在 CUDA 上
            return tensor  # 直接返回

        sync_flag, available_slice, byte_offset = (
            self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(tensor)
        )  # 从内存池获取切片
        if isinstance(available_slice, torch.Tensor):  # 如果获取到切片
            available_slice.copy_(tensor.view(torch.int8).view(-1), non_blocking=True)  # 异步复制数据
            return CudaIpcTensorTransportProxy(  # 返回 IPC 代理对象
                data=available_slice,
                info_data=tensor,
                sync_buffer_meta=sync_flag,
                pool_ipc_handle=(
                    self.cudaipc_mmfeature_pool._pool_ipc_handle
                    if _IPC_POOL_HANDLE_CACHE
                    else None
                ),
                pool_byte_offset=byte_offset,
                pool_device_index=self.cudaipc_mmfeature_pool._pool_device_index,
            )
        if self.server_args.keep_mm_feature_on_device:  # 如果需要保留在设备上
            return tensor  # 返回原始张量
        return tensor.cpu()  # 否则移到 CPU

    def resolve_image_token_counts(self, images: List) -> List[int]:  # 解析每张图像的 token 数
        """Per-image expanded token counts, computed without re-tokenizing.
        计算每张图像扩展后的 token 数，无需重新分词。

        Default implementation uses the transformers in-tree convention
        ``_get_num_multimodal_tokens(image_sizes=...)`` (present on the in-tree
        VLM processors, e.g. Qwen-VL, Gemma3, GLM4V). Models whose processor
        does not implement it (e.g. Kimi) override this method.

        """
        assert images is not None  # 确保图像列表不为空
        image_sizes = [(image.height, image.width) for image in images]  # 获取每张图像的尺寸
        num_image_tokens = self._processor._get_num_multimodal_tokens(
            image_sizes=image_sizes
        ).num_image_tokens  # 调用处理器获取 token 数
        return [int(count) for count in num_image_tokens]  # 转为整数列表

    @staticmethod
    def _expand_input_ids(
        original_ids: List[int],  # 原始 token ID 列表
        counts: List[int],  # 每个占位符的扩展计数
        placeholder_token_id: Optional[int],  # 占位符 token ID
    ) -> List[int]:  # 返回重建后的 input_ids
        """Rebuild final input_ids for a pre-tokenized (list[int]) prompt.
        为预分词的提示词重建最终的 input_ids。

        Keep the user's ORIGINAL tokens verbatim and expand the i-th image
        placeholder into ``counts[i]`` copies of ``placeholder_token_id``. The HF
        processor's re-tokenization is discarded, so non-media tokens cannot
        drift.
        保留用户的原始 token，将第 i 个图像占位符扩展为 counts[i] 个占位符 token，
        丢弃 HF 处理器的重新分词结果以避免非媒体 token 偏移。

        """
        if placeholder_token_id is None:  # 如果未设置占位符 token ID
            raise ValueError("placeholder_token_id is not set for this processor")  # 抛出异常

        num_placeholders = sum(
            1 for token_id in original_ids if token_id == placeholder_token_id
        )  # 统计占位符数量
        if num_placeholders != len(counts):  # 如果数量不匹配
            raise ValueError(
                f"prompt has {num_placeholders} image placeholder token(s) but "
                f"{len(counts)} image(s) were provided"
            )  # 抛出异常

        rebuilt: List[int] = []  # 重建的 input_ids
        next_image_idx = 0  # 下一个图像索引
        for token_id in original_ids:  # 遍历原始 token
            if token_id == placeholder_token_id:  # 如果是占位符
                rebuilt.extend([placeholder_token_id] * counts[next_image_idx])  # 扩展为对应数量
                next_image_idx += 1  # 递增索引
            else:  # 普通 token
                rebuilt.append(token_id)  # 直接添加
        return rebuilt  # 返回重建结果

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,  # 基础输出
        mm_tokens: MultimodalSpecialTokens,  # 多模态特殊 token
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:  # 返回 (数据项, input_ids, 原始结果)
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).
        处理多模态数据并返回组合的多模态数据项和 input_ids，支持混合模态。

        Returns:
            Tuple of (list of mm_items, input_ids)
        """
        # Collect all items and categorize them
        all_loaded_data = base_output.organize_results()  # 组织所有加载数据
        # Handle text-only case
        if not all_loaded_data:  # 如果没有多模态数据
            input_ids = self._tokenizer(  # 对文本进行分词
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return [], input_ids, {}  # 返回空数据项

        dict_items, raw_images, raw_audios, raw_videos = [], [], [], []  # 分类列表
        for modality, item in all_loaded_data:  # 遍历所有数据
            if isinstance(item, dict):  # 如果是字典（预处理数据）
                dict_items.append((modality, item))  # 添加到字典项列表
            elif modality == Modality.IMAGE:  # 图像模态
                raw_images.append(item)  # 添加到原始图像列表
            elif modality == Modality.AUDIO:  # 音频模态
                raw_audios.append(item)  # 添加到原始音频列表
            elif modality == Modality.VIDEO:  # 视频模态
                raw_videos.append(item)  # 添加到原始视频列表
            else:  # 未知模态
                raise ValueError(f"Unknown multimodal item type: {type(item)}")  # 抛出异常
        # Process items and get input_ids
        all_collected_items: list[MultimodalDataItem] = []  # 存储所有数据项
        input_ids = None  # 初始化 input_ids
        # Handle raw items (need processing)
        if raw_images or raw_audios or raw_videos:  # 如果有原始数据需要处理
            collected_items, input_ids, ret = self._process_and_collect_mm_items(
                input_text=base_output.input_text,
                images=raw_images,
                audios=raw_audios,
                videos=raw_videos,
                **kwargs,
            )  # 处理并收集数据项
            all_collected_items = collected_items  # 保存数据项

            # When SGLANG_MM_AVOID_RETOKENIZE is on, keep the user's exact tokens to avoid retokenize drift.
            # Drift happens when Retokenization is not identity: Decode(X) => String => Re-tokenize => Y, X != Y.
            if (  # 如果启用避免重新分词
                envs.SGLANG_MM_AVOID_RETOKENIZE.get()
                and base_output.input_ids is not None
                and input_ids is not None
                and raw_images
                and not raw_audios
                and not raw_videos
            ):
                assert isinstance(
                    base_output.input_ids, list
                ), f"expected list[int] input_ids, got {type(base_output.input_ids)}"  # 验证类型
                try:
                    counts = self.resolve_image_token_counts(raw_images)  # 解析图像 token 数
                    image_placeholder_token_id = mm_tokens.image_token_id  # 获取占位符 token ID
                    if image_placeholder_token_id is None:  # 如果未设置
                        raise ValueError(
                            "image placeholder token id is not set for this processor"
                        )  # 抛出异常
                    processor_placeholder_count = int(
                        (input_ids == image_placeholder_token_id).sum().item()
                    )  # 统计处理器的占位符数量
                    if processor_placeholder_count != sum(counts):  # 如果数量不匹配
                        raise ValueError(
                            "processor image placeholder count mismatch: "
                            f"processor={processor_placeholder_count}, "
                            f"resolved={sum(counts)}"
                        )  # 抛出异常
                    input_ids = torch.tensor(
                        self._expand_input_ids(
                            base_output.input_ids,
                            counts,
                            image_placeholder_token_id,
                        ),
                        dtype=input_ids.dtype,
                    )  # 使用扩展后的 input_ids
                except Exception as e:  # 捕获异常
                    logger.warning(
                        f"Due to {e}, falling back to decode+retokenize, which may change prompt length (token drift)."
                    )  # 记录警告
        else:
            ret = None  # 无原始数据时结果为 None

        # Handle dict items (processed or precomputed)
        dict_ret = None  # 字典数据结果
        for modality, dict_item in dict_items:  # 遍历字典数据
            input_format = self._get_preprocessed_input_format(dict_item)  # 获取输入格式
            if input_format is not None and dict_ret is None:  # 如果是预处理数据
                dict_ret = dict_item  # 保存第一个预处理数据
            if input_format == MultimodalInputFormat.PROCESSOR_OUTPUT:  # 处理器输出格式
                items = self.collect_mm_items_from_processor_output(dict_item)  # 收集数据项
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT  # 设置格式
                all_collected_items.extend(items)  # 添加到总列表
            elif input_format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:  # 预计算嵌入格式
                dict_item = dict(dict_item)  # 转为字典
                feature = dict_item.pop("feature")  # 提取特征
                all_collected_items.append(
                    MultimodalDataItem(
                        modality=modality,
                        feature=feature,
                        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                        model_specific_data=dict_item,
                    )
                )  # 创建数据项并添加
        # Fallback tokenization if no raw items were processed
        if ret is None and dict_ret is not None:  # 如果没有原始数据但有预处理数据
            ret = dict_ret  # 使用预处理数据作为结果

        if input_ids is None:  # 如果 input_ids 仍为 None
            input_ids = self._ensure_input_ids_is_tensor(base_output.input_ids)  # 尝试从基础输出获取

        if input_ids is None:  # 如果仍然为 None
            for _, dict_item in dict_items:  # 遍历字典数据
                input_ids = self._ensure_input_ids_is_tensor(dict_item.get("input_ids"))  # 尝试获取
                if input_ids is not None:  # 如果获取到
                    break  # 跳出循环

        if input_ids is None:  # 如果仍然为 None
            input_ids = self._tokenizer(  # 对文本进行分词
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        for mm_item in all_collected_items:  # 遍历所有数据项
            if mm_item.offsets is not None:  # 如果偏移已存在
                continue  # 跳过
            mm_token_id = mm_tokens.get_token_id_by_modality(mm_item.modality)  # 获取 token ID
            if mm_token_id is None:  # 如果未找到
                raise ValueError(f"No token id found for modality: {mm_item.modality}")  # 抛出异常
            mm_item.offsets = self.get_mm_items_offset(  # 计算偏移
                input_ids=input_ids,
                mm_token_id=mm_token_id,
            )

        # Split bundled items into per-image/video items for better cache granularity
        from sglang.srt.managers.mm_utils import get_new_expanded_mm_items  # 导入扩展工具

        all_collected_items = get_new_expanded_mm_items(all_collected_items)  # 扩展数据项

        for item in all_collected_items:  # 遍历所有数据项
            if item.format in (  # 如果格式为预处理类型
                MultimodalInputFormat.PROCESSOR_OUTPUT,
                MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
            ):
                item.set_pad_value()  # 设置填充值

        """
        solution for cuda-ipc memory-leak:
        1. memory-pool:  each time get a slice from memory-pool and use it as transport-data (with async lock guard)
        2. if can not get a slice , transport normal tensor
        3. copy tensor in scheduler and release it (use position mark)
        4. copy
        """

        if SGL_USE_CUDA_IPC:  # 如果使用 CUDA IPC
            # post-process, prepare for cuda-ipc transfer
            for item in all_collected_items:  # 遍历所有数据项
                if isinstance(item.feature, torch.Tensor):  # 如果特征是张量
                    item.feature = self._wrap_tensor_for_cuda_ipc(item.feature)  # 包装为 IPC 对象
                if isinstance(item.precomputed_embeddings, torch.Tensor):  # 如果预计算嵌入是张量
                    item.precomputed_embeddings = self._wrap_tensor_for_cuda_ipc(
                        item.precomputed_embeddings
                    )  # 包装为 IPC 对象

        return all_collected_items, input_ids, ret  # 返回数据项、input_ids 和原始结果
