# Gemma4 多模态处理器模块
# 本模块实现了 Gemma4 模型的多模态数据处理逻辑，
# 支持图像、视频和音频输入，包括 GPU 加速的图像预处理、
# 视频帧采样、音频波形填充对齐等功能。
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

from typing import Dict, List, Optional, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 模块
import torch  # 导入 PyTorch 模块

from sglang.srt.managers.multimodal_processor import (  # 导入多模态处理器模块
    BaseMultimodalProcessor as SGLangBaseProcessor,  # 将基础处理器重命名为 SGLangBaseProcessor
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput  # 导入模态枚举和多模态处理器输出类
from sglang.srt.models.gemma4_audio import _SSCP_CONV_STRIDE_SIZES  # 导入 SSCP 卷积步幅大小配置
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration  # 导入 Gemma4 条件生成模型类
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens  # 导入多模态特殊标记类
from sglang.srt.utils.video_decoder import VideoDecoderWrapper  # 导入视频解码器包装类


class Gemma4SGLangProcessor(SGLangBaseProcessor):  # Gemma4 SGLang 处理器类
    """Multimodal processor for Gemma4 supporting image, video, and audio inputs."""
    """Gemma4 的多模态处理器，支持图像、视频和音频输入。"""

    models = [Gemma4ForConditionalGeneration]  # 支持的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 Gemma4 处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法

        self.IM_START_TOKEN_ID = hf_config.boi_token_id  # 图像起始标记的 token ID
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id  # 图像结束标记的 token ID

        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id  # 音频起始标记的 token ID
        self.AUDIO_END_TOKEN_ID = hf_config.eoa_token_id  # 音频结束标记的 token ID
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token_id=hf_config.image_token_id,  # 图像标记 token ID
            video_token_id=hf_config.video_token_id,  # 视频标记 token ID
            audio_token_id=hf_config.audio_token_id,  # 音频标记 token ID
        ).build(_processor)  # 使用处理器构建标记

        # Register image-processor and video-processor outputs so they are stored on
        # MultimodalDataItem via collect_mm_items_from_processor_output.
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE  # 注册图像位置 ID 到图像模态
        self.ATTR_NAME_TO_MODALITY["video_position_ids"] = Modality.VIDEO  # 注册视频位置 ID 到视频模态

    def _get_audio_pad_multiple(self) -> int:  # 获取音频波形填充对齐的倍数
        """Derive the waveform padding alignment from processor config.
        从处理器配置中推导波形填充对齐倍数。

        The HF processor's ceil(duration_ms / audio_ms_per_token) formula can
        overshoot by 1 token relative to what the SSCP convolutions produce.
        Padding waveforms to a multiple of (hop_length * first_conv_stride)
        aligns the two calculations.
        HuggingFace 处理器的 ceil(duration_ms / audio_ms_per_token) 公式可能会
        比 SSCP 卷积产生的结果多出 1 个 token。将波形填充到 (hop_length * first_conv_stride)
        的倍数可以使两种计算对齐。
        See: gemma-4-eap-extras/examples/gemma-4-audio-examples.ipynb
        """
        fe = getattr(self._processor, "feature_extractor", None)  # 获取特征提取器
        hop = getattr(fe, "hop_length", 160)  # 获取跳跃长度，默认为 160
        first_stride = _SSCP_CONV_STRIDE_SIZES[0][0]  # 获取第一个 SSCP 卷积的步幅大小
        return hop * first_stride  # 返回填充倍数

    def _video_decoder_to_tensor(self, vdw: VideoDecoderWrapper) -> torch.Tensor:  # 将视频解码器包装对象转换为张量
        """Convert a VideoDecoderWrapper to a (sampled_frames, C, H, W) uint8 tensor.
        将 VideoDecoderWrapper 转换为 (sampled_frames, C, H, W) uint8 张量。

        SGLang's load_video returns VideoDecoderWrapper which the HF
        Gemma4VideoProcessor does not recognise (expects torch.Tensor or
        np.ndarray).  We replicate HF's uniform frame sampling here to
        avoid materialising the entire video in memory, then delegate the
        rest (resize, patchify, position IDs) to the HF video processor.
        SGLang 的 load_video 返回 VideoDecoderWrapper，HuggingFace 的
        Gemma4VideoProcessor 不识别该类型（期望 torch.Tensor 或 np.ndarray）。
        我们在此复制 HuggingFace 的均匀帧采样逻辑，避免将整个视频加载到内存中，
        然后将其余操作（缩放、分块、位置 ID）委托给 HuggingFace 的视频处理器。
        """
        total = len(vdw)  # 获取视频总帧数
        num_frames = getattr(  # 获取目标帧数
            getattr(self._processor, "video_processor", None),  # 获取视频处理器
            "num_frames",  # 获取帧数属性
            32,  # 默认帧数为 32
        )
        if total <= num_frames:  # 如果视频总帧数不超过目标帧数
            indices = list(range(total))  # 使用所有帧
        else:  # 否则均匀采样
            indices = torch.arange(0, total, total / num_frames).int().tolist()  # 计算均匀采样的帧索引
        frames_np = vdw.get_frames_at(indices)  # 获取指定帧的 numpy 数组  # (N, H, W, C)
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()  # 转换为 (N, C, H, W) 格式的张量

    def process_mm_data(  # 处理多模态数据，对音频和视频进行预处理
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        if audios:  # 如果有音频数据
            pad_multiple = self._get_audio_pad_multiple()  # 获取音频填充对齐倍数
            padded = []  # 初始化填充后的音频列表
            for a in audios:  # 遍历每个音频
                a = np.asarray(a)  # 将音频转换为 NumPy 数组
                remainder = len(a) % pad_multiple  # 计算当前音频长度的余数
                if remainder != 0:  # 如果需要填充
                    a = np.pad(a, (0, pad_multiple - remainder), mode="constant")  # 用零填充到对齐倍数
                padded.append(a)  # 添加到填充列表
            audios = padded  # 替换为填充后的音频列表
        if videos:  # 如果有视频数据
            videos = [  # 转换视频格式
                (
                    self._video_decoder_to_tensor(v)  # 将 VideoDecoderWrapper 转换为张量
                    if isinstance(v, VideoDecoderWrapper)  # 检查是否为 VideoDecoderWrapper 类型
                    else v  # 否则保持原样
                )
                for v in videos  # 遍历所有视频
            ]
            kwargs.setdefault("do_sample_frames", False)  # 设置不进行帧采样（已经手动采样了）
        return super().process_mm_data(  # 调用父类处理多模态数据
            input_text, images=images, videos=videos, audios=audios, **kwargs  # 传入处理后的参数
        )

    async def process_mm_data_async(  # 异步处理多模态数据，包括图像、视频和音频
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,  # 可选的图像数据列表
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,  # 可选的音频数据列表
        input_text: str = "",  # 输入文本，默认为空字符串
        request_obj=None,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        """Process multimodal data including images, video, and audio."""
        """处理包括图像、视频和音频在内的多模态数据。"""
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            video_data=request_obj.video_data if request_obj else None,  # 视频数据（从请求对象获取）
            audio_data=audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
            video_token_id=self.mm_tokens.video_token_id,  # 视频标记 token ID
            audio_token_id=self.mm_tokens.audio_token_id,  # 音频标记 token ID
        )
