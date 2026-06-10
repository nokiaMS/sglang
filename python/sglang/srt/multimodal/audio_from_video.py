# 本文件用于从视频字节数据中提取音频波形，使用 PyAV 库在进程内完成，避免子进程 fork 导致 CUDA 崩溃
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
"""Extract audio from video bytes using PyAV (in-process, CUDA-safe).

PyAV wraps FFmpeg's C libraries in-process, avoiding subprocess forks which
would crash CUDA-active workers.
"""

import io  # 导入 io 模块，用于将字节流包装为文件对象
import logging  # 导入 logging 模块，用于记录日志信息

import numpy as np  # 导入 numpy 模块，用于音频波数组操作

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


def extract_audio_from_video_bytes(
    video_bytes: bytes,  # 视频文件的原始字节数据（如 MP4 格式）
    target_sr: int = 16000,  # 目标采样率，默认为 16000 Hz
) -> np.ndarray | None:  # 返回一维 float32 音频数组，若无音频轨则返回 None
    """Extract mono audio from video bytes at the target sample rate.
    从视频字节数据中提取单声道音频，并以目标采样率输出。

    Args:
        video_bytes: Raw video file bytes (e.g. MP4).
        target_sr: Target sample rate for the output waveform.

    Returns:
        1-D float32 numpy array of audio samples, or None if the video
        has no audio track.
    """
    try:  # 尝试导入 PyAV 库
        import av  # 导入 PyAV 库
    except ImportError:  # 如果 PyAV 未安装
        logger.warning(  # 记录警告日志
            "PyAV (av) is not installed. Cannot extract audio from video. "
            "Install with: pip install av"
        )
        return None  # 返回 None 表示无法提取音频

    try:  # 尝试打开视频字节流
        container = av.open(io.BytesIO(video_bytes))  # 将视频字节流包装为 BytesIO 并用 PyAV 打开
    except Exception:  # 如果打开失败
        logger.warning("Failed to open video bytes for audio extraction")  # 记录警告
        return None  # 返回 None

    if not container.streams.audio:  # 检查视频是否包含音频流
        container.close()  # 关闭容器
        return None  # 无音频轨，返回 None

    try:  # 尝试提取音频
        audio_stream = container.streams.audio[0]  # 获取第一个音频流
        native_sr = audio_stream.rate or target_sr  # 获取原始采样率，若未知则使用目标采样率

        resampler = av.audio.resampler.AudioResampler(  # 创建音频重采样器
            format="flt",  # 输出格式为 float32
            layout="mono",  # 输出布局为单声道
            rate=target_sr,  # 目标采样率
        )

        chunks = []  # 用于存储重采样后的音频块
        for frame in container.decode(audio=0):  # 逐帧解码音频
            resampled = resampler.resample(frame)  # 对帧进行重采样
            for rf in resampled:  # 遍历重采样后的帧
                arr = rf.to_ndarray().flatten()  # 转换为一维 numpy 数组
                chunks.append(arr)  # 将音频块添加到列表

        container.close()  # 关闭视频容器

        if not chunks:  # 如果没有提取到任何音频块
            return None  # 返回 None

        waveform = np.concatenate(chunks).astype(np.float32)  # 将所有音频块拼接为一维 float32 数组
        return waveform  # 返回音频波形

    except Exception:  # 如果提取过程中发生异常
        logger.warning("Error extracting audio from video", exc_info=True)  # 记录警告及异常信息
        container.close()  # 关闭视频容器
        return None  # 返回 None
