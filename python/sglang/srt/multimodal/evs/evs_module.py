# 本文件定义 EVS（Efficient Video Sampling）模块的核心数据结构和基类，包括
# EVSDataItem、VideoEVSDataItem、EVSEmbeddingResult、EVSConfig 以及 EVS 基类
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


import dataclasses  # 导入数据类模块
import typing  # 导入类型提示模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from dataclasses import dataclass  # 导入数据类装饰器

import torch  # 导入 PyTorch 深度学习框架
from transformers import PretrainedConfig  # 导入 Hugging Face 预训练配置基类

from sglang.srt.managers.schedule_batch import MultimodalDataItem  # 导入多模态数据项类
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult  # 导入嵌入结果类
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor  # 导入基础多模态处理器
from sglang.utils import logger  # 导入日志记录器

from .evs_core import compute_retention_mask, replace_offsets_with_tokens_per_frame  # 导入 EVS 核心函数


@dataclasses.dataclass(kw_only=True)
class EVSDataItem(MultimodalDataItem):  # EVS 数据项，继承自 MultimodalDataItem
    thw_grids: list[tuple[int, int, int]]  # 每个视频的时间-高度-宽度网格尺寸列表


@dataclasses.dataclass(kw_only=True)
class VideoEVSDataItem(EVSDataItem):  # 视频 EVS 数据项，继承自 EVSDataItem
    pre_chunked_input_ids: torch.Tensor  # 预分块的 input_ids 张量

    def __post_init__(self):  # 初始化后验证
        assert self.is_video()  # 确保模态类型为视频


@dataclass(kw_only=True)
class EVSEmbeddingResult(EmbeddingResult):  # EVS 嵌入结果，继承自 EmbeddingResult
    """
    Embedding result that includes per-frame token counts after EVS pruning.
    包含 EVS 剪枝后每帧 token 数的嵌入结果。

    After pruning, each frame retains a different number of tokens based on its
    dissimilarity to the previous frame. This metadata is needed downstream to
    adjust the input_ids placeholder spans to match the actual embedding sizes.

    Attributes:
        embedding: The pruned video embeddings tensor.
        num_tokens_per_frame: Actual retained token count for each frame.
            For example, [256, 180, 195, 256] means frame 0 kept all 256 tokens
            (first frame is never pruned), while frames 1-2 were pruned.
    """

    num_tokens_per_frame: list[int]  # 每帧保留的 token 数量列表

    def redistribute_pruned_frames_placeholders(
        self,
        input_ids: torch.Tensor,  # 输入 ID 张量
        offsets: list[tuple[int, int]],  # 帧偏移列表
        *,  # 以下参数必须以关键字参数形式传入
        item: VideoEVSDataItem,  # 视频 EVS 数据项
        extend_prefix_len: int,  # 扩展前缀长度
        extend_seq_len: int,  # 扩展序列长度
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:  # 返回 (修改后的 input_ids, 新偏移)
        """根据 EVS 剪枝结果重新分配 input_ids 中的帧占位符区间"""
        assert len(input_ids) == extend_seq_len  # 验证 input_ids 长度
        assert isinstance(
            item, VideoEVSDataItem
        ), f"Expected VideoEVSDataItem, got {type(item)}"  # 验证数据项类型
        pre_chunked_input_ids = item.pre_chunked_input_ids  # 获取预分块的 input_ids
        filler_token_id = item.pad_value  # 获取填充 token ID
        input_ids_list = replace_offsets_with_tokens_per_frame(  # 替换帧偏移区间
            pre_chunked_input_ids=pre_chunked_input_ids,
            num_tokens_per_frame=self.num_tokens_per_frame,
            frame_offsets_inclusive=offsets,
            filler_token_id=filler_token_id,
        )
        input_ids = torch.tensor(  # 将列表转为张量
            input_ids_list, dtype=input_ids.dtype, device=input_ids.device
        )
        offsets = BaseMultimodalProcessor.get_mm_items_offset(  # 重新计算偏移
            input_ids, filler_token_id
        )
        input_ids = input_ids[extend_prefix_len : extend_prefix_len + extend_seq_len]  # 截取对应区间
        assert (
            len(input_ids) == extend_seq_len
        ), f"Input ids length changed after redistribution, got {len(input_ids)} != {extend_seq_len}"  # 验证长度
        return input_ids, offsets  # 返回修改后的 input_ids 和偏移


@dataclass(frozen=True, kw_only=True)
class EVSConfig:  # EVS 配置类
    video_pruning_rate: float  # 视频剪枝率
    spatial_merge_size: int = 1  # 空间合并尺寸，默认为 1

    def __post_init__(self):  # 初始化后验证
        assert (
            self.video_pruning_rate >= 0.0 and self.video_pruning_rate < 1.0
        ), f"Video pruning rate must be between 0.0 and 1.0, got {self.video_pruning_rate=}"  # 确保剪枝率在 [0, 1) 范围内


class EVS(torch.nn.Module, ABC):  # EVS 基类，继承自 Module 和 ABC
    """
    Base class for video models that support EVS pruning.
    支持 EVS 剪枝的视频模型基类。

    Subclass this alongside your model class and implement the static `create_evs_config`.
    On initialization, if video_pruning_rate > 0, this mixin replaces the model's
    get_video_feature() method with a wrapper that applies EVS pruning.

    Example: See `NemotronH_Nano_VL_V2`
    """

    @staticmethod
    @abstractmethod
    def create_evs_config(config: PretrainedConfig) -> EVSConfig:  # 从模型配置提取 EVS 参数
        """Extract EVS parameters from model config. Must be implemented by subclass.
        从模型配置中提取 EVS 参数，子类必须实现。"""
        raise NotImplementedError

    @abstractmethod
    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:  # 获取视频特征
        """Extract EVS parameters from model config. Must be implemented by subclass.
        获取视频特征，子类必须实现。"""
        raise NotImplementedError

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        *args: typing.Any,  # 可变位置参数
        **kwargs: typing.Any,  # 可变关键字参数
    ) -> None:
        super().__init__()  # 调用父类初始化
        model_name = self.__class__.__name__  # 获取模型类名
        self.original_get_video_feature = self.get_video_feature  # 保存原始的 get_video_feature 方法
        self.evs_config = self.create_evs_config(config)  # 创建 EVS 配置
        self.evs_enabled = self.evs_config.video_pruning_rate > 0.0  # 判断是否启用 EVS
        if self.evs_enabled:  # 如果启用
            logger.info(f"[EVS] enabled for {model_name} [{self.evs_config}]")  # 记录启用日志
            self.get_video_feature = self.evs_video  # 替换为 EVS 包装方法
        else:  # 如果未启用
            logger.info(
                f"[EVS] requested on model {model_name} but is disabled for pruning_rate == 0.0."
            )  # 记录未启用日志

    def evs_video(self, items: list[MultimodalDataItem]) -> EVSEmbeddingResult:  # EVS 剪枝处理
        """
        Apply EVS pruning to video embeddings.
        对视频嵌入应用 EVS 剪枝。

        Args:
            items: List containing a single VideoEVSDataItem with video features.

        Returns:
            EVSEmbeddingResult with pruned embeddings and actual token counts per frame.
        """
        logger.debug(
            f"[EVS] beginning for model {self.__class__.__name__} [evs_config={self.evs_config=}]"
        )  # 记录调试日志
        assert len(items) == 1, f"Expected 1 item, got {len(items)}"  # 确保只有一个数据项
        item = items[0]  # 获取数据项
        assert isinstance(
            item, VideoEVSDataItem
        ), f"Expected VideoEVSDataItem with modality VIDEO, got {item}"  # 确保是视频数据项

        q = self.evs_config.video_pruning_rate  # 获取剪枝率
        merge = self.evs_config.spatial_merge_size  # 获取空间合并尺寸
        videos_features = self.original_get_video_feature([item])  # 调用原始方法获取视频特征
        if videos_features.ndim == 3:  # 如果是三维张量 [B, T, D]
            videos_features = videos_features.flatten(0, 1)  # 展平为二维 [B*T, D]
        assert videos_features.ndim == 2, videos_features.ndim  # 确保是二维张量

        final_embeddings: list[torch.Tensor] = []  # 存储最终的嵌入结果
        num_tokens_per_frame: list[int] = []  # 存储每帧的 token 数

        sizes = [(t * h * w // merge**2) for t, h, w in item.thw_grids]  # 计算每个视频的 token 数
        for single_video, video_size_thw in zip(  # 遍历每个视频
            videos_features.split(sizes),
            item.thw_grids,
            strict=True,
        ):
            retention_mask = compute_retention_mask(  # 计算保留掩码
                single_video,
                video_size_thw=video_size_thw,
                spatial_merge_size=merge,
                q=q,
            )
            preserved = single_video[retention_mask]  # 根据掩码提取保留的 token
            final_embeddings.append(preserved)  # 添加到结果列表
            num_frames = video_size_thw[0]  # 获取帧数
            tokens_per_frame = (
                retention_mask.reshape(num_frames, -1).sum(dim=-1).tolist()
            )  # 计算每帧保留的 token 数
            num_tokens_per_frame.extend(tokens_per_frame)  # 添加到列表
        final_embeddings_tensor = torch.cat(final_embeddings)  # 拼接所有嵌入
        return EVSEmbeddingResult(  # 返回 EVS 嵌入结果
            embedding=final_embeddings_tensor,
            num_tokens_per_frame=num_tokens_per_frame,
        )
