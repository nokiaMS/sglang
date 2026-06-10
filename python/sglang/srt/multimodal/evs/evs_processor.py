# 本文件实现 EVS（Efficient Video Sampling）处理器，负责构建带有正确占位符 token 数的提示词，
# 当 EVS 启用时根据剪枝率分配更少的占位符，未启用时使用完整的 token 数
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


import torch  # 导入 PyTorch 深度学习框架
from transformers import PretrainedConfig  # 导入 Hugging Face 预训练配置基类

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem  # 导入模态枚举和多模态数据项
from sglang.utils import logger  # 导入日志记录器

from .evs_core import tokens_per_frame  # 导入每帧 token 数计算函数
from .evs_module import EVS, EVSConfig, EVSDataItem, VideoEVSDataItem  # 导入 EVS 模块核心类


def _non_evs_data_items(
    *,  # 以下参数必须以关键字参数形式传入
    image: torch.Tensor | None,  # 图像特征张量
    image_offsets: list[tuple[int, int]],  # 图像偏移列表
    video: torch.Tensor | None,  # 视频特征张量
    video_offsets: list[tuple[int, int]],  # 视频偏移列表
    input_ids_list: list[int],  # input_ids 列表
):  # 创建非 EVS 的多模态数据项列表
    """创建不使用 EVS 的标准多模态数据项列表"""
    items: list[MultimodalDataItem] = []  # 初始化数据项列表
    if image is not None:  # 如果有图像数据
        item = MultimodalDataItem(  # 创建图像数据项
            modality=Modality.IMAGE, feature=image, offsets=image_offsets
        )
        items.append(item)  # 添加到列表
    if video is not None:  # 如果有视频数据
        item = MultimodalDataItem(  # 创建视频数据项
            modality=Modality.VIDEO, feature=video, offsets=video_offsets
        )
        items.append(item)  # 添加到列表
    return items  # 返回数据项列表


class EVSProcessor:
    """
    This processor handles prompt construction with the correct number of
    placeholder tokens per frame. When EVS is active, it allocates fewer
    placeholders based on the pruning rate. When inactive, it uses the full
    token count.
    EVS 处理器，负责构建带有正确每帧占位符 token 数的提示词。
    EVS 启用时根据剪枝率分配更少的占位符，未启用时使用完整 token 数。
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,  # Hugging Face 模型配置
        config_to_evs_model: dict[type[PretrainedConfig], type[EVS]],  # 配置类到 EVS 模型的映射
    ):
        assert len(config_to_evs_model) > 0  # 确保映射不为空
        assert all(issubclass(model, EVS) for model in config_to_evs_model.values())  # 确保所有模型都是 EVS 子类

        self.evs_config: EVSConfig | None = None  # 初始化 EVS 配置为 None

        config_name = hf_config.__class__.__name__  # 获取配置类名
        evs_model = config_to_evs_model.get(hf_config.__class__)  # 查找匹配的 EVS 模型
        if evs_model is None:  # 如果没有匹配的模型
            logger.info(
                f"[EVS] no model matches {config_name} in {config_to_evs_model}"
            )  # 记录信息日志
            return  # 直接返回
        evs_config = evs_model.create_evs_config(hf_config)  # 创建 EVS 配置
        logger.info(
            f"""[EVS] {evs_config} {'enabled' if evs_config.video_pruning_rate > 0.0 else 'disabled'} for model={evs_model.__name__}; model_config={config_name}"""
        )  # 记录 EVS 配置信息
        if evs_config.video_pruning_rate > 0.0:  # 如果剪枝率大于 0
            self.evs_config = evs_config  # 保存 EVS 配置

    def static_size_data_items(
        self, *, frames_per_video: list[int], num_images: int, rows: int, cols: int
    ):  # 为具有静态 token 数的模型创建数据项
        """helper function to create data items for models with static image and video tokens per frame
        辅助函数，为具有静态每帧 token 数的模型创建数据项"""

        frame_num_tokens = rows * cols  # 计算每帧 token 数

        if self.evs_config is None:  # 如果 EVS 未启用
            tpf = [[frame_num_tokens] * num_frames for num_frames in frames_per_video]  # 每帧使用完整 token 数
            return _non_evs_data_items, tpf  # 返回非 EVS 数据项创建函数和每帧 token 数

        def create_evs_data_items(  # 定义 EVS 数据项创建函数
            *,  # 以下参数必须以关键字参数形式传入
            input_ids_list: list[int],  # input_ids 列表
            image: torch.Tensor | None,  # 图像特征
            image_offsets: list[tuple[int, int]],  # 图像偏移
            video: torch.Tensor | None,  # 视频特征
            video_offsets: list[tuple[int, int]],  # 视频偏移
        ) -> list[MultimodalDataItem]:  # 返回数据项列表
            items = []  # 初始化数据项列表
            if image is not None:  # 如果有图像数据
                image_thw_grids = [(1, rows, cols)] * num_images  # 为每张图像创建网格尺寸
                item = EVSDataItem(  # 创建 EVS 图像数据项
                    modality=Modality.IMAGE,
                    feature=image,
                    offsets=image_offsets,
                    thw_grids=image_thw_grids,
                )
                items.append(item)  # 添加到列表
            if video is not None:  # 如果有视频数据
                video_thw_grids = [
                    (num_frames, rows, cols) for num_frames in frames_per_video
                ]  # 为每个视频创建网格尺寸
                item = VideoEVSDataItem(  # 创建视频 EVS 数据项
                    modality=Modality.VIDEO,
                    feature=video,
                    offsets=video_offsets,
                    thw_grids=video_thw_grids,
                    pre_chunked_input_ids=input_ids_list,
                )
                items.append(item)  # 添加到列表
            return items  # 返回数据项列表

        tpf = [  # 计算每帧的 token 数
            tokens_per_frame(
                q=self.evs_config.video_pruning_rate,
                num_frames=num_frames,
                frame_num_tokens=frame_num_tokens,
            )
            for num_frames in frames_per_video  # 遍历每个视频的帧数
        ]

        return create_evs_data_items, tpf  # 返回 EVS 数据项创建函数和每帧 token 数
