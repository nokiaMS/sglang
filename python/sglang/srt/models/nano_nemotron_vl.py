# Nano Nemotron VL 视觉语言模型实现
# 该文件实现了 Nano Nemotron VL 模型，结合 RADIO 视觉编码器和 Nemotron-H 语言模型，
# 支持图像、视频和音频输入，通过像素混洗和 MLP 投影器将视觉特征映射到语言模型空间。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nano_nemotron_vl.py

import logging  # 导入日志库 # 导入日志模块
from typing import Iterable  # 导入可迭代类型 # 导入可迭代类型

import torch  # 导入 PyTorch # 导入 PyTorch 框架
import torch.nn as nn  # 导入神经网络模块 # 导入神经网络模块

from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config  # 导入配置 # 导入模型配置类
from sglang.srt.layers.activation import ReLU2  # 导入 ReLU2 激活 # 导入 ReLU2 激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化 # 导入 RMS 归一化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具 # 导入多模态工具
    MultiModalityDataPaddingPatternTokenPairs,  # 标记对填充模式 # 标记对填充模式
    general_mm_embed_routine,  # 通用多模态嵌入流程 # 通用多模态嵌入流程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类 # 导入调度批次相关类
    Modality,  # 模态枚举 # 模态类型枚举
    MultimodalDataItem,  # 多模态数据项 # 多模态数据项
    MultimodalInputs,  # 多模态输入 # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器 # 导入默认权重加载器
from sglang.srt.models.nemotron_h import NemotronHForCausalLM  # 导入 Nemotron-H 语言模型 # 导入 Nemotron-H 因果语言模型
from sglang.srt.models.parakeet import ProjectedParakeet  # 导入 Parakeet 音频编码器 # 导入 Parakeet 音频编码器
from sglang.srt.models.radio import RadioModel  # 导入 RADIO 视觉模型 # 导入 RADIO 视觉模型
from sglang.srt.models.utils import WeightsMapper  # 导入权重映射器 # 导入权重映射器
from sglang.srt.multimodal.evs import EVS, EVSConfig  # 导入 EVS 基类和配置 # 导入 EVS 基类和配置
from sglang.srt.multimodal.evs.evs_module import VideoEVSDataItem  # 导入视频 EVS 数据项 # 导入视频 EVS 数据项
from sglang.srt.utils import add_prefix  # 导入前缀工具 # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取日志器 # 获取模块日志器


class NemotronH_Nano_VL_V2(EVS):  # Nano Nemotron VL V2 模型 # 继承自 EVS 基类
    # The loader reads `hf_to_sglang_mapper` off the outer model class when
    # applying name rewrites to the quant config's `quantized_layers` keys;
    # the inner NemotronHForCausalLM mapper is not consulted there.
    hf_to_sglang_mapper = WeightsMapper(  # HF 到 SGLang 权重映射 # HuggingFace 到 SGLang 权重名称映射
        orig_to_new_prefix={
            "language_model.backbone.": "language_model.model.",  # 前缀映射 # 前缀映射
        },
    )

    @staticmethod
    def create_evs_config(config: NemotronH_Nano_VL_V2_Config):  # 创建 EVS 配置 # 创建 EVS 配置
        """根据模型配置创建 EVS 配置"""
        return EVSConfig(video_pruning_rate=config.video_pruning_rate)  # 返回视频剪枝率配置 # 返回包含视频剪枝率的配置

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: NemotronH_Nano_VL_V2_Config,  # 模型配置 # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__(config)  # 调用父类初始化 # 调用父类初始化

        self.downsample_ratio = config.downsample_ratio  # 下采样比率 # 下采样比率
        self.language_model = NemotronHForCausalLM(  # 语言模型 # 创建 Nemotron-H 语言模型
            config=config.llm_config,  # LLM 配置 # 语言模型配置
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("language_model", prefix),  # 前缀 # 参数名前缀
        )
        self.vision_model = RadioModel(config=config.create_radio_config()).to(  # 视觉模型 # 创建 RADIO 视觉模型
            self.language_model.config.dtype  # 转换数据类型 # 转换为语言模型的数据类型
        )

        vit_hidden_size = config.vit_hidden_size  # ViT 隐藏层大小 # 视觉编码器隐藏层维度
        self.rmsnorm_hidden_size = (  # RMSNorm 隐藏层大小 # RMSNorm 输入维度
            vit_hidden_size * int(round(1 / self.downsample_ratio)) ** 2  # 考虑下采样的维度 # 考虑下采样后的维度
        )
        vision_projection_hidden_size = config.projector_hidden_size  # 投影器隐藏层大小 # 投影器中间维度
        llm_hidden_size = config.llm_config.hidden_size  # LLM 隐藏层大小 # 语言模型隐藏层维度
        self.llm_hidden_size = llm_hidden_size  # 保存 LLM 隐藏层大小 # 保存语言模型隐藏层维度
        self.model_dtype = self.language_model.config.torch_dtype  # 模型数据类型 # 模型的 PyTorch 数据类型

        self.mlp1 = nn.Sequential(  # MLP 投影器 # 视觉特征投影 MLP
            RMSNorm(  # RMSNorm # RMS 归一化层
                hidden_size=self.rmsnorm_hidden_size,  # 输入大小 # 输入维度
                eps=1e-5,  # epsilon # epsilon 值
            ),
            nn.Linear(  # 线性层 1 # 第一层线性变换
                self.rmsnorm_hidden_size,  # 输入大小 # 输入维度
                vision_projection_hidden_size,  # 输出大小 # 输出维度
                bias=False,  # 无偏置 # 无偏置
            ),
            ReLU2(),  # ReLU2 激活 # ReLU2 激活函数
            nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False),  # 线性层 2 # 第二层线性变换
        ).to(self.model_dtype)  # 转换数据类型 # 转换为模型数据类型

        self.sound_encoder: ProjectedParakeet | None = None  # 音频编码器 # 音频编码器（可选）
        if getattr(config, "sound_config", None) is not None:  # 如果有音频配置 # 如果有音频配置
            logger.info(  # 记录日志 # 记录日志
                "Found sound config, initializing sound encoder for Nemotron AVLM"
            )
            self.sound_encoder = ProjectedParakeet(  # 创建音频编码器 # 创建 Parakeet 音频编码器
                config.sound_config,  # 音频配置 # 音频配置
                dtype=self.language_model.config.torch_dtype,  # 数据类型 # 数据类型
                llm_hidden_size=llm_hidden_size,  # LLM 隐藏层大小 # 语言模型隐藏层维度
                max_model_len=getattr(config, "max_model_len", 8192),  # 最大模型长度 # 最大模型长度
            )

        self.config = config  # 保存配置 # 保存模型配置

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):  # 填充输入 ID # 填充输入标记 ID
        """对输入标记 ID 进行多模态填充，处理视觉和音频数据"""
        im_start_id: int = mm_inputs.im_start_id  # 图像起始 ID # 图像起始标记 ID
        im_end_id: int = mm_inputs.im_end_id  # 图像结束 ID # 图像结束标记 ID

        visual_items = [item for item in mm_inputs.mm_items if not item.is_audio()]  # 视觉数据项 # 筛选视觉数据项
        audio_items = [item for item in mm_inputs.mm_items if item.is_audio()]  # 音频数据项 # 筛选音频数据项

        all_data_offsets = []  # 所有数据偏移 # 所有数据偏移量

        if visual_items:  # 如果有视觉数据 # 如果有视觉数据
            mm_inputs.mm_items = visual_items  # 更新多模态项 # 更新多模态项为视觉项
            helper = MultiModalityDataPaddingPatternTokenPairs(  # 创建填充助手 # 创建标记对填充模式
                [(im_start_id, im_end_id)]  # 图像标记对 # 图像起始和结束标记对
            )
            input_ids = helper.pad_input_tokens(input_ids, mm_inputs)  # 填充输入标记 # 填充输入标记
            all_data_offsets.extend(mm_inputs.data_offsets)  # 扩展偏移 # 扩展数据偏移

        audio_start_id = getattr(mm_inputs, "audio_start_id", None)  # 音频起始 ID # 音频起始标记 ID
        audio_end_id = getattr(mm_inputs, "audio_end_id", None)  # 音频结束 ID # 音频结束标记 ID
        if audio_items and audio_start_id is not None and audio_end_id is not None:  # 如果有音频数据和标记 # 如果有音频数据和标记
            mm_inputs.mm_items = audio_items  # 更新多模态项 # 更新多模态项为音频项
            helper = MultiModalityDataPaddingPatternTokenPairs(  # 创建填充助手 # 创建标记对填充模式
                [(audio_start_id, audio_end_id)]  # 音频标记对 # 音频起始和结束标记对
            )
            input_ids = helper.pad_input_tokens(input_ids, mm_inputs)  # 填充输入标记 # 填充输入标记
            all_data_offsets.extend(mm_inputs.data_offsets)  # 扩展偏移 # 扩展数据偏移

        mm_inputs.mm_items = visual_items + audio_items  # 合并视觉和音频项 # 合并视觉和音频数据项
        mm_inputs.data_offsets = all_data_offsets  # 更新数据偏移 # 更新数据偏移

        if audio_items:  # 如果有音频数据 # 如果有音频数据
            for item in visual_items:  # 遍历视觉项 # 遍历视觉数据项
                if isinstance(item, VideoEVSDataItem):  # 如果是视频 EVS 项 # 如果是视频 EVS 数据项
                    item.pre_chunked_input_ids = input_ids  # 保存预分块的输入 ID # 保存预分块的输入标记

        return input_ids  # 返回填充后的 ID # 返回填充后的输入标记 ID

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:  # 像素混洗 # 像素混洗操作，降低空间分辨率增加通道数
        """执行像素混洗操作，降低空间分辨率并增加通道维度"""
        n, w, h, c = x.size()  # 获取形状 # 获取批次、宽、高、通道
        # N, W, H, C --> N, W, H * scale, C // scale # 维度变换
        x = x.view(  # 重塑 # 重塑张量
            n,  # 批次 # 批次大小
            w,  # 宽度 # 宽度
            int(h * scale_factor),  # 缩放后的高度 # 缩放后的高度
            int(c / scale_factor),  # 缩放后的通道 # 缩放后的通道数
        )
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale # 维度置换
        x = x.permute(0, 2, 1, 3).contiguous()  # 置换维度 # 置换高度和宽度维度
        # N, H * scale, W, C // scale -->
        # N, H * scale, W * scale, C // (scale ** 2) # 维度变换
        x = x.view(  # 重塑 # 重塑张量
            n,  # 批次 # 批次大小
            int(h * scale_factor),  # 缩放后的高度 # 缩放后的高度
            int(w * scale_factor),  # 缩放后的宽度 # 缩放后的宽度
            int(c / (scale_factor * scale_factor)),  # 缩放后的通道 # 缩放后的通道数
        )
        if self.config.ps_version != "v1":  # 如果不是 v1 版本 # 如果不是 v1 版本的像素混洗
            x = x.permute(0, 2, 1, 3).contiguous()  # 再次置换 # 再次置换高度和宽度维度
        return x  # 返回结果 # 返回像素混洗结果

    def extract_feature_dynamic(self, pixel_values_list: list[torch.Tensor]):  # 动态特征提取 # 从变尺寸图像中提取特征
        """Extract features from variable-size images (dynamic resolution).

        Each image has different spatial dimensions. They are passed as a list
        to RADIO which handles ragged packing with cu_seqlens internally.
        """
        """从不同尺寸的图像中提取特征（动态分辨率）"""
        features, num_patches_list = self.vision_model(pixel_values_list)  # 视觉模型前向 # 通过视觉模型提取特征
        patch_size = self.config.patch_size  # 补丁大小 # 补丁大小
        results = []  # 结果列表 # 存储结果
        offset = 0  # 偏移量 # 偏移量
        for i, num_patches in enumerate(num_patches_list):  # 遍历每张图像 # 遍历每张图像
            img_feats = features[0, offset : offset + num_patches]  # 获取当前图像特征 # 获取当前图像的特征
            h_patches = pixel_values_list[i].shape[-2] // patch_size  # 高度补丁数 # 高度方向的补丁数
            w_patches = pixel_values_list[i].shape[-1] // patch_size  # 宽度补丁数 # 宽度方向的补丁数
            img_feats = img_feats.reshape(1, h_patches, w_patches, -1)  # 重塑为空间格式 # 重塑为空间格式
            img_feats = self.pixel_shuffle(img_feats, self.downsample_ratio)  # 像素混洗 # 应用像素混洗
            img_feats = img_feats.view(-1, self.rmsnorm_hidden_size)  # 展平 # 展平为 2D
            img_feats = self.mlp1(img_feats)  # MLP 投影 # 通过 MLP 投影
            results.append(img_feats)  # 添加到结果 # 添加到结果列表
            offset += num_patches  # 更新偏移 # 更新偏移量
        return torch.cat(results, dim=0)  # 拼接结果 # 拼接所有图像的特征

    def extract_video_feature_temporal(self, pixel_values, num_frames):  # 视频时序特征提取 # 提取带时序压缩的视频特征
        """Extract video features with temporal compression (tubelet grouping)."""
        """提取带时序压缩的视频特征（管状分组）"""
        vit_embeds = self.vision_model(pixel_values, num_frames=num_frames)  # 视觉模型前向 # 通过视觉模型提取特征
        num_tubelets = vit_embeds.shape[0]  # 管状块数量 # 管状块数量
        patch_size = self.config.patch_size  # 补丁大小 # 补丁大小
        h_patches = pixel_values.shape[-2] // patch_size  # 高度补丁数 # 高度方向的补丁数
        w_patches = pixel_values.shape[-1] // patch_size  # 宽度补丁数 # 宽度方向的补丁数
        vit_embeds = vit_embeds.reshape(num_tubelets, h_patches, w_patches, -1)  # 重塑为空间格式 # 重塑为空间格式
        vit_embeds = self.pixel_shuffle(vit_embeds, self.downsample_ratio)  # 像素混洗 # 应用像素混洗
        vit_embeds = vit_embeds.view(-1, self.rmsnorm_hidden_size)  # 展平 # 展平为 2D
        vit_embeds = self.mlp1(vit_embeds)  # MLP 投影 # 通过 MLP 投影
        vit_embeds = vit_embeds.view(num_tubelets, -1, self.llm_hidden_size)  # 重塑为 3D # 重塑为 3D
        return vit_embeds  # 返回视频特征 # 返回视频特征

    def get_input_embeddings(self):  # 获取输入嵌入 # 获取语言模型的输入嵌入层
        """获取语言模型的输入嵌入层"""
        return self.language_model.get_input_embeddings()  # 委托给语言模型 # 委托给语言模型

    def extract_feature(self, pixel_values):  # 特征提取 # 从固定尺寸图像中提取特征
        """从固定尺寸的图像中提取特征，使用微批次处理"""
        micro_batch_size = 128  # 微批次大小 # 微批次大小
        n = pixel_values.shape[0]  # 图像数量 # 图像数量
        patch_size = self.config.patch_size  # 补丁大小 # 补丁大小
        h_patches = pixel_values.shape[-2] // patch_size  # 高度补丁数 # 高度方向的补丁数
        w_patches = pixel_values.shape[-1] // patch_size  # 宽度补丁数 # 宽度方向的补丁数
        vit_embeds_list = []  # 嵌入列表 # 存储嵌入
        for i in range(0, n, micro_batch_size):  # 遍历微批次 # 遍历微批次
            chunk = pixel_values[i : i + micro_batch_size]  # 获取微批次 # 获取当前微批次
            batch_size = chunk.shape[0]  # 当前批次大小 # 当前批次大小
            vit_embeds = self.vision_model(chunk)  # 视觉模型前向 # 通过视觉模型提取特征
            vit_embeds = vit_embeds.to(dtype=self.model_dtype)  # 转换数据类型 # 转换数据类型
            vit_embeds = vit_embeds.reshape(batch_size, h_patches, w_patches, -1)  # 重塑为空间格式 # 重塑为空间格式
            vit_embeds = self.pixel_shuffle(  # 像素混洗 # 应用像素混洗
                vit_embeds, scale_factor=self.downsample_ratio  # 缩放因子 # 缩放因子
            )
            vit_embeds = vit_embeds.view(-1, self.rmsnorm_hidden_size)  # 展平 # 展平为 2D
            vit_embeds = self.mlp1(vit_embeds)  # MLP 投影 # 通过 MLP 投影
            vit_embeds = vit_embeds.view(batch_size, -1, self.llm_hidden_size)  # 重塑为 3D # 重塑为 3D
            vit_embeds_list.append(vit_embeds)  # 添加到列表 # 添加到列表
        vit_embeds = torch.cat(vit_embeds_list, dim=0)  # 拼接所有微批次 # 拼接所有微批次
        return vit_embeds  # 返回特征 # 返回特征

    def get_image_feature(self, items: list[MultimodalDataItem]):  # 获取图像特征 # 提取并投影图像特征
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        """将视觉模型的最后隐藏状态投影到语言模型空间"""
        is_dynamic = any(getattr(item, "is_dynamic", False) for item in items)  # 检查是否有动态分辨率 # 检查是否有动态分辨率
        if is_dynamic:  # 如果是动态分辨率 # 如果是动态分辨率
            pixel_values_list = [item.feature for item in items]  # 获取像素值列表 # 获取像素值列表
            return self.extract_feature_dynamic(pixel_values_list)  # 动态特征提取 # 调用动态特征提取

        pixel_values = torch.cat([item.feature for item in items])  # 拼接像素值 # 拼接像素值
        image_features = self.extract_feature(pixel_values)  # 特征提取 # 调用特征提取
        return image_features  # 返回图像特征 # 返回图像特征

    def get_video_feature(self, items: list[MultimodalDataItem]):  # 获取视频特征 # 提取并投影视频特征
        """
        Projects the last hidden state from the video model into language model space.

        Returns:
            video_features (`torch.Tensor`): Video feature tensor of shape `(num_videos, video_length, embed_dim)`).
        """
        """将视频模型的最后隐藏状态投影到语言模型空间"""
        pixel_values = torch.cat([item.feature for item in items])  # 拼接像素值 # 拼接像素值
        if getattr(self.config, "video_temporal_patch_size", 1) > 1:  # 如果有时序补丁 # 如果有时序补丁大小大于 1
            num_frames = pixel_values.shape[0]  # 帧数 # 帧数
            return self.extract_video_feature_temporal(pixel_values, num_frames)  # 时序特征提取 # 调用时序特征提取
        video_features = self.extract_feature(pixel_values)  # 特征提取 # 调用特征提取
        return video_features  # 返回视频特征 # 返回视频特征

    def get_audio_feature(self, items: list[MultimodalDataItem]):  # 获取音频特征 # 提取并投影音频特征
        """
        Encode audio features through the Parakeet sound encoder.

        Each item carries mel spectrogram features, an attention mask, and a
        clip count. Multiple clips per audio item are grouped and concatenated
        (trimmed to valid output lengths) to form a single embedding per item.
        """
        """通过 Parakeet 音频编码器编码音频特征"""
        assert self.sound_encoder is not None  # 断言音频编码器已初始化 # 断言音频编码器已初始化

        all_features = []  # 所有特征 # 存储所有特征
        all_masks = []  # 所有掩码 # 存储所有掩码
        all_num_clips = []  # 所有片段数 # 存储所有片段数
        for item in items:  # 遍历数据项 # 遍历音频数据项
            all_features.append(item.feature)  # 添加特征 # 添加特征
            all_masks.append(item.feature_attention_mask)  # 添加掩码 # 添加注意力掩码
            all_num_clips.append(item.audio_num_clips)  # 添加片段数 # 添加片段数

        input_audio_features = torch.cat(all_features, dim=0)  # 拼接特征 # 拼接所有音频特征
        feature_attention_mask = torch.cat(all_masks, dim=0)  # 拼接掩码 # 拼接所有掩码

        target_device = next(self.sound_encoder.parameters()).device  # 目标设备 # 获取音频编码器的设备
        input_audio_features = input_audio_features.to(  # 转换设备和类型 # 转换到正确的设备和数据类型
            dtype=self.language_model.config.torch_dtype, device=target_device  # 数据类型和设备 # 数据类型和设备
        )
        feature_attention_mask = feature_attention_mask.to(device=target_device)  # 转换设备 # 转换掩码到正确设备

        sound_embeds = self.sound_encoder(input_audio_features, feature_attention_mask)  # 音频编码器前向 # 通过音频编码器提取特征

        valid_input_lens = feature_attention_mask.sum(dim=1)  # 有效输入长度 # 计算每个样本的有效输入长度
        valid_output_lens = (  # 有效输出长度 # 计算每个样本的有效输出长度
            self.sound_encoder.encoder._get_subsampling_output_length(valid_input_lens)  # 子采样输出长度 # 通过子采样计算输出长度
            .long()  # 转为长整型 # 转为长整型
            .tolist()  # 转为列表 # 转为列表
        )

        grouped_embeds = []  # 分组嵌入 # 存储分组的嵌入
        clip_offset = 0  # 片段偏移 # 片段偏移量
        for num_clips in all_num_clips:  # 遍历片段数 # 遍历每个音频项的片段数
            embeds = []  # 嵌入列表 # 存储当前音频项的嵌入
            for clip_idx in range(clip_offset, clip_offset + num_clips):  # 遍历片段 # 遍历当前音频项的所有片段
                valid_len = valid_output_lens[clip_idx]  # 有效长度 # 获取有效输出长度
                embeds.append(sound_embeds[clip_idx, :valid_len])  # 添加有效嵌入 # 添加有效的嵌入
            grouped_embeds.append(torch.cat(embeds, dim=0))  # 拼接片段嵌入 # 拼接片段嵌入
            clip_offset += num_clips  # 更新偏移 # 更新偏移量

        return torch.cat(grouped_embeds, dim=0)  # 返回所有音频嵌入 # 拼接所有音频嵌入

    @torch.no_grad()  # 禁用梯度 # 禁用梯度计算
    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入 # 是否获取嵌入
    ):
        data_embedding_funcs = {  # 数据嵌入函数映射 # 数据嵌入函数映射
            Modality.IMAGE: self.get_image_feature,  # 图像特征 # 图像模态使用图像特征提取
            Modality.VIDEO: self.get_video_feature,  # 视频特征 # 视频模态使用视频特征提取
        }
        if self.sound_encoder is not None:  # 如果有音频编码器 # 如果有音频编码器
            data_embedding_funcs[Modality.AUDIO] = self.get_audio_feature  # 音频特征 # 音频模态使用音频特征提取

        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入流程 # 调用通用多模态嵌入流程
            input_ids=input_ids,  # 输入 ID # 输入标记 ID
            forward_batch=forward_batch,  # 前向批次 # 前向批次信息
            language_model=self.language_model,  # 语言模型 # 语言模型
            multimodal_model=self,  # 多模态模型 # 多模态模型自身
            data_embedding_funcs=data_embedding_funcs,  # 数据嵌入函数 # 数据嵌入函数映射
            positions=positions,  # 位置编码 # 位置编码
        )
        return hidden_states  # 返回隐藏状态 # 返回隐藏状态

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):  # 加载权重 # 加载模型权重
        """加载模型权重，分别处理 LLM、视觉、适配器和音频权重"""
        adapter_dict = dict(self.mlp1.named_parameters())  # 适配器参数字典 # 获取 MLP1 的参数字典

        def is_llm(name: str) -> bool:  # 判断是否为 LLM 权重 # 判断是否为语言模型权重
            return name.startswith("language_model")  # 以 language_model 开头 # 以 language_model 开头

        def is_adapter_weights(weight: tuple[str, torch.Tensor]):  # 判断是否为适配器权重 # 判断是否为适配器权重
            return weight[0].startswith("mlp1")  # 以 mlp1 开头 # 以 mlp1 开头

        def is_vision_weights(name: str) -> bool:  # 判断是否为视觉权重 # 判断是否为视觉模型权重
            return name.startswith("vision_model.radio_model.")  # 以 vision_model.radio_model. 开头 # 以 vision_model.radio_model. 开头

        def is_sound_weights(name: str) -> bool:  # 判断是否为音频权重 # 判断是否为音频权重
            return name.startswith("sound")  # 以 sound 开头 # 以 sound 开头

        # Separate weights by component # 按组件分离权重
        llm_weights = []  # LLM 权重 # 存储语言模型权重
        vision_weights = []  # 视觉权重 # 存储视觉模型权重
        sound_weights = []  # 音频权重 # 存储音频权重

        for name, w in weights:  # 遍历权重 # 遍历所有权重
            if is_llm(name):  # 如果是 LLM 权重 # 如果是语言模型权重
                # Strip 'language_model.' prefix for LLM weights # 去掉 language_model. 前缀
                llm_weights.append((".".join(name.split(".")[1:]), w))  # 添加到 LLM 权重 # 添加到 LLM 权重列表
            elif is_adapter_weights((name, w)):  # 如果是适配器权重 # 如果是适配器权重
                # Load vision-language adapter weights directly # 直接加载视觉-语言适配器权重
                trimmed_name = ".".join(name.split(".")[1:])  # 去掉 mlp1 前缀 # 去掉 mlp1. 前缀
                param = adapter_dict[trimmed_name]  # 获取参数 # 获取对应参数
                with torch.no_grad():  # 禁用梯度 # 禁用梯度
                    default_weight_loader(param, w)  # 加载权重 # 加载权重
            elif is_vision_weights(name):  # 如果是视觉权重 # 如果是视觉模型权重
                # Convert: vision_model.radio_model.* → radio_model.* # 转换名称前缀
                hf_key = name[len("vision_model.") :]  # 去掉 vision_model. 前缀 # 去掉 vision_model. 前缀
                vision_weights.append((hf_key, w))  # 添加到视觉权重 # 添加到视觉权重列表
            elif is_sound_weights(name):  # 如果是音频权重 # 如果是音频权重
                sound_weights.append((name, w))  # 添加到音频权重 # 添加到音频权重列表

        self.language_model.load_weights(llm_weights)  # 加载 LLM 权重 # 加载语言模型权重
        self.vision_model.load_weights(vision_weights)  # 加载视觉权重 # 加载视觉模型权重
        if self.sound_encoder is not None and len(sound_weights) > 0:  # 如果有音频编码器和权重 # 如果有音频编码器和权重
            self.sound_encoder.load_weights(sound_weights)  # 加载音频权重 # 加载音频权重


class NemotronH_Nano_Omni_Reasoning_V3(NemotronH_Nano_VL_V2):  # Nano Nemotron 全能推理 V3 # 继承自 V2
    """NemotronH Nano Omni Reasoning V3 模型，继承自 V2"""
    pass  # 无额外实现 # 无额外实现


EntryClass = [NemotronH_Nano_VL_V2, NemotronH_Nano_Omni_Reasoning_V3]  # 入口类 # 模型入口类列表
