# LLaVA多模态视觉语言模型推理实现文件
# 本文件实现了LLaVA模型的推理专用版本，兼容HuggingFace权重
# 支持LLaMA、Qwen2、Mistral等多种语言模型后端
# 支持CLIP和Siglip视觉编码器，以及多种图像处理方式（anyres、spatial_unpad等）
# 包含LLaVA-Llama、LLaVA-Qwen、LLaVA-Mistral和LLaVA-ForConditionalGeneration等变体

# Copyright 2023-2024 SGLang Team
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
"""Inference-only LLaVa model compatible with HuggingFace weights."""

from __future__ import annotations  # 启用延迟类型注解评估

import math  # 数学函数
import re  # 正则表达式
from array import array  # 数组类型
from functools import lru_cache  # LRU缓存装饰器
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union  # 类型提示

import numpy as np  # NumPy库
import torch  # PyTorch核心库
from torch import nn  # 神经网络模块
from transformers import (  # Transformers库
    CLIPVisionConfig,  # CLIP视觉配置
    CLIPVisionModel,  # CLIP视觉模型
    LlavaConfig,  # LLaVA配置
    MistralConfig,  # Mistral配置
    Qwen2Config,  # Qwen2配置
    SiglipVisionModel,  # Siglip视觉模型
)
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM  # 自动模型
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector  # LLaVA多模态投影器

# leave till last and symbol only in case circular import
import sglang.srt.models as sgl_models  # SGLang模型注册表（延迟导入避免循环依赖）
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.managers.mm_utils import general_mm_embed_routine  # 通用多模态嵌入例程
from sglang.srt.managers.schedule_batch import (  # 调度批次相关
    Modality,  # 模态类型
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.models.llama import LlamaForCausalLM  # LLaMA因果语言模型
from sglang.srt.models.mistral import MistralForCausalLM  # Mistral因果语言模型
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # Qwen2因果语言模型
from sglang.srt.multimodal.mm_utils import (  # 多模态工具函数
    get_anyres_image_grid_shape,  # 获取anyres图像网格形状
    unpad_image,  # 去除图像填充
    unpad_image_shape,  # 获取去填充图像形状
)
from sglang.srt.utils import add_prefix, flatten_nested_list, logger  # 工具函数

_KNOWN_BROKEN_AUTOMODEL_CONFIG = "VoxtralRealtimeTextConfig"  # 已知损坏的AutoModel配置
_KNOWN_BROKEN_AUTOMODEL_ERROR = "Could not find VoxtralRealtimeTextModel"  # 已知损坏的AutoModel错误


class LlavaBaseForCausalLM(nn.Module):
    """LLaVA基础因果语言模型，提供图像编码、输入填充和权重加载的通用实现"""

    @staticmethod
    def _infer_image_aspect_ratio(mm_items):
        """Determine image_aspect_ratio from processor metadata or item count."""
        """从处理器元数据或项目数量推断图像宽高比"""
        # Check if processor stored the aspect_ratio it used
        for item in mm_items:  # 检查处理器是否存储了宽高比
            ar = item.model_specific_data.get("image_aspect_ratio")
            if ar is not None:  # 找到则直接返回
                return ar
        # Fallback: multi-image or video → pad, single image → anyres
        image_items = [item for item in mm_items if item.is_image()]  # 筛选图像项
        has_video = any(item.is_video() for item in mm_items)  # 检查是否有视频
        if len(image_items) > 1 or has_video:  # 多图像或视频使用pad模式
            return "pad"
        return "anyres"  # 单图像使用anyres模式

    def pad_input_ids(
        self, input_ids: array[int], image_inputs: MultimodalInputs  # 输入ID数组，多模态输入
    ) -> array[int]:
        """将图像占位符替换为实际图像特征对应的填充token"""
        image_sizes = flatten_nested_list(  # 获取所有图像尺寸
            [item.image_sizes for item in image_inputs.mm_items]
        )

        pad_values = [item.pad_value for item in image_inputs.mm_items]  # 获取填充值

        # hardcode for spatial_unpad + anyres
        # Use per-item aspect_ratio from processor if available, else infer
        image_aspect_ratio = self._infer_image_aspect_ratio(image_inputs.mm_items)  # 推断宽高比
        offset_list = []  # 偏移量列表
        image_inputs.image_pad_len = []  # 图像填充长度列表
        for image_idx, image_s in enumerate(image_sizes):  # 遍历每张图像
            if len(image_sizes) > 16:  # 大量图像时使用2x2池化
                # 2x2 pooling with stride 2
                new_image_feature_len = (
                    math.ceil(self.image_size / self.patch_size / 2) ** 2
                )
            else:
                new_image_feature_len = self.image_feature_len  # multi-image

            height = width = self.num_patches_per_side  # patch数
            if "anyres" in image_aspect_ratio:  # anyres模式
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(  # 获取网格形状
                    image_s,
                    self.image_grid_pinpoints,
                    self.vision_tower.config.image_size,
                )
                h = num_patch_height * height  # 总高度
                w = num_patch_width * width  # 总宽度
                new_h, new_w = unpad_image_shape(h, w, image_s)  # 去填充形状

                if "anyres_max" in self.config.image_aspect_ratio:  # anyres_max模式
                    matched_anyres_max_num_patches = re.match(
                        r"anyres_max_(\d+)", self.config.image_aspect_ratio
                    )
                    if matched_anyres_max_num_patches:  # 提取最大patch数
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))
                    # times = math.sqrt(h * w / (max_num_patches * unit**2))
                    times = math.sqrt(  # 计算缩放倍数
                        new_h * new_w / (max_num_patches * self.image_feature_len)
                    )
                    if times > 1.1:  # 缩放超过1.1时调整尺寸
                        new_h = int(new_h // times)
                        new_w = int(new_w // times)
                new_image_feature_len += new_h * (new_w + 1)  # 加上去填充后的特征数

            try:
                offset = input_ids.index(self.config.image_token_index)  # 查找图像token位置
            except ValueError:  # 未找到图像token
                offset = 0
            # old_len + pad_len - 1, because we need to remove image_token_id
            pad_token = pad_values[image_idx % len(pad_values)]  # 获取填充token
            input_ids = (  # 替换图像token为填充token
                input_ids[:offset]
                + array("q", [pad_token]) * new_image_feature_len
                + input_ids[offset + 1 :]
            )
            offset_list.append(offset)  # 记录偏移量
            image_inputs.image_pad_len.append(new_image_feature_len)  # 记录填充长度

        image_inputs.image_offsets = offset_list  # 保存偏移量列表
        return input_ids  # 返回填充后的输入ID

    def encode_images(
        self, pixel_values: Union[torch.Tensor, List[torch.Tensor]]  # 像素值
    ) -> torch.Tensor:
        """
        encode images by vision tower and multimodal projector
        通过视觉塔和多模态投影器编码图像
        Args:
            pixel_values: torch.Tensor or List[torch.Tensor]: each tensor for an input image
            每个张量对应一个输入图像
        Returns:
            torch.Tensor: encoded image features from the input image; if multiple, flattened by seq_len axis
            编码后的图像特征；多个图像沿seq_len轴展平
        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)  # 视觉编码
        # NOTE: This is not memory efficient. (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]  # 选择特征层
        if self.vision_feature_select_strategy in ["default", "patch"]:  # patch策略：去掉CLS token
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":  # full策略：保留所有token
            selected_image_feature = selected_image_feature
        else:  # 不支持的策略
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )
        image_features = self.multi_modal_projector(selected_image_feature)  # 多模态投影
        return image_features  # 返回图像特征

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """LLaVA前向传播：处理图像特征并嵌入到语言模型中"""
        image_inputs = forward_batch.mm_inputs  # 获取多模态输入

        if forward_batch.forward_mode.is_extend():  # 扩展（预填充）模式
            # Clamp input ids. This is because the input_ids for the image tokens are
            # filled with the hash values of the image for the prefix matching in the radix attention.
            # There values are useless because their embeddings will be replaced by vision embeddings anyway.
            input_ids.clamp_(min=0, max=self.config.vocab_size - 1)  # 限制输入ID范围

            # Embed text inputs
            input_embeds = self.language_model.model.embed_tokens(input_ids)  # 文本嵌入

            # Compute max image offset per request to determine need_vision
            max_image_offset = []  # 每个请求的最大图像偏移
            for im in image_inputs:
                if im and im.image_offsets:  # 有图像偏移时
                    max_image_offset.append(
                        np.max(np.array(im.image_offsets) + np.array(im.image_pad_len))
                    )
                else:  # 无图像时
                    max_image_offset.append(-1)

            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()  # 起始位置
            need_vision = start_positions <= np.array(max_image_offset)  # 需要视觉特征的请求

            if need_vision.any():  # 有需要视觉的请求
                bs = forward_batch.batch_size  # 批次大小

                # Build per-image lists filtered by need_vision
                modalities_list = []  # 模态列表
                aspect_ratios = []  # per-image aspect ratio
                for i in range(bs):  # 遍历每个请求
                    if need_vision[i] and image_inputs[i]:  # 需要视觉且有输入
                        items = image_inputs[i].mm_items
                        ar = self._infer_image_aspect_ratio(items)  # 推断宽高比
                        for item in items:
                            modalities_list.append(item.modality)  # 记录模态
                            aspect_ratios.append(ar)  # 记录宽高比

                pixel_values = flatten_nested_list(  # 获取像素值
                    [
                        [item.feature for item in image_inputs[i].mm_items]
                        for i in range(bs)
                        if need_vision[i]
                    ]
                )
                # Per-image sizes (each entry is [(w,h)] for one image)
                image_sizes = [  # 每张图像的尺寸
                    item.image_sizes
                    for i in range(bs)
                    if need_vision[i]
                    for item in image_inputs[i].mm_items
                ]

                ########## Encode Image ########

                if pixel_values[0].ndim == 4:  # llava-hd格式：4D
                    # llava-hd: BS, num_patch, C=3, H=336, W=336, num_patch obtained from process_images
                    np.concatenate(pixel_values, axis=0)  # 沿batch维度拼接
                    # ndim=4
                    concat_images = torch.tensor(  # 转为torch张量
                        np.concatenate(pixel_values, axis=0),
                        device=self.vision_tower.device,
                    )
                    image_features = self.encode_images(concat_images)  # 编码图像
                    split_sizes = [image.shape[0] for image in pixel_values]  # 分割大小
                    image_features = torch.split(image_features, split_sizes, dim=0)  # 分割特征
                    # hd image_features: BS, num_patch, 576, 4096
                else:  # 普通像素格式：3D
                    # normal pixel: BS, C=3, H=336, W=336
                    pixel_values = torch.tensor(
                        np.array(pixel_values), device=self.vision_tower.device
                    )
                    image_features = self.encode_images(pixel_values)  # 编码图像
                    # image_features: BS, 576, 4096

                if self.mm_patch_merge_type.startswith("spatial"):  # 空间patch合并
                    new_image_features = []  # 新的图像特征列表
                    height = width = self.num_patches_per_side  # 每侧patch数
                    for image_idx, image_feature in enumerate(image_features):  # 遍历每张图像
                        image_aspect_ratio = aspect_ratios[image_idx]  # 当前图像宽高比
                        if (  # anyres多patch图像
                            image_feature.shape[0] > 1
                            and "anyres" in image_aspect_ratio
                            and modalities_list[image_idx] == Modality.IMAGE
                        ):
                            base_image_feature = image_feature[0]  # 基础特征
                            image_feature = image_feature[1:]  # 额外patch特征
                            assert height * width == base_image_feature.shape[0]  # 验证patch数

                            if "anyres_max" in image_aspect_ratio:  # anyres_max模式
                                matched_anyres_max_num_patches = re.match(
                                    r"anyres_max_(\d+)", image_aspect_ratio
                                )
                                if matched_anyres_max_num_patches:  # 提取最大patch数
                                    max_num_patches = int(
                                        matched_anyres_max_num_patches.group(1)
                                    )

                            if (  # anyres或anyres_max模式
                                image_aspect_ratio == "anyres"
                                or "anyres_max" in image_aspect_ratio
                            ):
                                vision_tower_image_size = self.image_size  # 视觉塔图像大小
                                try:
                                    num_patch_width, num_patch_height = (  # 获取网格形状
                                        get_anyres_image_grid_shape(
                                            image_sizes[image_idx][0],
                                            self.config.image_grid_pinpoints,
                                            vision_tower_image_size,
                                        )
                                    )
                                except Exception as e:  # 异常时使用默认值
                                    print(f"Error: {e}")
                                    num_patch_width, num_patch_height = 2, 2
                                image_feature = image_feature.view(  # 重塑特征形状
                                    num_patch_height, num_patch_width, height, width, -1
                                )
                            else:  # 非anyres模式
                                image_feature = image_feature.view(
                                    2, 2, height, width, -1
                                )

                            # (
                            #     num_patch_width,
                            #     num_patch_height,
                            # ) = get_anyres_image_grid_shape(
                            #     image_sizes[image_idx][0],
                            #     self.image_grid_pinpoints,
                            #     self.vision_tower.config.image_size,
                            # )

                            # image_feature = image_feature.view(
                            #     num_patch_height, num_patch_width, height, width, -1
                            # )

                            if "unpad" in self.mm_patch_merge_type:  # unpad模式
                                unit = image_feature.shape[2]  # 单元大小
                                image_feature = image_feature.permute(  # 排列变换
                                    4, 0, 2, 1, 3
                                ).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(  # 展平
                                    2, 3
                                )
                                image_feature = unpad_image(  # 去填充
                                    image_feature, image_sizes[image_idx][0]
                                )
                                if (  # anyres_max模式下的缩放
                                    "anyres_max" in image_aspect_ratio
                                    and matched_anyres_max_num_patches
                                ):
                                    c, h, w = image_feature.shape
                                    times = math.sqrt(  # 计算缩放倍数
                                        h * w / (max_num_patches * unit**2)
                                    )
                                    if times > 1.1:  # 缩放
                                        image_feature = image_feature[None]  # 增加批次维度
                                        image_feature = nn.functional.interpolate(
                                            image_feature,
                                            [int(h // times), int(w // times)],
                                            mode="bilinear",  # 双线性插值
                                        )[0]
                                image_feature = torch.cat(  # 拼接换行符
                                    (
                                        image_feature,
                                        self.language_model.model.image_newline[
                                            :, None, None
                                        ].expand(*image_feature.shape[:-1], 1),  # 扩展换行符
                                    ),
                                    dim=-1,
                                )
                                image_feature = image_feature.flatten(1, 2).transpose(  # 展平并转置
                                    0, 1
                                )
                            else:  # 非unpad模式
                                image_feature = image_feature.permute(  # 排列变换
                                    0, 2, 1, 3, 4
                                ).contiguous()
                                image_feature = image_feature.flatten(0, 3)  # 展平
                            image_feature = torch.cat(  # 拼接基础特征
                                (base_image_feature, image_feature), dim=0
                            )
                            image_feature = image_feature.unsqueeze(0)  # 增加批次维度
                        else:  # 非anyres模式或视频
                            if modalities_list[image_idx] == Modality.VIDEO:  # video
                                # 2x2 pooling
                                num_of_frames = image_feature.shape[0]  # 帧数
                                image_feature = image_feature.view(  # 重塑形状
                                    num_of_frames, height, width, -1
                                )
                                image_feature = image_feature.permute(  # 排列变换
                                    0, 3, 1, 2
                                ).contiguous()  # N, C, H, W
                                height, weight = image_feature.shape[2:]  # 高宽
                                scaled_shape = [  # 缩放后形状
                                    math.ceil(height / 2),
                                    math.ceil(weight / 2),
                                ]
                                image_feature = nn.functional.interpolate(  # 双线性插值下采样
                                    image_feature, size=scaled_shape, mode="bilinear"
                                )
                                image_feature = (  # 展平空间维度
                                    image_feature.flatten(2)
                                    .transpose(1, 2)
                                    .contiguous()
                                )  # N, C, H*W
                            if "unpad" in self.mm_patch_merge_type:  # unpad模式
                                image_feature = torch.cat(  # 拼接换行符
                                    (
                                        image_feature,
                                        # Expand to (bs, 1, hidden_dim) and concat at the end of the image tokens
                                        self.language_model.model.image_newline[  # 换行符
                                            None, None
                                        ].expand(
                                            image_feature.shape[0],
                                            1,
                                            image_feature.shape[-1],
                                        ),  # 扩展
                                    ),
                                    dim=1,
                                )

                        new_image_features.append(image_feature)  # 添加到列表
                    image_features = new_image_features  # 更新特征列表

                # Fill in the placeholder for the image
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()  # 扩展起始位置
                extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()  # 扩展序列长度
                prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu  # 前缀长度
                # Fill in the image features using flat indexing (one pt per image)
                pt = 0  # 图像特征指针
                for i in range(bs):  # 遍历每个请求
                    if not need_vision[i]:  # 不需要视觉则跳过
                        continue

                    start_idx = extend_start_loc_cpu[i]  # 起始索引
                    seq_len = extend_seq_lens[i]  # 序列长度
                    prefix_len = prefix_lens_cpu[i]  # 前缀长度
                    n_images = len(image_inputs[i].image_offsets)  # 图像数量

                    for j in range(n_images):  # 遍历每张图像
                        image_offset = image_inputs[i].image_offsets[j]  # 图像偏移

                        if (  # 图像完全在前缀中则跳过
                            image_offset + image_inputs[i].image_pad_len[j]
                            <= prefix_len
                        ):
                            pt += 1
                            continue
                        if image_offset >= prefix_len + seq_len:  # 图像完全在序列外
                            pt += n_images - j
                            break

                        tmp_image_feature = image_features[pt]  # 当前图像特征
                        # Squeeze batch dim from per-image features [1, feat, hidden]
                        if tmp_image_feature.ndim == 3:  # 3D特征去掉批次维度
                            tmp_image_feature = tmp_image_feature[0]
                        pad_len = tmp_image_feature.shape[0]  # 填充长度

                        input_offset = image_offset - prefix_len  # 输入偏移
                        left_idx = start_idx + input_offset  # 左索引
                        right_idx = left_idx + pad_len  # 右索引
                        assert right_idx > start_idx  # 确保右索引大于起始
                        if input_offset < 0:  # 图像部分在前缀中
                            left_idx = start_idx
                            tmp_image_feature = tmp_image_feature[-input_offset:]  # 截断前缀部分
                        if right_idx > start_idx + seq_len:  # 图像部分超出序列
                            tmp_image_feature = tmp_image_feature[
                                : start_idx + seq_len - right_idx
                            ]
                            right_idx = start_idx + seq_len
                        try:  # 将图像特征嵌入到输入嵌入中
                            input_embeds[left_idx:right_idx] = tmp_image_feature
                        except RuntimeError as e:  # 运行时错误处理
                            print(f"RuntimeError in image encoding: {e}")
                            print(f"{input_embeds.shape=}, {tmp_image_feature.shape=}")
                            print(
                                f"{start_idx=}, {image_offset=}, {prefix_len=}, {pad_len=}"
                            )
                        pt += 1  # 移动到下一个图像

            return self.language_model(  # 返回语言模型输出
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        elif forward_batch.forward_mode.is_decode():  # 解码模式
            return self.language_model(input_ids, positions, forward_batch)  # 直接调用语言模型

    def get_embed_and_head(self):
        # Spec-decode plumbing: expose the LM's embed/head so the EAGLE draft
        # can share them with the target. self.language_model is a Llama-family
        # CausalLM that defines this method.
        """获取嵌入层和语言模型头（供推测解码EAGLE使用）"""
        return self.language_model.get_embed_and_head()  # 返回嵌入和LM头

    def set_embed_and_head(self, embed, head):
        """设置嵌入层和语言模型头"""
        self.language_model.set_embed_and_head(embed, head)  # 设置嵌入和LM头

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，包括视觉塔、多模态投影器和语言模型"""
        # Load clip vision model by cfg['mm_vision_tower']:
        # huggingface_name or path_of_clip_relative_to_llava_model_dir
        # We put the initialization here instead of __init__ to allow it being reused by other subclasses.
        vision_path = self.config.mm_vision_tower  # 视觉塔路径
        if "clip" in vision_path:  # CLIP视觉模型
            self.vision_tower = CLIPVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
        elif "siglip" in vision_path:  # Siglip视觉模型
            self.vision_tower = SiglipVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
            # Siglip needs all feature tokens
            self.config.mm_vision_select_feature = "full"  # Siglip使用全特征
        self.vision_tower.eval()  # 设置为评估模式

        self.vision_feature_layer = self.config.mm_vision_select_layer  # 视觉特征层
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature  # 特征选择策略
        self.image_size = self.vision_tower.config.image_size  # 图像大小
        self.patch_size = self.vision_tower.config.patch_size  # patch大小

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")  # patch合并类型
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")  # 图像宽高比
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)  # 网格锚点

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)  # 图像特征长度
        if (  # patch或full策略
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass  # 不需要额外调整
        elif self.vision_feature_select_strategy == "cls_patch":  # cls_patch策略
            self.image_feature_len += 1  # 加1（CLS token）
        else:  # 不支持的策略
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # load mm_projector
        projector_weights = {  # 投影器权重名称映射
            "model.mm_projector.0": "multi_modal_projector.linear_1",  # 投影器第一层
            "model.mm_projector.2": "multi_modal_projector.linear_2",  # 投影器第二层
            "model.vision_tower.vision_tower": "vision_tower",  # 视觉塔
            # transformers 5.6.0 flattened CLIPVisionModel/SiglipVisionModel,
            # dropping the `vision_model` intermediate wrapper.
            "vision_tower.vision_model.": "vision_tower.",  # 新版transformers命名
            # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
            "model.image_newline": "language_model.model.image_newline",  # 换行符
        }
        params_dict = dict(self.named_parameters())  # 参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "projector" in name or "vision_tower" in name or "image_newline" in name:  # 投影器/视觉塔/换行符
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:  # 重映射权重名称
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取加载器
                weight_loader(param, loaded_weight)  # 加载权重
            else:  # 语言模型权重
                self.language_model.load_weights([(name, loaded_weight)])  # 加载到语言模型

    @property
    def num_patches_per_side(self):
        """获取每侧patch数"""
        return self.image_size // self.patch_size  # 图像大小除以patch大小


class LlavaLlamaForCausalLM(LlavaBaseForCausalLM):
    """LLaVA + LLaMA语言模型变体"""

    def __init__(
        self,
        config: LlavaConfig,  # LLaVA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.vision_tower = None  # 视觉塔（在load_weights中初始化）
        self.config.vision_config.hidden_size = config.mm_hidden_size  # 视觉隐藏大小
        self.config.text_config.hidden_size = config.hidden_size  # 文本隐藏大小

        self.multi_modal_projector = LlavaMultiModalProjector(config)  # 多模态投影器
        self.language_model = LlamaForCausalLM(  # LLaMA语言模型
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):  # unpad模式需要换行符
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaQwenForCausalLM(LlavaBaseForCausalLM):
    """LLaVA + Qwen2语言模型变体"""

    def __init__(
        self,
        config: LlavaConfig,  # LLaVA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.vision_tower = None  # 视觉塔（在load_weights中初始化）

        if getattr(self.config, "vision_config", None) is None:  # 初始化视觉配置
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:  # 初始化文本配置
            self.config.text_config = Qwen2Config(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size  # 视觉隐藏大小
        self.config.text_config.hidden_size = config.hidden_size  # 文本隐藏大小

        if getattr(self.config, "projector_hidden_act", None) is None:  # 投影器激活函数
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:  # 图像token索引
            self.config.image_token_index = 151646

        self.multi_modal_projector = LlavaMultiModalProjector(config)  # 多模态投影器
        self.language_model = Qwen2ForCausalLM(  # Qwen2语言模型
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):  # unpad模式需要换行符
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaMistralForCausalLM(LlavaBaseForCausalLM):
    """LLaVA + Mistral语言模型变体"""

    def __init__(
        self,
        config: LlavaConfig,  # LLaVA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.vision_tower = None  # 视觉塔（在load_weights中初始化）

        if getattr(self.config, "vision_config", None) is None:  # 初始化视觉配置
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:  # 初始化文本配置
            self.config.text_config = MistralConfig(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size  # 视觉隐藏大小
        self.config.text_config.hidden_size = config.hidden_size  # 文本隐藏大小

        if getattr(self.config, "projector_hidden_act", None) is None:  # 投影器激活函数
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:  # 图像token索引
            self.config.image_token_index = 32000

        self.multi_modal_projector = LlavaMultiModalProjector(config)  # 多模态投影器
        self.language_model = MistralForCausalLM(  # Mistral语言模型
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):  # unpad模式需要换行符
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaForConditionalGeneration(LlavaBaseForCausalLM):
    """
    An adaptor class to enable support for multiple mmlm such as mistral-community/pixtral-12b
    It follows the structure of (vision_tower, multi_modal_projector, language_model)
    适配器类，支持多种多模态语言模型（如pixtral-12b）
    遵循(vision_tower, multi_modal_projector, language_model)结构

    Once a model config is loaded, text_config and vision_config will be extracted, and
    LlavaForConditionalGeneration will load the language_model and vision_tower models
    according to config.
    加载配置后，提取text_config和vision_config，并根据配置加载语言模型和视觉塔
    """

    MULTIMODAL_PROJECTOR_TYPE = LlavaMultiModalProjector  # 多模态投影器类型

    @property
    def dtype(self):
        """获取模型数据类型"""
        return self.torch_dtype

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        """将图像占位符填充到输入ID中，优先使用视觉塔的填充方法"""
        if hasattr(self.vision_tower, "pad_input_ids"):  # 视觉塔有自定义填充方法
            return self.vision_tower.pad_input_ids(input_ids, image_inputs)
        else:  # 使用基类方法
            return super().pad_input_ids(input_ids, image_inputs)

    def _get_sgl_model_cls(self, config, auto_model_type: Type[AutoModel] = AutoModel):
        """
        Get the SGLang model implementation class according to config.
        根据配置获取SGLang模型实现类

        Args:
            config: The config object of the model.  # 模型配置对象
            auto_model_type: The type of the auto model.  # 自动模型类型

        Returns:
            The SGLang model implementation class.  # SGLang模型实现类
        """
        config_cls_name = config.__class__.__name__  # 配置类名
        arch_name_mapping = self._config_cls_name_to_arch_name_mapping(auto_model_type)  # 配置到架构映射
        if arch := arch_name_mapping.get(config_cls_name):  # 查找对应架构
            if isinstance(arch, tuple):  # 多个匹配时取第一个
                arch = arch[0]
                logger.warning(
                    f"Multiple {auto_model_type.__name__} models found for submodule config `{config_cls_name}`, defaulting to [0]: {arch.__name__}"
                )
            try:  # 从模型注册表解析
                return sgl_models.registry.ModelRegistry.resolve_model_cls(arch)[0]
            except Exception as e:  # 解析失败
                raise ValueError(
                    f"{auto_model_type.__name__} found a corresponding model `{arch}` for config class `{config_cls_name}`, but failed to load it from SGLang ModelRegistry. \n{e}"
                )
        else:  # 未找到对应架构
            raise ValueError(
                f"{auto_model_type.__name__} cannot find a corresponding model for config class `{config_cls_name}`"
            )

    @lru_cache
    def _config_cls_name_to_arch_name_mapping(
        self, auto_model_type: Type[AutoModel]
    ) -> Dict[str, str]:
        """构建配置类名到架构名的映射（带LRU缓存）"""
        mapping = {}  # 映射字典
        for config_cls in auto_model_type._model_mapping.keys():  # 遍历所有映射
            try:
                archs = auto_model_type._model_mapping.get(config_cls, None)  # 获取架构
            except ValueError as exc:  # 已知损坏的配置
                if (
                    auto_model_type is not AutoModel
                    or config_cls.__name__ != _KNOWN_BROKEN_AUTOMODEL_CONFIG
                    or _KNOWN_BROKEN_AUTOMODEL_ERROR not in str(exc)
                ):
                    raise  # 非已知错误则重新抛出
                logger.warning(  # 跳过已知损坏的映射
                    "Skipping broken %s mapping for config %s: %s",
                    auto_model_type.__name__,
                    config_cls.__name__,
                    exc,
                )
                continue
            if archs is not None:  # 有架构映射
                if isinstance(archs, tuple):  # 多个架构
                    mapping[config_cls.__name__] = tuple(
                        arch.__name__ for arch in archs
                    )
                else:  # 单个架构
                    mapping[config_cls.__name__] = archs.__name__
        return mapping  # 返回映射

    def __init__(
        self,
        config: LlavaConfig,  # LLaVA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        assert hasattr(config, "text_config")  # 必须有文本配置
        assert hasattr(config, "vision_config")  # 必须有视觉配置
        self.config = config  # 保存配置
        self.text_config = self.config.text_config  # 文本配置
        self.vision_config = self.config.vision_config  # 视觉配置
        self.torch_dtype = getattr(self.config, "torch_dtype")  # 数据类型

        if not getattr(self.text_config, "torch_dtype"):  # 设置文本配置数据类型
            self.text_config.torch_dtype = self.torch_dtype
        if not getattr(self.vision_config, "torch_dtype"):  # 设置视觉配置数据类型
            self.vision_config.torch_dtype = self.torch_dtype

        if not hasattr(self.config, "vocab_size"):  # 设置词表大小
            self.config.vocab_size = self.text_config.vocab_size
        if not hasattr(self.config, "image_aspect_ratio"):  # 默认宽高比
            self.config.image_aspect_ratio = "anyres"
        if not hasattr(self.config, "image_grid_pinpoints"):  # 默认网格锚点
            # from transformers.models.llava_onevision.configuration_llava_onevision import LlavaOnevisionConfig
            # self.config.image_grid_pinpoints = LlavaOnevisionConfig().image_grid_pinpoints
            self.config.image_grid_pinpoints = [
                [96, 96],
                [224, 224],
                [384, 384],
                [512, 512],
                [768, 768],
                [1024, 1024],
            ]
        if not hasattr(self.config, "mm_patch_merge_type"):  # 默认patch合并类型
            self.config.mm_patch_merge_type = "flat"
        if not hasattr(self.config, "image_token_index"):  # 默认图像token索引
            self.config.image_token_index = 10
        if not hasattr(self.config, "projector_hidden_act"):  # 默认投影器激活
            self.config.projector_hidden_act = "gelu"

        self.vision_feature_layer = getattr(self.config, "vision_feature_layer", -1)  # 视觉特征层
        self.vision_feature_select_strategy = getattr(  # 特征选择策略
            self.config, "vision_feature_select_strategy", "full"
        )
        self.image_size = self.vision_config.image_size  # 图像大小
        self.patch_size = self.vision_config.patch_size  # patch大小

        self.mm_patch_merge_type = self.config.mm_patch_merge_type  # patch合并类型
        self.image_aspect_ratio = self.config.image_aspect_ratio  # 图像宽高比
        self.image_grid_pinpoints = self.config.image_grid_pinpoints  # 网格锚点

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)  # 图像特征长度

        self.multi_modal_projector = self.MULTIMODAL_PROJECTOR_TYPE(config)  # 多模态投影器

        language_model_cls = self._get_sgl_model_cls(  # 获取语言模型类
            self.text_config, AutoModelForCausalLM
        )
        vision_model_cls = self._get_sgl_model_cls(self.vision_config, AutoModel)  # 获取视觉模型类
        self.language_model = language_model_cls(  # 初始化语言模型
            self.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.vision_tower = vision_model_cls(  # 初始化视觉塔
            self.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        if "unpad" in getattr(self.config, "mm_patch_merge_type", ""):  # unpad模式需要换行符
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(self.text_config.hidden_size, dtype=self.torch_dtype)
            )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract features from image inputs.
        从图像输入中提取特征

        Args:
            items: List of MultimodalDataItem objects containing image data
                Note that an item can be either "image" or "multi-images"
                多模态数据项列表，每个项可以是单图像或多图像

        Returns:
            torch.Tensor: features from image inputs, concatenated
            拼接后的图像特征
        """
        features = []  # 特征列表
        for item in items:
            # in each item, we assume pixel_values is always batched
            pixel_values, image_sizes = item.feature, item.image_sizes  # 像素值和图像尺寸
            image_outputs = self.vision_tower(  # 视觉编码
                pixel_values, image_sizes, output_hidden_states=True
            )
            selected_image_feature = image_outputs.hidden_states[  # 选择特征层
                self.vision_feature_layer
            ]

            if self.vision_feature_select_strategy in ["default", "patch"]:  # patch策略
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":  # full策略
                selected_image_feature = selected_image_feature
            else:  # 不支持的策略
                raise ValueError(
                    f"Unexpected select feature: {self.vision_feature_select_strategy}"
                )
            features.append(  # 投影后添加到列表
                self.multi_modal_projector(selected_image_feature.squeeze(0))
            )
        ret = torch.cat(features, dim=0)  # 拼接所有特征
        return ret  # 返回特征

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        get_embedding: bool = False,  # 是否获取嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ):
        """LLaVA条件生成模型前向传播：使用通用多模态嵌入例程"""
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入
            input_ids=input_ids,
            forward_batch=forward_batch,
            get_embedding=get_embedding,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,  # 图像特征提取函数
            },
            placeholder_tokens=None,  # using mm_item.pad_value
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for LlavaForConditionalGeneration.
        加载LLaVA条件生成模型的权重

        Unlike the base class implementation, this one doesn't need to handle
        weight name remapping as the weights are already properly structured with
        'language_model' and 'vision_tower' prefixes in the safetensors files.
        与基类不同，此方法不需要处理权重名重映射
        """
        if (  # patch或full策略
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass  # 不需要额外调整
        elif self.vision_feature_select_strategy == "cls_patch":  # cls_patch策略
            self.image_feature_len += 1  # 加1（CLS token）
        else:  # 不支持的策略
            raise ValueError(
                f"Unexpected select feature: {self.vision_feature_select_strategy}"
            )

        # Create dictionaries for direct parameter loading
        params_dict = dict(self.named_parameters())  # 参数字典

        # Load weights directly without remapping
        for name, loaded_weight in weights:  # 遍历所有权重
            for part in ("language_model", "vision_tower"):  # 分发到对应子模型
                if name.startswith(part):  # 属于子模型的权重
                    name = name[len(part + ".") :]  # 去掉前缀
                    getattr(self, part).load_weights([(name, loaded_weight)])  # 加载到子模型
                    break
            else:  # 非子模型权重（如投影器）
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)  # 直接加载


EntryClass = [  # 入口类列表
    LlavaLlamaForCausalLM,
    LlavaQwenForCausalLM,
    LlavaMistralForCausalLM,
    LlavaForConditionalGeneration,
]
