# LLaVa视频模型的SGLang推理实现，兼容HuggingFace权重
# 支持视频帧编码、空间池化下采样和多模态投影
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License"); # 根据Apache 2.0许可证授权
# you may not use this file except in compliance with the License. # 您不得在未遵守许可证的情况下使用此文件
# You may obtain a copy of the License at # 您可以在以下地址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 依许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and # 请参阅许可证以获取管理权限和
# limitations under the License. # 限制的具体语言
# ==============================================================================
"""Inference-only LLaVa video model compatible with HuggingFace weights.""" # 仅推理的LLaVa视频模型，兼容HuggingFace权重

from __future__ import annotations # 启用延迟注解求值

from array import array # 导入数组类型
from typing import Iterable, Optional, Tuple # 导入类型提示

import numpy as np # 导入NumPy数值计算库
import torch # 导入PyTorch深度学习框架
from torch import nn # 导入神经网络模块
from transformers import CLIPVisionModel, LlavaConfig # 导入CLIP视觉模型和LLaVA配置
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector # 导入LLaVA多模态投影器

from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.managers.schedule_batch import MultimodalInputs, flatten_nested_list # 导入多模态输入和嵌套列表展平工具
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.llama import LlamaForCausalLM # 导入Llama因果语言模型
from sglang.srt.utils import add_prefix # 导入前缀添加工具


class LlavaVidForCausalLM(nn.Module): # LLaVa视频因果语言模型类
    def __init__( # 初始化方法
        self,
        config: LlavaConfig, # LLaVA配置对象
        quant_config: Optional[QuantizationConfig] = None, # 可选的量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.vision_tower = None # 视觉塔，稍后在load_weights中初始化
        self.config.vision_config.hidden_size = config.mm_hidden_size # 设置视觉配置隐藏维度
        self.config.text_config.hidden_size = config.hidden_size # 设置文本配置隐藏维度
        self.multi_modal_projector = LlavaMultiModalProjector(config) # 创建多模态投影器
        self.mm_spatial_pool_stride = getattr(self.config, "mm_spatial_pool_stride", 2) # 获取空间池化步长，默认为2
        self.resampler = nn.AvgPool2d( # 创建二维平均池化重采样器
            kernel_size=self.mm_spatial_pool_stride, stride=self.mm_spatial_pool_stride # 核大小和步长均使用空间池化步长
        )
        self.language_model = LlamaForCausalLM( # 创建Llama语言模型
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.num_frames = getattr(self.config, "num_frames", 16) # 获取视频帧数，默认16
        if "unpad" in getattr(config, "mm_patch_merge_type", ""): # 如果补丁合并类型包含"unpad"
            self.language_model.model.image_newline = nn.Parameter( # 创建图像换行符参数
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )

    def pad_input_ids( # 填充输入ID，将图像标记替换为填充标记
        self, input_ids: array[int], image_inputs: MultimodalInputs
    ) -> array[int]:
        pad_values = array("q", (item.pad_value for item in image_inputs.mm_items)) # 获取每个多模态项的填充值
        new_image_feature_len = self.image_feature_len # 获取图像特征长度

        pad_ids = pad_values * ( # 生成足够的填充ID
            (new_image_feature_len + len(pad_values)) // len(pad_values)
        )
        offset = input_ids.index(self.config.image_token_index) # 找到图像标记在输入中的位置
        # old_len + pad_len - 1, because we need to remove image_token_id # 旧长度+填充长度-1，因为需要移除图像标记ID
        new_input_ids = ( # 构造新的输入ID序列
            input_ids[:offset] # 图像标记之前的部分
            + pad_ids[:new_image_feature_len] # 用填充标记替换图像标记
            + input_ids[offset + 1 :] # 图像标记之后的部分
        )
        image_inputs.image_offsets = [offset] # 记录图像偏移位置
        return new_input_ids # 返回填充后的输入ID

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor: # 编码图像，提取视觉特征
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True) # 通过视觉塔提取特征，输出所有隐藏状态
        # NOTE: This is not memory efficient. (output_hidden_states=True) will save all the hidden stated. # 注意：这并不节省内存。output_hidden_states=True会保存所有隐藏状态。

        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer] # 选择指定层的隐藏状态作为图像特征
        if self.vision_feature_select_strategy in ["default", "patch"]: # 如果特征选择策略为default或patch
            selected_image_feature = selected_image_feature[:, 1:] # 去除CLS标记
        elif self.vision_feature_select_strategy == "full": # 如果特征选择策略为full
            selected_image_feature = selected_image_feature # 保留全部特征
        else:
            raise ValueError( # 否则抛出异常
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )

        height = width = self.num_patches_per_side # 获取每边补丁数
        num_of_frames = selected_image_feature.shape[0] # 获取帧数
        selected_image_feature = selected_image_feature.view( # 重塑特征形状为(帧数, 高度, 宽度, 特征维度)
            num_of_frames, height, width, -1
        )
        selected_image_feature = selected_image_feature.permute(0, 3, 1, 2).contiguous() # 转置为(帧数, 特征维度, 高度, 宽度)
        selected_image_feature = ( # 通过重采样器进行空间池化
            self.resampler(selected_image_feature) # 应用二维平均池化
            .flatten(2) # 展平空间维度
            .transpose(1, 2) # 转置为(帧数, 空间位置, 特征维度)
            .contiguous()
        )

        image_features = self.multi_modal_projector(selected_image_feature) # 通过多模态投影器投影特征

        return image_features # 返回编码后的图像特征

    @torch.no_grad() # 禁用梯度计算
    def forward( # 前向推理方法
        self,
        input_ids: torch.LongTensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        image_inputs = forward_batch.mm_inputs # 获取多模态输入
        if forward_batch.forward_mode.is_extend(): # 如果是扩展模式（prefill）
            bs = forward_batch.batch_size # 获取批次大小

            # Clamp input ids. See llava.py for more details # 限制输入ID范围，详见llava.py
            input_ids = input_ids.clamp_(min=0, max=self.config.vocab_size - 1) # 将负值ID限制到有效范围

            # Embed text inputs # 嵌入文本输入
            input_embeds = self.language_model.model.embed_tokens(input_ids) # 通过词嵌入层获取文本嵌入

            # Whether the requests need vision inputs # 判断请求是否需要视觉输入
            max_image_offset = [] # 最大图像偏移列表
            for im in image_inputs: # 遍历每个请求的多模态输入
                if im and im.image_offsets: # 如果存在图像偏移
                    max_image_offset.append(max(im.image_offsets)) # 记录最大偏移
                else:
                    max_image_offset.append(-1) # 无图像时设为-1
            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy() # 获取每个请求的起始位置
            need_vision = start_positions <= np.array(max_image_offset) # 判断哪些请求需要视觉特征

            if need_vision.any(): # 如果有请求需要视觉输入
                pixel_values = flatten_nested_list( # 展平像素值列表
                    [
                        [item.feature for item in image_inputs[i].mm_items] # 获取每个多模态项的特征
                        for i in range(bs)
                        if need_vision[i]
                    ]
                )
                image_offsets = [ # 获取图像偏移列表
                    flatten_nested_list(
                        [item.offsets for item in image_inputs[i].mm_items] # 获取每个项的偏移
                    )
                    for i in range(bs)
                    if need_vision[i]
                ]

                ########## Encode Image ######## # 编码图像

                if pixel_values[0].ndim == 4: # 如果是4维张量（HD格式）
                    # llava-hd: BS, num_patch, C=3, H=336, W=336, num_patch obtained from process_images # llava-hd格式：批次大小, 补丁数, 通道数=3, 高=336, 宽=336
                    np.concatenate(pixel_values, axis=0) # 在批次维度上拼接像素值
                    # ndim=4 # 维度为4
                    concat_images = torch.tensor( # 将拼接后的数组转为张量
                        np.concatenate(pixel_values, axis=0),
                        device=self.vision_tower.device, # 放到视觉塔所在设备
                    )
                    # image_features = self.encode_images(concat_images)
                    # split_sizes = [image.shape[0] for image in pixel_values]
                    # image_features = torch.split(image_features, split_sizes, dim=0)
                    image_features = self.encode_images( # 编码拼接后的图像
                        concat_images
                    )  # , prompts)#, image_counts, long_video=long_video)
                    split_sizes = [image.shape[0] for image in pixel_values] # 计算每张图像的补丁数
                    image_features = torch.split(image_features, split_sizes, dim=0) # 按补丁数拆分特征

                    # hd image_features: BS, num_patch, 576, 4096 # HD图像特征形状
                else: # 普通像素格式
                    # normal pixel: BS, C=3, H=336, W=336 # 普通像素：批次大小, 通道=3, 高=336, 宽=336
                    pixel_values = torch.tensor( # 转为张量
                        np.array(pixel_values), device=self.vision_tower.device
                    )
                    image_features = self.encode_images(pixel_values) # 编码图像
                    # image_features: BS, 576, 4096 # 图像特征形状

                new_image_features = [] # 新图像特征列表
                for image_idx, image_feature in enumerate(image_features): # 遍历每张图像的特征
                    new_image_features.append(image_feature.flatten(0, 1)) # 将帧和补丁维度展平
                image_features = new_image_features # 更新图像特征

                # Fill in the placeholder for the image # 填充图像占位符
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy() # 获取扩展起始位置
                prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu # 获取前缀长度
                pt = 0 # 图像特征指针
                for i in range(bs): # 遍历每个请求
                    if not need_vision[i]: # 如果不需要视觉输入则跳过
                        continue

                    start_idx = extend_start_loc_cpu[i] # 获取当前请求在批次中的起始索引
                    prefix_len = prefix_lens_cpu[i] # 获取前缀长度

                    # Multiple images # 多张图像
                    for image_offset in image_offsets[i]: # 遍历每个图像偏移
                        if image_offset < prefix_len: # 如果图像偏移在前缀内则跳过
                            continue

                        tmp_image_feature = image_features[pt] # 获取当前图像特征
                        pad_len = tmp_image_feature.shape[0] # 获取填充长度

                        left_idx = start_idx + (image_offset - prefix_len) # 计算左边界索引
                        right_idx = start_idx + (image_offset - prefix_len) + pad_len # 计算右边界索引
                        try:
                            input_embeds[left_idx:right_idx] = tmp_image_feature # 将图像特征填入嵌入矩阵
                        except RuntimeError as e: # 捕获运行时错误
                            print(f"RuntimeError in image encoding: {e}") # 打印错误信息
                            print(f"{input_embeds.shape=}, {tmp_image_feature.shape=}") # 打印张量形状
                            print(
                                f"{start_idx=}, {image_offset=}, {prefix_len=}, {pad_len=}" # 打印调试信息
                            )
                        pt += 1 # 移动到下一个图像特征

            return self.language_model( # 通过语言模型处理
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        elif forward_batch.forward_mode.is_decode(): # 如果是解码模式
            return self.language_model(input_ids, positions, forward_batch) # 直接通过语言模型处理

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        # Load clip vision model by cfg['mm_vision_tower']: # 通过配置加载CLIP视觉模型
        # huggingface_name or path_of_clip_relative_to_llava_model_dir # HuggingFace名称或CLIP模型的相对路径
        # We put the initialization here instead of __init__ to allow it being reused by other subclasses. # 将初始化放在此处而非__init__中，以允许其他子类复用
        vision_path = self.config.mm_vision_tower # 获取视觉模型路径
        self.vision_tower = CLIPVisionModel.from_pretrained( # 加载预训练的CLIP视觉模型
            vision_path, torch_dtype=torch.float16
        ).cuda()
        self.vision_tower.eval() # 设置为评估模式

        self.vision_feature_layer = self.config.mm_vision_select_layer # 获取视觉特征选择层
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature # 获取视觉特征选择策略
        self.image_size = self.vision_tower.config.image_size # 获取图像尺寸
        self.patch_size = self.vision_tower.config.patch_size # 获取补丁尺寸

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat") # 获取补丁合并类型，默认flat
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square") # 获取图像宽高比，默认square
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None) # 获取图像网格锚点

        print(f"target_frames: {self.num_frames}") # 打印目标帧数
        self.image_feature_len = self.num_frames * int( # 计算图像特征长度
            (self.image_size / self.patch_size / self.mm_spatial_pool_stride) ** 2
        )
        if self.vision_feature_select_strategy == "patch": # 如果特征选择策略为patch
            pass # 无需调整
        elif self.vision_feature_select_strategy == "cls_patch": # 如果特征选择策略为cls_patch
            self.image_feature_len += 1 # 特征长度加1（包含CLS标记）
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}") # 抛出异常

        # load mm_projector # 加载多模态投影器
        projector_weights = { # 投影器权重名称映射
            "model.mm_projector.0": "multi_modal_projector.linear_1", # 投影器第一层线性层
            "model.mm_projector.2": "multi_modal_projector.linear_2", # 投影器第二层线性层
            "model.vision_resampler.mm_projector.0": "multi_modal_projector.linear_1", # 视觉重采样器投影器第一层
            "model.vision_resampler.mm_projector.2": "multi_modal_projector.linear_2", # 视觉重采样器投影器第二层
            "model.vision_tower.vision_tower": "vision_tower", # 视觉塔
            # transformers 5.6.0 flattened CLIPVisionModel/SiglipVisionModel, # transformers 5.6.0扁平化了CLIPVisionModel/SiglipVisionModel
            # dropping the `vision_model` intermediate wrapper. # 移除了vision_model中间包装器
            "vision_tower.vision_model.": "vision_tower.", # 视觉模型前缀映射
            # Update the vision tower weights if we find them in the checkpoint (it may be finetuned). # 如果检查点中包含视觉塔权重则更新（可能已被微调）
            "model.image_newline": "language_model.model.image_newline", # 图像换行符权重映射
        }
        params_dict = dict(self.named_parameters()) # 获取模型参数字典
        for name, loaded_weight in weights: # 遍历所有权重
            # FIXME: why projector weights read two times? # 待修复：投影器权重为何被读取两次？
            if "projector" in name or "vision_tower" in name or "image_newline" in name: # 如果是投影器、视觉塔或换行符权重
                for weight_name, param_name in projector_weights.items(): # 遍历权重名称映射
                    if weight_name in name: # 如果匹配
                        name = name.replace(weight_name, param_name) # 替换名称
                if name in params_dict: # 如果参数存在
                    param = params_dict[name] # 获取参数
                else:
                    print(f"Warning: {name} not found in the model") # 打印警告信息
                    continue # 跳过
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重
            else: # 其他权重
                self.language_model.load_weights([(name, loaded_weight)]) # 通过语言模型加载

    @property
    def num_patches_per_side(self): # 每边补丁数属性
        return self.image_size // self.patch_size # 返回图像尺寸除以补丁尺寸


EntryClass = LlavaVidForCausalLM # 入口类为LlavaVidForCausalLM
