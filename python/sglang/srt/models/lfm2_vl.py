# LFM2-VL视觉语言模型的SGLang推理实现
# 结合SigLip2视觉编码器和LFM2语言模型，支持NaFlex可变分辨率
# Copyright 2026 Liquid AI. All rights reserved.
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
"""Inference-only LFM2-VL model compatible with HuggingFace weights. # 仅推理的LFM2-VL模型，兼容HuggingFace权重

LFM2-VL is a vision-language model that combines: # LFM2-VL是一个视觉语言模型，结合了：
- SigLip2 vision encoder with NaFlex variable-resolution support # SigLip2视觉编码器，支持NaFlex可变分辨率
- LFM2 language model (hybrid attention + short convolution) # LFM2语言模型（混合注意力+短卷积）
- Multimodal projector with pixel unshuffle downsampling # 带有像素反洗牌下采样的多模态投影器
"""

import logging # 导入日志模块
from typing import Iterable, List, Optional, Tuple # 导入类型提示

import numpy as np # 导入NumPy数值计算库
import torch # 导入PyTorch深度学习框架
from torch import nn # 导入神经网络模块
from transformers.activations import ACT2FN # 导入激活函数映射

from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear # 导入并行线性层
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.managers.mm_utils import ( # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens, # 多模态数据填充模式
    general_mm_embed_routine, # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import ( # 导入调度批次
    MultimodalDataItem, # 多模态数据项
    MultimodalInputs, # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.lfm2 import Lfm2ForCausalLM # 导入LFM2因果语言模型
from sglang.srt.models.siglip2 import Siglip2Model # 导入SigLip2模型
from sglang.srt.utils import add_prefix # 导入前缀添加工具

logger = logging.getLogger(__name__) # 获取日志记录器


class Lfm2VlMultiModalProjector(nn.Module): # LFM2-VL多模态投影器类
    """Multimodal projector with pixel unshuffle downsampling and TP/DP support.""" # 带像素反洗牌下采样和TP/DP支持的多模态投影器

    def __init__( # 初始化方法
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 可选的量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        in_channels = config.vision_config.hidden_size * (config.downsample_factor**2) # 输入通道数=视觉隐藏维度*下采样因子²
        self.factor = config.downsample_factor # 保存下采样因子
        self.use_layer_norm = config.projector_use_layernorm # 是否使用LayerNorm
        self.layer_norm = ( # 条件性创建LayerNorm
            nn.LayerNorm(in_channels) if config.projector_use_layernorm else None
        )

        self.linear_1 = ColumnParallelLinear( # 第一层并行线性层
            in_channels,
            config.projector_hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
        )
        self.act = ACT2FN[config.projector_hidden_act] # 获取激活函数
        self.linear_2 = RowParallelLinear( # 第二层行并行线性层
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
        )

    def forward( # 前向传播方法
        self,
        vision_features_packed: torch.Tensor, # 打包的视觉特征
        spatial_shapes: torch.Tensor, # 空间形状
    ) -> torch.Tensor:
        """Project packed vision features with pixel unshuffle. # 使用像素反洗牌投影打包的视觉特征

        Args: # 参数：
            vision_features_packed: (total_tokens, hidden_size) packed in tile order. # (总令牌数, 隐藏维度)按瓦片顺序打包
            spatial_shapes: (num_tiles, 2) on CPU (height, width) per tile. # (瓦片数, 2)在CPU上，每个瓦片的高宽

        Returns: # 返回：
            projected_packed: (total_projected_tokens, text_hidden_size) # (总投影令牌数, 文本隐藏维度)
        """
        factor = self.factor # 获取下采样因子
        hidden_size = vision_features_packed.shape[-1] # 获取隐藏维度

        # Compute tile lengths from spatial shapes # 从空间形状计算瓦片长度
        lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist() # 计算每个瓦片的令牌数

        # Split packed tensor into per-tile tensors # 将打包的张量拆分为每个瓦片的张量
        tile_features = torch.split(vision_features_packed, lengths, dim=0) # 按瓦片长度拆分

        # Apply pixel unshuffle to each tile using reshape/permute (GPU operations) # 使用reshape/permute对每个瓦片应用像素反洗牌（GPU操作）
        unshuffled_parts = [] # 反洗牌结果列表
        for tile, (h, w) in zip(tile_features, spatial_shapes.tolist()): # 遍历每个瓦片及其空间形状
            if h == 0 or w == 0: # 如果高度或宽度为零
                continue # 跳过
            # Reshape: (H*W, C) -> (H, W, C) -> (H/f, f, W/f, f, C) # 重塑：(H*W, C) -> (H, W, C) -> (H/f, f, W/f, f, C)
            tile_2d = tile.view(h, w, hidden_size) # 重塑为二维
            tile_blocks = tile_2d.view( # 重塑为分块形式
                h // factor, factor, w // factor, factor, hidden_size
            )
            # Permute: (H/f, f, W/f, f, C) -> (H/f, W/f, f, f, C) # 置换维度
            tile_permuted = tile_blocks.permute(0, 2, 1, 3, 4)
            # Reshape: (H/f, W/f, f*f*C) # 重塑为反洗牌后的形状
            tile_unshuffled = tile_permuted.reshape(
                (h // factor) * (w // factor), factor * factor * hidden_size
            )
            unshuffled_parts.append(tile_unshuffled) # 添加到结果列表

        if unshuffled_parts: # 如果有反洗牌结果
            unshuffled = torch.cat(unshuffled_parts, dim=0) # 在令牌维度上拼接
        else: # 否则创建空张量
            unshuffled = vision_features_packed.new_empty(
                (0, factor * factor * hidden_size)
            )

        if self.use_layer_norm: # 如果使用LayerNorm
            unshuffled = self.layer_norm(unshuffled) # 应用LayerNorm
        hidden_states, _ = self.linear_1(unshuffled) # 通过第一层线性层
        hidden_states = self.act(hidden_states) # 应用激活函数
        projected_packed, _ = self.linear_2(hidden_states) # 通过第二层线性层
        return projected_packed # 返回投影结果


class Lfm2VlForConditionalGeneration(nn.Module): # LFM2-VL条件生成模型类
    """LFM2-VL Vision-Language Model.""" # LFM2-VL视觉语言模型

    def __init__( # 初始化方法
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 可选的量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置

        # Vision tower: Native Siglip2 implementation # 视觉塔：原生Siglip2实现
        self.vision_tower = Siglip2Model( # 创建SigLip2视觉模型
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        # Multimodal projector # 多模态投影器
        self.multi_modal_projector = Lfm2VlMultiModalProjector( # 创建多模态投影器
            config,
            quant_config=quant_config,
            prefix=add_prefix("multi_modal_projector", prefix),
        )

        # Language model: reuse SGLang's LFM2 implementation # 语言模型：复用SGLang的LFM2实现
        self.language_model = Lfm2ForCausalLM( # 创建LFM2语言模型
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config) # 创建logits处理器

    def pad_input_ids( # 填充输入ID
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens() # 创建多模态数据填充模式
        result = pattern.pad_input_tokens(input_ids, mm_inputs) # 使用模式填充令牌
        return result # 返回填充结果

    def get_input_embeddings(self) -> nn.Embedding: # 获取输入嵌入层
        return self.language_model.model.embed_tokens # 返回语言模型的词嵌入层

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor: # 获取图像特征
        """Process images through vision tower and projector. # 通过视觉塔和投影器处理图像

        Handles SigLip2's NaFlex variable-resolution output. # 处理SigLip2的NaFlex可变分辨率输出
        Pixel values arrive padded from the base processor; we pack them # 像素值从基础处理器以填充形式传入；我们将它们打包
        using the attention mask before feeding into the vision tower. # 使用注意力掩码后再输入视觉塔
        """
        # Collect data from all items # 从所有数据项收集数据
        all_pixel_values = [] # 所有像素值列表
        all_attention_masks = [] # 所有注意力掩码列表
        all_spatial_shapes = [] # 所有空间形状列表

        for item in items: # 遍历每个数据项
            pv = item.feature # 获取像素值
            am = item.pixel_attention_mask # 获取像素注意力掩码
            ss = item.spatial_shapes # 获取空间形状

            if isinstance(pv, np.ndarray): # 如果像素值是NumPy数组
                pv = torch.from_numpy(pv) # 转为PyTorch张量
            if isinstance(am, np.ndarray): # 如果注意力掩码是NumPy数组
                am = torch.from_numpy(am) # 转为PyTorch张量
            if isinstance(ss, np.ndarray): # 如果空间形状是NumPy数组
                ss = torch.from_numpy(ss) # 转为PyTorch张量

            all_pixel_values.append(pv) # 添加到像素值列表
            all_attention_masks.append(am) # 添加到注意力掩码列表
            all_spatial_shapes.append(ss) # 添加到空间形状列表

        pixel_values = torch.cat(all_pixel_values, dim=0) # 在批次维度上拼接像素值
        attention_mask = torch.cat(all_attention_masks, dim=0) # 拼接注意力掩码
        spatial_shapes = torch.cat(all_spatial_shapes, dim=0) # 拼接空间形状

        pixel_values = pixel_values.to( # 将像素值移到视觉塔所在设备
            device=self.vision_tower.device,
            dtype=self.vision_tower.dtype,
        )
        spatial_shapes_cpu = spatial_shapes.cpu() # 将空间形状移到CPU

        # Pack padded pixel values using attention mask # 使用注意力掩码打包填充的像素值
        packed_list = [] # 打包结果列表
        for i in range(pixel_values.shape[0]): # 遍历每个瓦片
            mask = attention_mask[i].bool() # 获取当前瓦片的注意力掩码
            packed_list.append(pixel_values[i][mask]) # 仅保留掩码为True的像素值

        if not packed_list: # 如果没有打包结果
            return torch.tensor( # 返回空张量
                [], device=self.vision_tower.device, dtype=self.vision_tower.dtype
            )

        pixel_values_packed = torch.cat(packed_list, dim=0) # 拼接打包的像素值

        # Compute cu_seqlens and max_seqlen for packed attention # 为打包注意力计算累积序列长度和最大序列长度
        spatial_shapes_list = spatial_shapes_cpu.tolist() # 将空间形状转为列表
        lengths_list = [int(h * w) for h, w in spatial_shapes_list] # 计算每个瓦片的令牌数
        total_tokens = sum(lengths_list) # 计算总令牌数

        if total_tokens == 0: # 如果没有令牌
            return torch.tensor( # 返回空张量
                [], device=self.vision_tower.device, dtype=self.vision_tower.dtype
            )

        lengths = torch.tensor( # 创建令牌长度张量
            lengths_list, dtype=torch.int32, device=pixel_values_packed.device
        )
        cu_seqlens = torch.zeros( # 创建累积序列长度张量
            len(lengths_list) + 1,
            dtype=torch.int32,
            device=pixel_values_packed.device,
        )
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0) # 计算累积和
        max_seqlen = lengths.max() # 获取最大序列长度

        # Forward through vision tower # 通过视觉塔前向传播
        vision_outputs = self.vision_tower( # 调用视觉塔
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes_cpu,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Get the packed features (remove batch dim if present) # 获取打包特征（如有批次维度则移除）
        if vision_outputs.dim() == 3: # 如果是3维
            vision_features_packed = vision_outputs[0] # 去除批次维度
        else:
            vision_features_packed = vision_outputs # 直接使用

        # Project through multimodal projector # 通过多模态投影器投影
        projected_packed = self.multi_modal_projector( # 调用多模态投影器
            vision_features_packed=vision_features_packed,
            spatial_shapes=spatial_shapes_cpu,
        )

        return projected_packed # 返回投影结果

    @torch.no_grad() # 禁用梯度计算
    def forward( # 前向推理方法
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        return general_mm_embed_routine( # 调用通用多模态嵌入例程
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载权重
        """Load weights from HuggingFace format.""" # 从HuggingFace格式加载权重
        # Collect weights by destination # 按目标收集权重
        vision_weights = [] # 视觉塔权重列表
        projector_weights = [] # 投影器权重列表
        lm_weights = [] # 语言模型权重列表

        for name, loaded_weight in weights: # 遍历所有权重
            if name.startswith("model.vision_tower."): # 如果是视觉塔权重
                # model.vision_tower.* → * (strip model.vision_tower. prefix) # 去除model.vision_tower.前缀
                # siglip2.py expects names like "vision_model.embeddings.patch_embedding.weight" # siglip2.py期望类似vision_model.embeddings.patch_embedding.weight的名称
                new_name = name.replace("model.vision_tower.", "", 1) # 去除前缀
                vision_weights.append((new_name, loaded_weight)) # 添加到视觉权重列表
            elif name.startswith("model.multi_modal_projector."): # 如果是投影器权重
                # model.multi_modal_projector.* → multi_modal_projector.* # 去除model.前缀
                new_name = name.replace(
                    "model.multi_modal_projector.", "multi_modal_projector.", 1
                )
                projector_weights.append((new_name, loaded_weight)) # 添加到投影器权重列表
            elif name.startswith("model.language_model."): # 如果是语言模型权重
                # model.language_model.* → language_model.model.* # 映射到language_model.model.*
                new_name = name.replace(
                    "model.language_model.", "language_model.model.", 1
                )
                lm_weights.append((new_name, loaded_weight)) # 添加到语言模型权重列表
            elif name.startswith("lm_head."): # 如果是语言模型头权重
                # lm_head.* → language_model.lm_head.* # 映射到language_model.lm_head.*
                new_name = name.replace("lm_head.", "language_model.lm_head.", 1)
                lm_weights.append((new_name, loaded_weight)) # 添加到语言模型权重列表
            else:
                # Try direct mapping # 尝试直接映射
                lm_weights.append((name, loaded_weight)) # 直接添加到语言模型权重列表

        # Load vision tower weights using its own load_weights method # 使用视觉塔自己的load_weights方法加载权重
        self.vision_tower.load_weights(vision_weights) # 加载视觉塔权重

        # Load projector weights # 加载投影器权重
        params_dict = dict(self.named_parameters()) # 获取模型参数字典
        for name, loaded_weight in projector_weights: # 遍历投影器权重
            if name not in params_dict: # 如果参数不存在
                continue # 跳过
            param = params_dict[name] # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
            weight_loader(param, loaded_weight) # 加载权重

        # Load language model weights via Lfm2ForCausalLM.load_weights # 通过Lfm2ForCausalLM.load_weights加载语言模型权重
        # Strip the "language_model." prefix since Lfm2ForCausalLM expects # 去除"language_model."前缀，因为Lfm2ForCausalLM期望
        # names like "model.layers.0..." and "lm_head.weight" # 类似"model.layers.0..."和"lm_head.weight"的名称
        lm_weights_stripped = [] # 去除前缀后的语言模型权重列表
        for name, loaded_weight in lm_weights: # 遍历语言模型权重
            if name.startswith("language_model."): # 如果以language_model.开头
                name = name[len("language_model."):] # 去除前缀
            lm_weights_stripped.append((name, loaded_weight)) # 添加到列表
        self.language_model.load_weights(lm_weights_stripped) # 加载语言模型权重


EntryClass = Lfm2VlForConditionalGeneration # 入口类为Lfm2VlForConditionalGeneration
