# LightOnOCR视觉语言OCR模型的SGLang推理实现
# 结合Pixtral视觉编码器和Qwen3语言解码器
# Copyright 2025 SGLang Team
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

"""
Support for lightonai/LightOnOCR-2-1B. # 支持lightonai/LightOnOCR-2-1B模型

LightOnOCR is a vision-language OCR model that combines: # LightOnOCR是一个视觉语言OCR模型，结合了：
- Pixtral vision encoder (24 layers, 1024 hidden dim) # Pixtral视觉编码器（24层，1024隐藏维度）
- Spatial merge projection with RMSNorm + PatchMerger (2x2 = 4x token reduction) # 空间合并投影，包含RMSNorm和PatchMerger（2x2=4倍令牌缩减）
- Qwen3 language decoder (28 layers, 1024 hidden dim) # Qwen3语言解码器（28层，1024隐藏维度）

Key differences from PixtralForConditionalGeneration: # 与PixtralForConditionalGeneration的主要区别：
- Uses Qwen3ForCausalLM instead of MistralLarge3ForCausalLM as the language model # 使用Qwen3ForCausalLM而非MistralLarge3ForCausalLM作为语言模型
- Has an RMSNorm applied to vision encoder output before patch merging # 在补丁合并前对视觉编码器输出应用RMSNorm
- Does not use image break/end tokens (single contiguous image token range) # 不使用图像分隔/结束标记（单段连续图像标记范围）
- HuggingFace checkpoint uses a vision_projection namespace for norm, patch_merger, # HuggingFace检查点使用vision_projection命名空间存放norm、patch_merger
  and adapter weights # 和适配器权重

References: # 参考文献：
- https://huggingface.co/lightonai/LightOnOCR-2-1B
"""

from dataclasses import fields # 导入数据类字段工具
from typing import Iterable, List, Tuple # 导入类型提示

import torch # 导入PyTorch深度学习框架
import torch.nn as nn # 导入神经网络模块

from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.managers.mm_utils import ( # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens, # 多模态数据填充模式
    general_mm_embed_routine, # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs # 导入多模态数据项和输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.pixtral import ( # 从Pixtral模型导入组件
    PATCH_MERGE, # 补丁合并标识
    PatchMerger, # 补丁合并器
    PixtralHFVisionModel, # Pixtral HF视觉模型
    VisionEncoderArgs, # 视觉编码器参数
    VisionLanguageAdapter, # 视觉语言适配器
)
from sglang.srt.models.qwen3 import Qwen3ForCausalLM # 导入Qwen3因果语言模型


class LightOnOCRForConditionalGeneration(nn.Module): # LightOnOCR条件生成模型类
    """
    LightOnOCR model for SGLang inference. # 用于SGLang推理的LightOnOCR模型

    Architecture: # 架构：
    - Pixtral-based vision encoder (PixtralHFVisionModel, 24 layers) # 基于Pixtral的视觉编码器（24层）
    - RMSNorm on vision encoder output # 视觉编码器输出上的RMSNorm
    - Spatial merge via PatchMerger (2x2 = 4x token reduction) # 通过PatchMerger进行空间合并（4倍令牌缩减）
    - VisionLanguageAdapter projection to text hidden size # VisionLanguageAdapter投影到文本隐藏维度
    - Qwen3-based decoder (28 layers) with QK norms # 基于Qwen3的解码器（28层），带QK归一化
    """

    merge_by_field_config = True # 启用按字段合并配置

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: # 获取占位符字符串
        if modality.startswith("image"): # 如果模态以image开头
            return None # 返回None（无需占位符）
        raise ValueError("Only image modality is supported") # 仅支持图像模态

    def __init__(self, *, config, prefix: str = "", **kwargs): # 初始化方法
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        quant_config = kwargs.get("quant_config") # 获取量化配置

        # Build VisionEncoderArgs from config # 从配置构建视觉编码器参数
        vision_config = config.vision_config # 获取视觉配置
        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)} # 获取VisionEncoderArgs的字段名集合
        vision_args = { # 构建视觉参数字典
            key: value
            for key, value in vision_config.to_dict().items()
            if key in dataclass_fields # 仅保留匹配字段的键值对
        }
        # LightOnOCR stores these at the top-level config # LightOnOCR将这些参数存储在顶层配置中
        if "image_token_id" not in vision_args: # 如果缺少image_token_id
            vision_args["image_token_id"] = getattr(config, "image_token_id", 151655) # 从顶层配置获取，默认151655
        if "spatial_merge_size" not in vision_args: # 如果缺少spatial_merge_size
            vision_args["spatial_merge_size"] = getattr(config, "spatial_merge_size", 2) # 从顶层配置获取，默认2
        if "adapter_bias" not in vision_args: # 如果缺少adapter_bias
            vision_args["adapter_bias"] = getattr( # 从顶层配置获取
                config, "multimodal_projector_bias", True
            )
        # LightOnOCR uses patch merging for spatial merge # LightOnOCR使用补丁合并进行空间合并
        vision_args["mm_projector_id"] = PATCH_MERGE # 设置投影器类型为补丁合并
        self.vision_args = VisionEncoderArgs(**vision_args) # 创建视觉编码器参数对象

        # Vision encoder (Pixtral HF variant with SGLang parallel layers) # 视觉编码器（Pixtral HF变体，使用SGLang并行层）
        self.vision_encoder = PixtralHFVisionModel(vision_config, quant_config=None) # 创建Pixtral视觉模型

        # RMSNorm applied to vision encoder output before patch merging # 在补丁合并前对视觉编码器输出应用RMSNorm
        self.vision_projection_norm = RMSNorm(self.vision_args.hidden_size, eps=1e-5) # 创建RMSNorm层

        # Patch merger for spatial token reduction # 用于空间令牌缩减的补丁合并器
        self.patch_merger = PatchMerger( # 创建补丁合并器
            vision_encoder_dim=self.vision_args.hidden_size,
            spatial_merge_size=self.vision_args.spatial_merge_size,
        )

        # Vision-to-language projection adapter # 视觉到语言的投影适配器
        self.vision_language_adapter = VisionLanguageAdapter( # 创建视觉语言适配器
            self.vision_args, dim=config.text_config.hidden_size
        )

        # Language model # 语言模型
        self.language_model = Qwen3ForCausalLM( # 创建Qwen3语言模型
            config=config.text_config,
            quant_config=quant_config,
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs): # 填充输入ID
        pattern = MultiModalityDataPaddingPatternMultimodalTokens() # 创建多模态数据填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs) # 使用模式填充令牌

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor: # 获取图像特征
        """Process images through vision encoder and projection pipeline.""" # 通过视觉编码器和投影流水线处理图像
        images = [item.feature for item in items] # 从数据项中提取图像特征

        # Extract image sizes from model-specific data or infer from tensor shape # 从模型特定数据提取图像尺寸或从张量形状推断
        image_sizes_list = [] # 图像尺寸列表
        for item in items: # 遍历每个数据项
            if item.model_specific_data and "image_sizes" in item.model_specific_data: # 如果有模型特定数据中的image_sizes
                sizes_tensor = item.model_specific_data["image_sizes"] # 获取尺寸张量
                for size in sizes_tensor: # 遍历每个尺寸
                    image_sizes_list.append((int(size[0]), int(size[1]))) # 转换为整数元组
            else: # 否则从张量形状推断
                img = item.feature # 获取图像张量
                for _ in range(img.shape[0]): # 遍历每个图像
                    image_sizes_list.append((img.shape[-2], img.shape[-1])) # 从张量形状获取高宽

        # Stack pixel values # 堆叠像素值
        if len(images) > 1: # 如果有多张图像
            pixel_values = torch.cat(images, dim=0) # 在批次维度上拼接
        else:
            pixel_values = images[0] # 直接使用单张图像

        # Vision encoder forward # 视觉编码器前向传播
        image_features = self.vision_encoder(pixel_values, image_sizes=image_sizes_list) # 通过视觉编码器
        image_features = image_features.view(-1, image_features.shape[-1]) # 展平为二维张量

        # Norm before patch merge (matches HF Mistral3MultiModalProjector order) # 补丁合并前的归一化（与HF Mistral3MultiModalProjector顺序一致）
        image_features = self.vision_projection_norm(image_features) # 应用RMSNorm

        # Spatial merge via patch merger — use actual image sizes (not padded tensor # 通过补丁合并器进行空间合并——使用实际图像尺寸（非填充张量
        # shape) because PixtralHFVisionModel crops embeddings to real dimensions. # 尺寸），因为PixtralHFVisionModel将嵌入裁剪为真实尺寸
        patch_size = self.vision_args.patch_size # 获取补丁尺寸
        img_patch_dims = [ # 计算每张图像的补丁维度
            (h // patch_size, w // patch_size) for (h, w) in image_sizes_list
        ]
        image_features = self.patch_merger(image_features, image_sizes=img_patch_dims) # 应用补丁合并

        # Project to language model dimension # 投影到语言模型维度
        image_embeds = self.vision_language_adapter(image_features) # 通过视觉语言适配器投影
        return image_embeds # 返回图像嵌入

    def get_language_model(self) -> torch.nn.Module: # 获取语言模型
        return self.language_model # 返回语言模型

    def forward( # 前向推理方法
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批次信息
    ):
        return general_mm_embed_routine( # 调用通用多模态嵌入例程
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def compute_logits( # 计算logits
        self,
        hidden_states: torch.Tensor, # 隐藏状态
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states) # 通过语言模型计算logits

    def get_embed_and_head(self): # 获取嵌入层和输出头
        return self.language_model.get_embed_and_head() # 从语言模型获取

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载权重
        """Load weights from HuggingFace checkpoint. # 从HuggingFace检查点加载权重

        HF checkpoint weight layout (after stripping ``model.`` prefix): # HF检查点权重布局（去除model.前缀后）：
        - ``vision_encoder.*`` -> self.vision_encoder # 视觉编码器权重
        - ``vision_projection.norm.*`` -> self.vision_projection_norm # 视觉投影归一化权重
        - ``vision_projection.patch_merger.*`` -> self.patch_merger # 补丁合并器权重
        - ``vision_projection.linear_1.*`` -> self.vision_language_adapter.w_in # 适配器输入层权重
        - ``vision_projection.linear_2.*`` -> self.vision_language_adapter.w_out # 适配器输出层权重
        - ``language_model.*`` -> self.language_model (Qwen3ForCausalLM) # 语言模型权重
        """
        vision_encoder_dict = dict(self.vision_encoder.named_parameters()) # 视觉编码器参数字典
        patch_merger_dict = dict(self.patch_merger.named_parameters()) # 补丁合并器参数字典
        norm_dict = dict(self.vision_projection_norm.named_parameters()) # 归一化层参数字典
        adapter_dict = dict(self.vision_language_adapter.named_parameters()) # 适配器参数字典

        # PixtralHFVisionModel uses SGLang parallel layers with stacked params # PixtralHFVisionModel使用SGLang并行层和堆叠参数
        stacked_params_mapping = [ # 堆叠参数映射
            (".attention.qkv_proj", ".attention.q_proj", "q"), # QKV投影中的Q
            (".attention.qkv_proj", ".attention.k_proj", "k"), # QKV投影中的K
            (".attention.qkv_proj", ".attention.v_proj", "v"), # QKV投影中的V
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0), # 门控上投影中的门控
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1), # 门控上投影中的上投影
        ]

        def llm_weights_generator(): # 语言模型权重生成器
            for name, w in weights: # 遍历所有权重
                # HF checkpoint prefixes all weights with model. # HF检查点为所有权重添加model.前缀
                if name.startswith("model."): # 如果以model.开头
                    name = name[len("model."):] # 去除model.前缀

                if name.startswith("vision_encoder."): # 如果是视觉编码器权重
                    trimmed = name[len("vision_encoder."):] # 去除vision_encoder.前缀

                    # Handle stacked params (QKV, gate/up) # 处理堆叠参数（QKV、gate/up）
                    loaded = False # 标记是否已加载
                    for param_name, weight_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射
                        if weight_name in trimmed: # 如果权重名称在名称中
                            transformed = trimmed.replace(weight_name, param_name) # 替换为堆叠参数名
                            if transformed in vision_encoder_dict: # 如果转换后的名称存在于参数字典中
                                param = vision_encoder_dict[transformed] # 获取参数
                                weight_loader = getattr( # 获取权重加载器
                                    param, "weight_loader", default_weight_loader
                                )
                                with torch.no_grad(): # 禁用梯度计算
                                    weight_loader(param, w, shard_id) # 加载权重分片
                                loaded = True # 标记为已加载
                                break # 跳出循环

                    if not loaded: # 如果未通过堆叠映射加载
                        # Handle o_proj -> proj rename # 处理o_proj到proj的重命名
                        if ".attention.o_proj" in trimmed: # 如果包含o_proj
                            trimmed = trimmed.replace( # 替换为proj
                                ".attention.o_proj", ".attention.proj"
                            )
                        if trimmed in vision_encoder_dict: # 如果名称在参数字典中
                            param = vision_encoder_dict[trimmed] # 获取参数
                            weight_loader = getattr( # 获取权重加载器
                                param, "weight_loader", default_weight_loader
                            )
                            with torch.no_grad(): # 禁用梯度计算
                                weight_loader(param, w) # 加载权重

                elif name.startswith("vision_projection."): # 如果是视觉投影权重
                    remaining = name[len("vision_projection."):] # 去除vision_projection.前缀

                    if remaining.startswith("patch_merger."): # 如果是补丁合并器权重
                        trimmed = remaining[len("patch_merger."):] # 去除patch_merger.前缀
                        if trimmed in patch_merger_dict: # 如果名称在参数字典中
                            param = patch_merger_dict[trimmed] # 获取参数
                            with torch.no_grad(): # 禁用梯度计算
                                default_weight_loader(param, w) # 使用默认加载器

                    elif remaining.startswith("norm."): # 如果是归一化权重
                        trimmed = remaining[len("norm."):] # 去除norm.前缀
                        if trimmed in norm_dict: # 如果名称在参数字典中
                            param = norm_dict[trimmed] # 获取参数
                            with torch.no_grad(): # 禁用梯度计算
                                default_weight_loader(param, w) # 使用默认加载器

                    else: # 其他投影权重
                        # linear_1 -> w_in, linear_2 -> w_out # 线性层1映射到w_in，线性层2映射到w_out
                        trimmed = remaining.replace("linear_1.", "w_in.").replace( # 替换名称
                            "linear_2.", "w_out."
                        )
                        if trimmed in adapter_dict: # 如果名称在适配器参数字典中
                            param = adapter_dict[trimmed] # 获取参数
                            with torch.no_grad(): # 禁用梯度计算
                                default_weight_loader(param, w) # 使用默认加载器

                else: # 其他权重
                    # Language model weights and any other weights # 语言模型权重和其他权重
                    if name.startswith("language_model."): # 如果是语言模型权重
                        # Qwen3ForCausalLM expects model.* prefix # Qwen3ForCausalLM期望model.*前缀
                        name = "model." + name[len("language_model."):] # 添加model.前缀
                    yield (name, w) # 生成权重元组

        self.language_model.load_weights(llm_weights_generator()) # 通过生成器加载语言模型权重


EntryClass = LightOnOCRForConditionalGeneration # 入口类为LightOnOCRForConditionalGeneration
