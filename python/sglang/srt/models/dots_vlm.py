# Dots-VL 视觉语言模型实现
# 本文件实现了Dots-VL多模态视觉语言模型，
# 结合DotsVisionTransformer视觉编码器和DeepseekV2语言模型，
# 支持图像和视频输入，兼容HuggingFace权重格式。

# Copyright 2025 The RedNote HiLab team.
# Copyright 2025 The SGLang team.
# 版权所有 2025 RedNote HiLab团队。
# 版权所有 2025 SGLang团队。
#
# This code is based on the DeepseekVL2ForCausalLM and DotsVisionTransformer
# implementation in this library.
# 本代码基于本库中的DeepseekVL2ForCausalLM和DotsVisionTransformer实现。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 根据Apache许可证2.0版（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按"原样"分发的，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解管理权限和限制的特定语言。
"""Inference-only Dots-VL model compatible with HuggingFace weights.
仅推理的Dots-VL模型，兼容HuggingFace权重。"""

from typing import Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch核心库
from torch import nn  # 导入神经网络模块

from sglang.srt.configs.dots_vlm import DotsVLMConfig  # 导入DotsVLM配置类
from sglang.srt.distributed import get_pp_group  # 导入获取流水线并行组函数
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs  # 导入多模态数据项和输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和流水线代理张量
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM  # 导入DeepseekV2因果语言模型

from .dots_vlm_vit import DotsVisionTransformer  # 导入Dots视觉Transformer


class DotsVLMForCausalLM(nn.Module):
    """DotsVLM model for sglang inference
    用于SGLang推理的DotsVLM模型"""

    def __init__(  # 初始化方法
        self, config: DotsVLMConfig, quant_config: Optional[QuantizationConfig] = None  # 配置和量化配置
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.image_token_id = config.im_span_id  # 获取图像token ID
        self.video_token_id = config.video_span_id  # 获取视频token ID
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if not config.encoder_only:  # 如果不是仅编码器模式
            self.language_model = DeepseekV2ForCausalLM(  # 创建DeepseekV2语言模型
                config.language_config, quant_config  # 传入语言配置和量化配置
            )

        # Initialize vision tower (matching transformers naming for weight compatibility)
        # 初始化视觉塔（匹配transformers命名以保持权重兼容性）
        self.vision_tower = DotsVisionTransformer(config.vision_config)  # 创建Dots视觉Transformer

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):  # 为虚拟头填充注意力QKV权重
        """pad attn qkv weights for dummy heads
        为虚拟头填充注意力qkv权重"""
        num_dummy_heads = self.config.vision_config.num_dummy_heads  # 获取虚拟头数量
        if num_dummy_heads == 0:  # 如果没有虚拟头
            return loaded_weight  # 直接返回原始权重
        head_dim = self.config.vision_config.head_dim  # 获取每个头的维度

        if "attn.qkv_proj" in name:  # 如果是QKV投影权重
            wq, wk, wv = loaded_weight.chunk(3, dim=0)  # 将权重拆分为Q、K、V三部分
            if name.endswith(".weight"):  # 如果是权重张量
                dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]  # 虚拟头权重的形状
            elif name.endswith(".bias"):  # 如果是偏置张量
                dummy_shape = [num_dummy_heads, head_dim]  # 虚拟头偏置的形状
            else:  # 其他情况
                raise RuntimeError(f"Unsupported weight with name={name}")  # 抛出运行时错误
            pad_func = lambda x: torch.cat(  # 填充函数：在头部维度末尾添加零填充
                [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0  # 拆分头部维度后拼接零张量
            ).flatten(0, 1)  # 再展平回原格式
            wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)  # 对Q、K、V分别填充
            loaded_weight = torch.cat([wq, wk, wv], dim=0)  # 重新拼接为完整权重
        if "attn.proj.weight" in name:  # 如果是输出投影权重
            padded_weight = loaded_weight.new_zeros(  # 创建零填充权重
                loaded_weight.shape[0], head_dim * num_dummy_heads  # 在最后一维添加虚拟头维度
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)  # 拼接填充权重
        if "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:  # 如果是Q或K归一化权重
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)  # 创建零填充归一化权重
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)  # 拼接填充权重
        return loaded_weight  # 返回填充后的权重

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        """Load weights for the model, separating vision and language weights
        加载模型权重，分离视觉和语言权重"""
        weights = list(weights)  # 将权重迭代器转为列表

        # Separate vision tower weights and language model weights
        # 分离视觉塔权重和语言模型权重
        vision_weights = []  # 视觉权重列表
        language_weights = []  # 语言权重列表

        for name, loaded_weight in weights:  # 遍历所有权重
            if name.startswith("vision_tower."):  # 如果是视觉塔权重
                vision_name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换QKV命名格式
                vision_weights.append((vision_name, loaded_weight))  # 添加到视觉权重列表
            else:  # 否则
                # All other weights go to language model
                # 所有其他权重归属语言模型
                language_weights.append((name, loaded_weight))  # 添加到语言权重列表

        # Load vision tower weights
        # 加载视觉塔权重
        if not self.config.language_only:  # 如果不是仅语言模式
            vision_state_dict = dict(vision_weights)  # 将视觉权重转为字典
            params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取模型参数字典
            for name, loaded_weight in vision_state_dict.items():  # 遍历视觉权重
                if name not in params_dict:  # 如果参数名不存在
                    raise ValueError(f"Weight {name} not found in params_dict")  # 抛出值错误
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                loaded_weight = self._pad_vit_attn_dummy_heads(name, loaded_weight)  # 填充虚拟头权重
                weight_loader(param, loaded_weight)  # 加载权重

        # Load language model weights
        # 加载语言模型权重
        if not self.config.encoder_only and language_weights:  # 如果不是仅编码器模式且有语言权重
            self.language_model.load_weights(language_weights)  # 加载语言模型权重

    @classmethod  # 类方法装饰器
    def get_model_config_for_expert_location(cls, config):  # 获取专家位置相关的模型配置
        return DeepseekV2ForCausalLM.get_model_config_for_expert_location(config)  # 委托给DeepseekV2

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID，插入多模态token
        """Pad input_ids with multimodal tokens
        用多模态token填充input_ids"""
        # Get image token ID for padding pattern
        # 获取图像token ID用于填充模式
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态token填充模式
        padded_input_ids = pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式处理输入token
        return padded_input_ids  # 返回填充后的输入ID

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 提取图像特征
        # Extract pixel values and grid information (following reference pattern)
        # 提取像素值和网格信息（遵循参考模式）
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接所有图像的像素值
            self.vision_tower.dtype  # 转换为视觉塔的数据类型
        )
        image_grid_thw = torch.concat(  # 拼接所有图像的网格信息（时间、高度、宽度）
            [item.image_grid_thw for item in items], dim=0
        ).to(self.vision_tower.device)  # 移动到视觉塔的设备

        # Add dimension checks like in reference code
        # 添加维度检查，如参考代码中所示
        assert pixel_values.dim() == 2, f"{pixel_values.dim()=}"  # 断言像素值为2维
        assert image_grid_thw.dim() == 2, f"{image_grid_thw.dim()=}"  # 断言网格信息为2维

        # Process through vision tower
        # 通过视觉塔处理
        image_embeds = self.vision_tower(pixel_values, image_grid_thw)  # 调用视觉Transformer获取图像嵌入

        # Ensure consistent dtype for FlashInfer compatibility
        # 确保数据类型一致以兼容FlashInfer
        # Force bfloat16 to match model's expected dtype
        # 强制bfloat16以匹配模型期望的数据类型
        if image_embeds.dtype != torch.bfloat16 and hasattr(  # 如果不是bfloat16且语言模型有嵌入层
            self.language_model.model, "embed_tokens"
        ):
            target_dtype = self.language_model.model.embed_tokens.weight.dtype  # 获取目标数据类型
            image_embeds = image_embeds.to(target_dtype)  # 转换数据类型

        return image_embeds  # 返回图像嵌入

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，可选
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
                input_ids=input_ids,  # 输入token ID
                positions=positions,  # 位置编码
                forward_batch=forward_batch,  # 前向批次信息
                multimodal_model=self,  # 多模态模型（自身）
                language_model=self.language_model,  # 语言模型
            )

        else:  # 否则（非第一个秩）
            hidden_states = self.language_model(  # 直接调用语言模型
                input_ids=input_ids,  # 输入token ID
                positions=positions,  # 位置编码
                forward_batch=forward_batch,  # 前向批次信息
                pp_proxy_tensors=pp_proxy_tensors,  # 流水线代理张量
            )

        return hidden_states  # 返回隐藏状态


EntryClass = [DotsVLMForCausalLM]  # 模型入口类注册
