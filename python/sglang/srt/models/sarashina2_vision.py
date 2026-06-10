# Sarashina2Vision视觉语言模型实现文件
# 本文件实现了Sarashina2Vision模型，结合Llama文本骨干和Qwen2VL视觉编码器
# 支持多模态输入，将图像特征通过Qwen2视觉变换器和归一化层映射到语言模型的嵌入空间

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
"""Inference-only Sarashina2Vision model compatible with HuggingFace weights."""

import logging  # 导入日志模块
from typing import Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入PyTorch神经网络模块
from transformers import LlamaConfig  # 导入Llama配置类

from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.llama import LlamaForCausalLM  # 导入Llama因果语言模型
from sglang.srt.models.qwen2_vl import Qwen2VisionTransformer  # 导入Qwen2视觉变换器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取日志记录器


class Sarashina2VisionForCausalLM(nn.Module):  # Sarashina2Vision因果语言模型类
    """
    Sarashina2Vision model that combines:
    - Llama text backbone (sbintuitions/sarashina2-7b)
    - Qwen2VL vision encoder
    """
    # Sarashina2Vision模型结合了Llama文本骨干和Qwen2VL视觉编码器

    def __init__(  # 初始化方法
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        # Extract text and vision configurations
        # 提取文本和视觉配置
        text_config = getattr(config, "text_config", config)  # 获取文本配置
        vision_config = getattr(config, "vision_config", None)  # 获取视觉配置

        # Create vision transformer first (like original model)
        # 首先创建视觉变换器（与原始模型一致）
        if vision_config is not None:  # 如果视觉配置存在
            self.visual = Qwen2VisionTransformer(  # 创建Qwen2视觉变换器
                vision_config,  # 视觉配置
                norm_eps=getattr(config, "rms_norm_eps", 1e-5),  # 归一化epsilon
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("visual", prefix),  # 添加前缀
            )
        else:  # 如果视觉配置不存在
            self.visual = None  # 视觉编码器为None

        # Layer norm for vision outputs (matching original model)
        # 视觉输出的LayerNorm（匹配原始模型）
        self.norm = nn.LayerNorm(text_config.hidden_size)  # 归一化层

        # Create Llama text model (using 'llm' name to match original)
        # 创建Llama文本模型（使用'llm'名称以匹配原始模型）
        if hasattr(text_config, "model_type") and text_config.model_type == "llama":  # 如果是Llama类型
            llama_config = LlamaConfig(**text_config.__dict__)  # 创建Llama配置
            # Set vocab_size from main config if available
            # 如果可用，从主配置设置词汇表大小
            if hasattr(config, "vocab_size"):  # 如果配置有vocab_size
                llama_config.vocab_size = config.vocab_size  # 设置词汇表大小
            self.llm = LlamaForCausalLM(  # 创建Llama因果语言模型
                llama_config,  # Llama配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("llm", prefix),  # 添加前缀
            )
        else:  # 如果不是Llama类型
            # Set vocab_size from main config if available
            # 如果可用，从主配置设置词汇表大小
            if hasattr(config, "vocab_size"):  # 如果配置有vocab_size
                config.vocab_size = config.vocab_size  # 设置词汇表大小
            self.llm = LlamaForCausalLM(  # 创建Llama因果语言模型
                config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("llm", prefix),  # 添加前缀
            )

        # Image token indices from config
        # 从配置获取图像token索引
        self.image_token_index = getattr(config, "image_token_index", 14)  # 图像token索引
        self.start_image_token_index = getattr(  # 起始图像token索引
            config, "start_image_token_index", 102397  # 默认值
        )
        self.end_image_token_index = getattr(config, "end_image_token_index", 102398)  # 结束图像token索引

        # Ensure vocabulary size matches
        # 确保词汇表大小匹配
        if hasattr(config, "vocab_size"):  # 如果配置有vocab_size
            self.llm.config.vocab_size = config.vocab_size  # 设置语言模型词汇表大小

        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 创建池化层

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID方法
        """Pad input tokens with multimodal data hashes for RadixAttention."""
        # 使用多模态数据哈希填充输入token，用于RadixAttention
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式填充token

    def get_input_embeddings(self):  # 获取输入嵌入方法
        """Get input embeddings from the language model."""
        # 从语言模型获取输入嵌入
        return self.llm.get_input_embeddings()  # 返回语言模型的输入嵌入

    def get_image_embeds(  # 获取图像嵌入方法
        self,
        pixel_values: torch.Tensor,  # 像素值张量
        image_grid_thw: torch.Tensor,  # 图像网格时间-高度-宽度张量
    ) -> torch.Tensor:
        """Extract image embeddings using the vision transformer."""
        # 使用视觉变换器提取图像嵌入
        if self.visual is None:  # 如果视觉编码器未初始化
            raise ValueError("Visual encoder not initialized")  # 抛出异常

        # Use the existing Qwen2VisionTransformer forward method
        # 使用现有的Qwen2VisionTransformer前向方法
        hidden_states = self.visual(pixel_values, image_grid_thw)  # 通过视觉变换器

        # Apply normalization layer
        # 应用归一化层
        return self.norm(hidden_states)  # 返回归一化后的隐藏状态

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取图像特征方法
        """Extract image features for SGLang compatibility."""
        # 提取图像特征以兼容SGLang
        if self.visual is None:  # 如果视觉编码器未初始化
            raise ValueError("Visual encoder not initialized")  # 抛出异常

        # Concatenate pixel values and grid_thw from all items
        # 拼接所有数据项的像素值和grid_thw
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接像素值
            self.visual.dtype  # 转换为视觉模型精度
        )
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)  # 拼接网格信息

        assert pixel_values.dim() == 2, pixel_values.dim()  # 断言像素值为2维
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()  # 断言网格信息为2维

        # Use the get_image_embeds method
        # 使用get_image_embeds方法
        return self.get_image_embeds(pixel_values, image_grid_thw)  # 返回图像嵌入

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # 模型的前向传播
        # Handles token-to-feature mapping for expanded tokens
        # 处理扩展token的token到特征映射
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.llm.model,  # 语言模型
            multimodal_model=self,  # 多模态模型（自身）
            positions=positions,  # 位置编码
        )

        if get_embedding:  # 如果需要获取嵌入
            return self.pooler(hidden_states, forward_batch)  # 返回池化后的嵌入
        else:  # 否则
            return self.logits_processor(  # 返回logits处理结果
                input_ids, hidden_states, self.llm.lm_head, forward_batch  # 输入ID、隐藏状态、语言模型头、前向批次
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        """Load model weights."""
        # 加载模型权重
        params_dict = dict(self.named_parameters())  # 获取参数字典
        loaded_params = set()  # 已加载参数集合

        # Collect weights that need to be fused
        # 收集需要融合的权重
        qkv_weights = {}  # QKV权重字典
        gate_up_weights = {}  # gate_up权重字典

        for name, loaded_weight in weights:  # 遍历所有权重
            # Handle weight name mappings
            # 处理权重名称映射

            # Map visual attention weights: qkv -> qkv_proj
            # 映射视觉注意力权重：qkv -> qkv_proj
            if ".attn.qkv." in name:  # 如果是视觉注意力QKV权重
                mapped_name = name.replace(".attn.qkv.", ".attn.qkv_proj.")  # 替换名称
                if mapped_name in params_dict:  # 如果映射后的名称存在于参数字典
                    param = params_dict[mapped_name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                    loaded_params.add(mapped_name)  # 添加到已加载集合
                    continue  # 继续下一个权重

            # Handle Llama attention weights - need to fuse q, k, v into qkv
            # 处理Llama注意力权重 - 需要将q、k、v融合为qkv
            if ".self_attn.q_proj.weight" in name:  # 如果是查询投影权重
                base = name.replace(".q_proj.weight", "")  # 获取基础名称
                qkv_weights[base] = qkv_weights.get(base, {})  # 获取或创建字典
                qkv_weights[base]["q"] = loaded_weight  # 存储查询权重
                continue  # 继续下一个权重
            elif ".self_attn.k_proj.weight" in name:  # 如果是键投影权重
                base = name.replace(".k_proj.weight", "")  # 获取基础名称
                qkv_weights[base] = qkv_weights.get(base, {})  # 获取或创建字典
                qkv_weights[base]["k"] = loaded_weight  # 存储键权重
                continue  # 继续下一个权重
            elif ".self_attn.v_proj.weight" in name:  # 如果是值投影权重
                base = name.replace(".v_proj.weight", "")  # 获取基础名称
                qkv_weights[base] = qkv_weights.get(base, {})  # 获取或创建字典
                qkv_weights[base]["v"] = loaded_weight  # 存储值权重
                continue  # 继续下一个权重

            # Handle Llama MLP weights - need to fuse gate and up projections
            # 处理Llama MLP权重 - 需要融合gate和up投影
            if ".mlp.gate_proj.weight" in name:  # 如果是门控投影权重
                base = name.replace(".gate_proj.weight", "")  # 获取基础名称
                gate_up_weights[base] = gate_up_weights.get(base, {})  # 获取或创建字典
                gate_up_weights[base]["gate"] = loaded_weight  # 存储门控权重
                continue  # 继续下一个权重
            elif ".mlp.up_proj.weight" in name:  # 如果是上投影权重
                base = name.replace(".up_proj.weight", "")  # 获取基础名称
                gate_up_weights[base] = gate_up_weights.get(base, {})  # 获取或创建字典
                gate_up_weights[base]["up"] = loaded_weight  # 存储上投影权重
                continue  # 继续下一个权重

            # Direct mapping for other weights
            # 其他权重的直接映射
            if name in params_dict:  # 如果名称存在于参数字典
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重
                loaded_params.add(name)  # 添加到已加载集合

        # Fuse QKV weights for Llama attention layers
        # 融合Llama注意力层的QKV权重
        for base, weights_dict in qkv_weights.items():  # 遍历QKV权重
            if "q" in weights_dict and "k" in weights_dict and "v" in weights_dict:  # 如果q、k、v都存在
                qkv_name = f"{base}.qkv_proj.weight"  # 构造融合后的名称
                if qkv_name in params_dict:  # 如果融合名称存在于参数字典
                    # Concatenate q, k, v weights
                    # 拼接q、k、v权重
                    q, k, v = weights_dict["q"], weights_dict["k"], weights_dict["v"]  # 获取q、k、v
                    qkv = torch.cat([q, k, v], dim=0)  # 沿第0维拼接
                    param = params_dict[qkv_name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(param, qkv)  # 加载融合权重
                    loaded_params.add(qkv_name)  # 添加到已加载集合

        # Fuse gate and up weights for Llama MLP layers
        # 融合Llama MLP层的gate和up权重
        for base, weights_dict in gate_up_weights.items():  # 遍历gate_up权重
            if "gate" in weights_dict and "up" in weights_dict:  # 如果gate和up都存在
                gate_up_name = f"{base}.gate_up_proj.weight"  # 构造融合后的名称
                if gate_up_name in params_dict:  # 如果融合名称存在于参数字典
                    # Concatenate gate and up weights
                    # 拼接gate和up权重
                    gate, up = weights_dict["gate"], weights_dict["up"]  # 获取gate和up
                    gate_up = torch.cat([gate, up], dim=0)  # 沿第0维拼接
                    param = params_dict[gate_up_name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(param, gate_up)  # 加载融合权重
                    loaded_params.add(gate_up_name)  # 添加到已加载集合


# Register the model
# 注册模型
EntryClass = Sarashina2VisionForCausalLM  # 注册入口类为Sarashina2VisionForCausalLM
