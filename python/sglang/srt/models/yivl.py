# Yi-VL视觉语言模型实现文件
# 本文件实现了Yi-VL模型，基于Llava架构，使用CLIP视觉编码器和自定义多模态投影器
# Yi-VL通过YiVLMultiModalProjector将视觉特征映射到语言模型的嵌入空间

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
"""Inference-only Yi-VL model."""

from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn as nn  # 导入PyTorch神经网络模块
from transformers import CLIPVisionModel, LlavaConfig  # 导入CLIP视觉模型和Llava配置

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.llava import LlavaLlamaForCausalLM  # 导入Llava Llama因果语言模型


class YiVLForCausalLM(LlavaLlamaForCausalLM):  # Yi-VL因果语言模型类，继承自LlavaLlamaForCausalLM
    def __init__(  # 初始化方法
        self,
        config: LlavaConfig,  # Llava配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__(config, quant_config, prefix=prefix)  # 调用父类初始化

        self.multi_modal_projector = YiVLMultiModalProjector(self.config)  # 创建Yi-VL多模态投影器
        self.vision_tower_subfolder = self.config.mm_vision_tower.replace(  # 获取视觉塔子文件夹路径
            "./", ""
        )  # 替换掉"./"前缀

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        # We have to use the subfolder of the main model directory (e.g. 01-ai/Yi-VL-6B)
        # 我们必须使用主模型目录的子文件夹
        self.vision_tower = CLIPVisionModel.from_pretrained(  # 从预训练加载CLIP视觉模型
            self.config._name_or_path,  # 模型路径
            torch_dtype=torch.float16,  # 使用float16精度
            subfolder=self.vision_tower_subfolder,  # 子文件夹路径
        ).to("cuda")  # 移动到GPU

        self.vision_tower.eval()  # 设置视觉塔为评估模式

        self.vision_feature_layer = self.config.mm_vision_select_layer  # 视觉特征选择层
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature  # 视觉特征选择策略
        self.image_size = self.vision_tower.config.image_size  # 图像尺寸
        self.patch_size = self.vision_tower.config.patch_size  # 补丁尺寸

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")  # 补丁合并类型
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")  # 图像宽高比
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)  # 图像网格锚点

        self.image_feature_len = int((self.image_size / self.patch_size) ** 2)  # 图像特征长度
        if self.vision_feature_select_strategy == "patch":  # 如果选择patch策略
            pass  # 不做额外处理
        elif self.vision_feature_select_strategy == "cls_patch":  # 如果选择cls_patch策略
            self.image_feature_len += 1  # 增加一个CLS token的长度
        else:  # 其他策略
            raise ValueError(f"Unexpected select feature: {self.select_feature}")  # 抛出异常

        # load mm_projector
        # 加载多模态投影器
        # TODO: support TP?
        # 待办：支持张量并行？
        projector_weights = {  # 投影器权重映射字典
            "model.mm_projector.0": "multi_modal_projector.linear_1",  # 第一个线性层
            "model.mm_projector.1": "multi_modal_projector.ln_1",  # 第一个LayerNorm
            "model.mm_projector.3": "multi_modal_projector.linear_2",  # 第二个线性层
            "model.mm_projector.4": "multi_modal_projector.ln_2",  # 第二个LayerNorm
            "model.vision_tower.vision_tower": "vision_tower",  # 视觉塔权重
            # transformers 5.6.0 flattened CLIPVisionModel/SiglipVisionModel,
            # dropping the `vision_model` intermediate wrapper.
            # transformers 5.6.0扁平化了CLIPVisionModel/SiglipVisionModel，去掉了vision_model中间包装器
            "vision_tower.vision_model.": "vision_tower.",  # 视觉模型前缀映射
        }
        params_dict = dict(self.named_parameters())  # 获取模型参数字典
        weights = list(weights)  # 将权重转换为列表
        for name, loaded_weight in weights:  # 遍历所有权重
            if "projector" in name or "vision_tower" in name:  # 如果是投影器或视觉塔权重
                for weight_name, param_name in projector_weights.items():  # 遍历权重映射
                    if weight_name in name:  # 如果权重名匹配
                        name = name.replace(weight_name, param_name)  # 替换权重名
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重

        # load language model
        # 加载语言模型
        self.language_model.load_weights(weights)  # 调用语言模型的权重加载方法


class YiVLMultiModalProjector(nn.Module):  # Yi-VL多模态投影器类
    def __init__(self, config: LlavaConfig):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self.linear_1 = nn.Linear(  # 第一个线性层
            config.vision_config.hidden_size, config.text_config.hidden_size  # 从视觉维度映射到文本维度
        )
        self.ln_1 = nn.LayerNorm(config.text_config.hidden_size)  # 第一个LayerNorm层
        self.act = nn.GELU()  # GELU激活函数
        self.linear_2 = nn.Linear(  # 第二个线性层
            config.text_config.hidden_size, config.text_config.hidden_size  # 文本维度到文本维度
        )
        self.ln_2 = nn.LayerNorm(config.text_config.hidden_size)  # 第二个LayerNorm层

    def forward(self, image_features):  # 前向传播方法
        hidden_states = self.linear_1(image_features)  # 通过第一个线性层
        hidden_states = self.ln_1(hidden_states)  # 通过第一个LayerNorm
        hidden_states = self.act(hidden_states)  # 通过GELU激活函数
        hidden_states = self.linear_2(hidden_states)  # 通过第二个线性层
        hidden_states = self.ln_2(hidden_states)  # 通过第二个LayerNorm
        return hidden_states  # 返回处理后的隐藏状态


EntryClass = YiVLForCausalLM  # 注册入口类为YiVLForCausalLM
