# Mistral 模型实现
# 本文件实现了Mistral模型，包括标准MistralForCausalLM（基于Llama架构）、
# 支持Mistral原生格式权重加载的MistralForCausalLMMistralFormat、
# 以及支持多模态的Mistral3ForConditionalGeneration。
# 主要处理权重名称从Mistral原生格式到HuggingFace/Llama格式的重映射。

# Copyright 2023-2024 SGLang Team  # 版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache 2.0许可证授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 许可证下的软件按"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限和
# limitations under the License.  # 限制的具体语言
# ==============================================================================  # 分隔线
"""Inference-only Mistral model."""  # 仅推理的Mistral模型

import logging  # 导入日志模块
from collections.abc import Iterable  # 导入可迭代类型
from typing import List  # 导入列表类型

import regex as re  # 导入正则表达式库
import torch  # 导入PyTorch
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector  # 导入Mistral3多模态投影器

from sglang.srt.managers.schedule_batch import MultimodalDataItem  # 导入多模态数据项
from sglang.srt.models.llama import LlamaForCausalLM  # 导入Llama因果语言模型

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class MistralForCausalLM(LlamaForCausalLM):  # Mistral因果语言模型，直接继承LlamaForCausalLM
    pass  # 无额外实现，完全复用Llama架构


class MistralForCausalLMMistralFormat(MistralForCausalLM):  # 支持Mistral原生格式权重加载的Mistral模型
    """Mistral GQA model loaded from mistral native format (params.json).  # 从Mistral原生格式（params.json）加载的Mistral GQA模型

    Handles weight name remapping from mistral native format to HF/Llama  # 处理从Mistral原生格式到HF/Llama格式的权重名称重映射
    format. This is the GQA counterpart to MistralLarge3ForCausalLM which  # 这是GQA版本，对应于处理MLA模型的
    handles MLA models in mistral native format.  # MistralLarge3ForCausalLM（处理Mistral原生格式的MLA模型）
    """

    # fmt: off  # 关闭格式化
    remapping = {  # Mistral原生格式到HF/Llama格式的权重名称映射表
        r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",  # 注意力层归一化权重映射
        r"layers\.(\d+)\.attention\.wq\.(\w+)": r"model.layers.\1.self_attn.q_proj.\2",  # 查询投影权重映射
        r"layers\.(\d+)\.attention\.wk\.(\w+)": r"model.layers.\1.self_attn.k_proj.\2",  # 键投影权重映射
        r"layers\.(\d+)\.attention\.wv\.(\w+)": r"model.layers.\1.self_attn.v_proj.\2",  # 值投影权重映射
        r"layers\.(\d+)\.attention\.wo\.(\w+)": r"model.layers.\1.self_attn.o_proj.\2",  # 输出投影权重映射
        r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",  # FFN层归一化权重映射
        r"layers\.(\d+)\.feed_forward\.w1\.(\w+)": r"model.layers.\1.mlp.gate_proj.\2",  # 前馈网络门控投影权重映射
        r"layers\.(\d+)\.feed_forward\.w2\.(\w+)": r"model.layers.\1.mlp.down_proj.\2",  # 前馈网络下投影权重映射
        r"layers\.(\d+)\.feed_forward\.w3\.(\w+)": r"model.layers.\1.mlp.up_proj.\2",  # 前馈网络上投影权重映射
        r"norm\.weight": "model.norm.weight",  # 最终归一化权重映射
        r"tok_embeddings\.weight": "model.embed_tokens.weight",  # 词嵌入权重映射
        r"output\.weight": "lm_head.weight",  # 输出头权重映射
    }
    # fmt: on  # 恢复格式化

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):  # 加载模型权重，先重映射再调用父类加载
        return super().load_weights(self._remap_mistral_to_llama(weights))  # 先将Mistral格式重映射为Llama格式

    def _remap_mistral_to_llama(  # 将Mistral原生格式权重名称重映射为HF/Llama格式
        self, weights: Iterable[tuple[str, torch.Tensor]]  # 权重迭代器
    ) -> Iterable[tuple[str, torch.Tensor]]:  # 返回重映射后的权重迭代器
        """Remap Mistral native format weight names to HF/Llama format."""  # 将Mistral原生格式权重名称重映射为HF/Llama格式
        for name, loaded_weight in weights:  # 遍历所有权重
            # Pass through weights already in HF/Llama layout so this loader  # 传递已经是HF/Llama格式的权重，使此加载器
            # tolerates mixed-format checkpoints (e.g. native body + HF-style  # 能够容忍混合格式的检查点（如原生主体 + HF风格的
            # multi_modal_projector weights spliced in by a parent class).  # 多模态投影器权重由父类拼接）
            if name.startswith("model.") or name.startswith("lm_head."):  # 如果名称已是HF/Llama格式
                yield name, loaded_weight  # 直接产出
                continue  # 跳过后续处理

            for k, v in self.remapping.items():  # 遍历映射规则
                match = re.fullmatch(k, name)  # 尝试完整匹配权重名称
                if match:  # 如果匹配成功
                    name = match.expand(v)  # 展开为映射后的名称
                    break  # 跳出映射规则循环
            else:  # 如果没有任何映射规则匹配
                logger.warning(f"Unrecognized weight: {name}. Skipping.")  # 记录无法识别的权重并跳过
                continue  # 跳过此权重

            if name.endswith(".qscale_act"):  # 如果名称以.qscale_act结尾（激活缩放因子）
                name = re.sub(r"\.qscale_act$", ".input_scale", name)  # 替换为.input_scale
            elif name.endswith(".qscale_weight"):  # 如果名称以.qscale_weight结尾（权重缩放因子）
                name = re.sub(r"\.qscale_weight$", ".weight_scale", name)  # 替换为.weight_scale

            yield name, loaded_weight  # 产出重映射后的权重名称和张量


class Mistral3ForConditionalGeneration:  # Mistral 3多模态条件生成模型
    MULTIMODAL_PROJECTOR_TYPE = Mistral3MultiModalProjector  # 多模态投影器类型

    def __init__(self, **kwargs):  # 初始化方法
        # lazy load inner class  # 延迟加载内部类
        # to bypass circular import  # 以避免循环导入
        from sglang.srt.models.llava import LlavaForConditionalGeneration  # 导入Llava条件生成模型

        # override config: mistral's projector adds patchmerger that doesn't require padding  # 覆盖配置：Mistral的投影器添加了不需要填充的patchmerger
        kwargs["config"].vision_config.pad_image_border = False  # 禁用图像边框填充

        self.inner = LlavaForConditionalGeneration(**kwargs)  # 创建Llava条件生成模型实例
        self.inner.multi_modal_projector = self.MULTIMODAL_PROJECTOR_TYPE(  # 使用Mistral3多模态投影器替换默认投影器
            kwargs["config"]  # 传递配置
        )
        self.inner.get_image_feature = self.get_image_feature  # 使用自定义的图像特征提取方法

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 从图像输入中提取特征
        """Extract features from image inputs.  # 从图像输入中提取特征

        Args:  # 参数说明
            items: List of MultimodalDataItem objects containing image data  # 包含图像数据的多模态数据项列表
                Note that an item can be either "image" or "multi-images"  # 注意一个项目可以是"image"或"multi-images"

        Returns:  # 返回值说明
            torch.Tensor: features from image inputs, concatenated  # 图像输入的特征，已拼接
        """
        features = []  # 特征列表
        for item in items:  # 遍历所有多模态数据项
            # in each item, we assume pixel_values is always batched  # 在每个项目中，假设pixel_values总是批处理的
            pixel_values, image_sizes = item.feature, item.image_sizes  # 获取像素值和图像大小
            image_outputs = self.vision_tower(  # 通过视觉塔获取图像输出
                pixel_values, image_sizes, output_hidden_states=True  # 传递像素值和图像大小，输出隐藏状态
            )
            selected_image_feature = image_outputs.hidden_states[  # 选择指定层的隐藏状态作为图像特征
                self.vision_feature_layer  # 视觉特征层索引
            ]

            if self.vision_feature_select_strategy in ["default", "patch"]:  # 如果特征选择策略为default或patch
                selected_image_feature = selected_image_feature[:, 1:]  # 去除CLS token
            elif self.vision_feature_select_strategy == "full":  # 如果特征选择策略为full
                selected_image_feature = selected_image_feature  # 保留全部特征
            else:  # 其他策略
                raise ValueError(  # 抛出值错误
                    f"Unexpected select feature: {self.vision_feature_select_strategy}"  # 未预期的特征选择策略
                )
            features.append(  # 将处理后的特征添加到列表
                self.multi_modal_projector(  # 通过多模态投影器处理
                    selected_image_feature.squeeze(0), image_sizes  # 去除批次维度并传递图像大小
                )
            )
        ret = torch.cat(features, dim=0)  # 在第0维度上拼接所有特征
        return ret  # 返回拼接后的特征

    def __getattr__(self, name):  # 属性访问代理，转发到内部模型
        return getattr(self.inner, name)  # 获取内部模型的属性

    def __hasattr__(self, name):  # 属性存在判断代理
        return hasattr(self.inner, name)  # 判断内部模型是否具有该属性

    def __call__(self, *args, **kwargs):  # 调用代理
        return self.inner(*args, **kwargs)  # 调用内部模型

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):  # 加载模型权重，处理transformers v5的权重名称格式
        """Normalize transformers v5 Mistral3 weight names for  # 将transformers v5的Mistral3权重名称规范化
        LlavaForConditionalGeneration.load_weights.  # 以适配LlavaForConditionalGeneration.load_weights

        v5 checkpoints lay out Mistral3 weights as:  # v5检查点中Mistral3权重的布局为：
          model.language_model.{embed_tokens,layers.*,norm}.*  # 语言模型权重
          model.vision_tower.*  # 视觉塔权重
          model.multi_modal_projector.*  # 多模态投影器权重
          lm_head.*  # 输出头权重

        The Llava loader routes by top-level `language_model.` /  # Llava加载器通过顶层`language_model.`/
        `vision_tower.` prefixes, stripping one segment before forwarding to  # `vision_tower.`前缀路由，在转发给
        the sub-module.  The sub-module's own `load_weights` expects the  # 子模块之前剥离一段。子模块自己的`load_weights`期望
        standard HF layout: `model.layers.*`, `model.embed_tokens.weight`,  # 标准HF布局：`model.layers.*`、`model.embed_tokens.weight`、
        `lm_head.weight` for Llama, and `vision_tower` internals at their  # Llama的`lm_head.weight`，以及`vision_tower`内部在
        top level.  So we rewrite:  # 顶层。所以我们重写：
          model.language_model.X   -> language_model.model.X  # 语言模型重映射
          model.vision_tower.X     -> vision_tower.X  # 视觉塔重映射
          model.multi_modal_projector.X -> multi_modal_projector.X  # 多模态投影器重映射
          lm_head.X                -> language_model.lm_head.X  # 输出头重映射
        """

        def normalize(ws):  # 权重名称规范化函数
            for name, w in ws:  # 遍历所有权重
                if name.startswith("model.language_model."):  # 如果名称以model.language_model.开头
                    rest = name[len("model.language_model.") :]  # 获取剩余部分
                    name = "language_model.model." + rest  # 重映射为language_model.model.前缀
                elif name.startswith("model.vision_tower."):  # 如果名称以model.vision_tower.开头
                    name = "vision_tower." + name[len("model.vision_tower.") :]  # 重映射为vision_tower.前缀
                elif name.startswith("model.multi_modal_projector."):  # 如果名称以model.multi_modal_projector.开头
                    name = (  # 重映射为multi_modal_projector.前缀
                        "multi_modal_projector."  # 多模态投影器前缀
                        + name[len("model.multi_modal_projector.") :]  # 加上剩余部分
                    )
                elif name.startswith("lm_head."):  # 如果名称以lm_head.开头
                    name = "language_model." + name  # 重映射为language_model.lm_head.前缀
                yield name, w  # 产出规范化后的权重名称和张量

        return self.inner.load_weights(normalize(weights))  # 使用规范化后的权重调用内部模型加载


EntryClass = [MistralForCausalLM, Mistral3ForConditionalGeneration]  # 模型注册入口类列表
