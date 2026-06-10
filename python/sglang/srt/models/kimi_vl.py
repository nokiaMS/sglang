# Kimi VL 视觉语言模型实现
# 该文件实现了 Kimi VL 条件生成模型，将 MoonViT 视觉编码器与
# DeepseekV2 语言模型结合，支持图像多模态输入，
# 通过 KimiVLMultiModalProjector 将视觉特征投影到语言模型空间。
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy  # 导入拷贝模块
import logging  # 导入日志模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Iterable, List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers.activations import GELUActivation  # 导入GELU激活函数

from sglang.srt.configs import KimiVLConfig  # 导入KimiVL配置
from sglang.srt.configs.deepseekvl2 import DeepseekV2Config  # 导入DeepseekV2配置
from sglang.srt.configs.kimi_vl import KimiVLConfig  # 导入KimiVL配置
from sglang.srt.configs.kimi_vl_moonvit import MoonViTConfig  # 导入MoonViT配置
from sglang.srt.layers.activation import QuickGELU  # 导入快速GELU激活
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # KV缩放名称重映射
)
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM  # 导入DeepseekV2语言模型
from sglang.srt.models.kimi_vl_moonvit import MoonVitPretrainedModel  # 导入MoonViT预训练模型
from sglang.srt.utils import add_prefix  # 导入前缀工具

logger = logging.getLogger(__name__)  # 获取日志记录器


# For dummy input only  # 仅用于虚拟输入
@dataclass  # 数据类装饰器
class MaxImageTokenMeta:
    """最大图像token元数据，用于虚拟输入。"""
    width: int = 1024  # 宽度
    height: int = 1024  # 高度


class KimiVLMultiModalProjector(nn.Module):
    """Kimi VL多模态投影器，将视觉特征投影到语言模型的隐藏空间。"""

    def __init__(self, config: KimiVLConfig):  # 初始化
        super().__init__()  # 调用父类初始化

        self.hidden_size = (  # 计算投影器隐藏大小
            config.vision_config.hidden_size  # 视觉隐藏大小
            * config.vision_config.merge_kernel_size[0]  # 合并核高度
            * config.vision_config.merge_kernel_size[1]  # 合并核宽度
        )

        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-5)  # 预归一化
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # 第一层线性变换
        self.act = GELUActivation()  # GELU激活（会被覆盖）
        self.act = QuickGELU()  # 使用快速GELU替换
        self.linear_2 = nn.Linear(  # 第二层线性变换
            self.hidden_size, config.text_config.hidden_size, bias=True  # 输出到文本隐藏大小
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """前向传播：将视觉特征投影到语言模型空间。"""
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)  # 预归一化并重塑
        hidden_states = self.linear_1(hidden_states)  # 第一层线性
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states = self.linear_2(hidden_states)  # 第二层线性
        return hidden_states  # 返回投影结果


class KimiVLForConditionalGeneration(nn.Module):
    """Kimi VL条件生成模型，结合MoonViT视觉编码器和DeepseekV2语言模型。"""
    def __init__(
        self,
        config: KimiVLConfig,  # KimiVL配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        **kwargs,  # fix init_tts argument error  # 修复init_tts参数错误
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        assert isinstance(config.vision_config, MoonViTConfig)  # 确认视觉配置类型

        self.vision_tower = MoonVitPretrainedModel(config.vision_config)  # 创建MoonViT视觉编码器

        self.multi_modal_projector = KimiVLMultiModalProjector(config=config)  # 创建多模态投影器
        self.quant_config = quant_config  # 保存量化配置

        self.language_model = None  # 语言模型初始为None
        if not config.encoder_only:  # 如果不是仅编码器模式
            text_config = copy.deepcopy(config.text_config)  # 深拷贝文本配置
            text_config.architectures = ["DeepseekV2ForCausalLM"]  # 设置架构
            self.language_model = DeepseekV2ForCausalLM(  # 创建DeepseekV2语言模型
                config=text_config,  # 文本配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("language_model", prefix),  # 参数前缀
            )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """提取图像特征：通过视觉编码器和投影器。"""
        pixel_values = (  # 拼接像素值
            torch.cat([item.feature for item in items], dim=0)  # 拼接所有图像特征
            .type(self.vision_tower.dtype)  # 转为视觉塔数据类型
            .to(self.vision_tower.device)  # 转到视觉塔设备
        )

        if (  # 如果像素值已经是投影后的特征
            pixel_values.dim() == 2  # 2维张量
            and pixel_values.shape[-1] == self.config.text_config.hidden_size  # 维度匹配文本隐藏大小
        ):
            return pixel_values  # 直接返回

        image_grid_hws = torch.cat([item.image_grid_hws for item in items], dim=0).to(  # 拼接网格高度宽度
            self.vision_tower.device  # 转到设备
        )
        image_features = self.vision_tower(pixel_values, image_grid_hws)  # 通过视觉编码器
        assert isinstance(image_features, list)  # 确认输出是列表
        # lengths = [x.shape[0] for x in image_features]  # 获取每个图像的token数
        res = self.multi_modal_projector(torch.cat(image_features))  # .split(lengths)  # 通过投影器
        return res  # 返回投影结果

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """填充输入token ID，替换多模态标记占位符。"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 执行填充

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入
    ):
        """前向传播：执行多模态条件生成。"""
        hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数
                Modality.IMAGE: self.get_image_feature,  # 图像特征提取
            },
            positions=positions,  # 位置编码
        )

        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，支持堆叠参数、专家参数和视觉/语言分离加载。"""
        config = self.config.text_config  # 获取文本配置
        _KEYS_TO_MODIFY_MAPPING = {  # 键名映射
            # "language_model.lm_head": "lm_head",  # 语言模型头映射
            # "language_model.model": "language_model",  # 语言模型映射
        }
        # only doing this for language model part for now.  # 目前仅对语言模型部分执行
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".gate_up_proj", ".gate_proj", 0),  # gate_up中的gate
            (".gate_up_proj", ".up_proj", 1),  # gate_up中的up
        ]
        if not config.use_mla:  # 如果不使用MLA
            stacked_params_mapping += [  # 添加QKV堆叠映射
                (".qkv_proj", ".q_proj", "q"),  # QKV中的Q
                (".qkv_proj", ".k_proj", "k"),  # QKV中的K
                (".qkv_proj", ".v_proj", "v"),  # QKV中的V
            ]
        if getattr(config, "n_routed_experts", None):  # 如果有路由专家
            # Params for weights, fp8 weight scales, fp8 activation scales  # 权重、FP8权重缩放、FP8激活缩放参数
            # (param_name, weight_name, expert_id, shard_id)  # (参数名, 权重名, 专家ID, 分片ID)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
                ckpt_gate_proj_name="gate_proj",  # 检查点gate名
                ckpt_down_proj_name="down_proj",  # 检查点down名
                ckpt_up_proj_name="up_proj",  # 检查点up名
                num_experts=config.n_routed_experts,  # 路由专家数
            )
        else:  # 无路由专家
            expert_params_mapping = []  # 空映射

        params_dict = dict(self.named_parameters())  # 参数字典
        for args in weights:  # 遍历权重
            name, loaded_weight = args[:2]  # 获取名称和权重
            kwargs = args[2] if len(args) > 2 else {}  # 获取额外参数

            is_vision_weight = ("vision" in name) or ("multi_modal_projector" in name)  # 判断是否为视觉权重
            if self.config.encoder_only and not is_vision_weight:  # 仅编码器模式跳过非视觉权重
                continue
            if self.config.language_only and is_vision_weight:  # 仅语言模式跳过视觉权重
                continue

            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入频率
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(config, name)  # 获取投机解码层索引
            if spec_layer is not None:  # 如果是投机解码层
                continue  # skip spec decode layers for main model  # 跳过主模型的投机解码层

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的cos/sin
                # Models trained using ColossalAI may include these tensors in  # ColossalAI训练的模型可能包含
                # the checkpoint. Skip them.  # 跳过
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():  # 遍历键名映射
                if key_to_modify in name:  # 如果需要修改
                    name = name.replace(key_to_modify, new_key)  # 替换键名
            use_default_weight_loading = False  # 是否使用默认加载
            if "vision" in name:  # 如果是视觉权重
                if self.vision_tower is not None:  # 如果视觉塔存在
                    # We only do sharding for language model and  # 仅对语言模型做分片
                    # not vision model for now.  # 视觉模型暂不分片
                    use_default_weight_loading = True  # 使用默认加载
            else:  # 语言模型权重
                for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                    if weight_name not in name:  # 如果权重名不匹配
                        continue  # 跳过
                    # We have mlp.experts[0].gate_proj in the checkpoint.  # 检查点中有专家gate_proj
                    # Since we handle the experts below in expert_params_mapping,  # 专家在expert_params_mapping中处理
                    # we need to skip here BEFORE we update the name, otherwise  # 需要在更新名称前跳过
                    # name will be updated to mlp.experts[0].gate_up_proj, which  # 否则名称会更新为gate_up_proj
                    # will then be updated below in expert_params_mapping  # 然后在expert_params_mapping中再次更新
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.  # 导致加载失败
                    if ("mlp.experts." in name) and name not in params_dict:  # 专家权重不在参数字典中
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 偏置不在参数字典中
                        continue  # 跳过
                    if name not in params_dict:  # 参数不存在
                        continue  # 跳过

                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(param, loaded_weight, shard_id, **kwargs)  # 加载权重
                    break  # 跳出内层循环
                else:  # 没有匹配到堆叠参数
                    for idx, (  # 遍历专家参数映射
                        param_name,
                        weight_name,
                        expert_id,
                        shard_id,
                    ) in enumerate(expert_params_mapping):
                        if weight_name not in name:  # 如果权重名不匹配
                            continue  # 跳过
                        name = name.replace(weight_name, param_name)  # 替换权重名
                        if name not in params_dict:  # 参数不存在
                            continue  # 跳过

                        param = params_dict[name]  # 获取参数
                        weight_loader = param.weight_loader  # 获取权重加载器
                        weight_loader(  # 加载权重
                            param,  # 参数
                            loaded_weight,  # 权重数据
                            name,  # 参数名
                            expert_id=expert_id,  # 专家ID
                            shard_id=shard_id,  # 分片ID
                            **kwargs,  # 额外参数
                        )
                        break  # 跳出内层循环
                    else:  # 也没有匹配到专家参数
                        use_default_weight_loading = True  # 使用默认加载
            if use_default_weight_loading:  # 如果使用默认加载
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 偏置不存在
                    continue  # 跳过
                # Remapping the name of FP8 kv-scale.  # 重映射FP8 KV缩放名称
                name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射
                if name is None:  # 如果重映射后为None
                    continue  # 跳过

                # if is_pp_missing_parameter(name, self):  # PP缺失参数检查
                #     continue  # 跳过

                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight, **kwargs)  # 加载权重
        if self.language_model is not None:  # 如果语言模型存在
            self.language_model.post_load_weights()  # 后处理权重


def get_spec_layer_idx_from_weight_name(
    config: DeepseekV2Config, weight_name: str
) -> Optional[int]:
    """从权重名获取投机解码层的索引。"""
    if hasattr(config, "num_nextn_predict_layers") and (  # 如果有预测层配置
        config.num_nextn_predict_layers > 0  # 且层数大于0
    ):
        layer_idx = config.num_hidden_layers  # 起始层索引
        for i in range(config.num_nextn_predict_layers):  # 遍历预测层
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):  # 如果权重属于预测层
                return layer_idx + i  # 返回层索引
    return None  # 不属于预测层


EntryClass = [KimiVLForConditionalGeneration]  # 入口类列表
