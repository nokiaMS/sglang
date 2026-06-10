# Qwen2-Audio 音频语言模型实现
# 本文件实现了 Qwen2-Audio 多模态音频语言模型，结合了音频编码器、
# 多模态投影器和 Qwen2 语言模型，支持音频输入的语音对话场景。
# coding=utf-8  # 编码声明
# Adapted from  # 适配自
# https://github.com/huggingface/transformers/blob/1d45d90e5d1552eccb6d8cc9b7bba283ccefb808/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py  # HuggingFace Transformers 中的 Qwen2-Audio 实现
# Copyright 2024 The Qwen team.  # Qwen 团队版权
# Copyright 2023 The vLLM team.  # vLLM 团队版权
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.  # EleutherAI 和 HuggingFace 版权
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX  # 基于 GPT-NeoX 库
# and OPT implementations in this library. It has been modified from its  # 和 OPT 实现，已修改
# original forms to accommodate minor architectural differences compared  # 以适应架构差异
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.  # 与 Meta AI 团队使用的模型
#
# Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 不得违反许可证使用
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何担保
# See the License for the specific language governing permissions and  # 查看许可证获取权限
# limitations under the License.  # 许可证限制
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""  # 仅推理的 Qwen2-Audio 模型

import logging  # 导入日志模块
from typing import Any, Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
import torch.nn as nn  # 导入神经网络模块
from transformers import Qwen2AudioEncoderConfig, Qwen2Config  # 导入 Qwen2 音频编码器配置和 Qwen2 配置
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig  # 导入 Qwen2-Audio 配置
from transformers.models.qwen2_audio.modeling_qwen2_audio import (  # 导入 Qwen2-Audio 原始实现
    Qwen2AudioEncoder,  # 音频编码器
    Qwen2AudioMultiModalProjector,  # 多模态投影器
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态类型
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入 Qwen2 因果语言模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取日志记录器


class Qwen2AudioForConditionalGeneration(nn.Module):
    """Qwen2-Audio 条件生成模型，整合音频编码器、投影器和语言模型"""

    # BitandBytes specific attributes  # BitandBytes 特定属性
    default_bitsandbytes_target_modules = [  # 默认 BitandBytes 目标模块
        ".gate_proj.",  # gate 投影
        ".down_proj.",  # 下投影
        ".up_proj.",  # 上投影
        ".q_proj.",  # Q 投影
        ".k_proj.",  # K 投影
        ".v_proj.",  # V 投影
        ".o_proj.",  # O 投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes 堆叠参数映射
        # shard_name, weight_name, index  # 分片名, 权重名, 索引
        "q_proj": ("qkv_proj", 0),  # Q 映射
        "k_proj": ("qkv_proj", 1),  # K 映射
        "v_proj": ("qkv_proj", 2),  # V 映射
        "gate_proj": ("gate_up_proj", 0),  # gate 映射
        "up_proj": ("gate_up_proj", 1),  # up 映射
    }

    def __init__(
        self,
        config: Qwen2AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen2-Audio 条件生成模型"""
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        if getattr(self.config, "audio_config", None) is None:  # 如果没有音频配置
            self.config.audio_config = Qwen2AudioEncoderConfig(  # 创建默认音频编码器配置
                self.config._name_or_path  # 使用模型路径
            )

        if getattr(self.config, "text_config", None) is None:  # 如果没有文本配置
            self.config.text_config = Qwen2Config(self.config._name_or_path)  # 创建默认文本配置

        self.audio_tower = Qwen2AudioEncoder(  # 创建音频编码器
            config.audio_config,  # 音频配置
        )
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config)  # 创建多模态投影器
        self.language_model = Qwen2ForCausalLM(  # 创建 Qwen2 语言模型
            config.text_config, quant_config, prefix=add_prefix("model", prefix)  # 文本配置、量化配置和前缀
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态填充模式

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """使用多模态 token 填充模式对输入 ID 进行填充"""
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)  # 返回填充后的 token

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """从多模态数据项中提取音频特征"""
        # Extract audio features from input items  # 从输入项中提取音频特征
        input_features = torch.cat([item.feature for item in items], dim=0).type(  # 拼接音频特征并转换类型
            self.audio_tower.dtype  # 音频编码器数据类型
        )

        audio_embeds = self.audio_tower(input_features).last_hidden_state  # 通过音频编码器获取隐藏状态
        audio_embeds = self.multi_modal_projector(audio_embeds)  # 通过多模态投影器

        audio_feature_lens = torch.cat([item.audio_feature_lens for item in items])  # 拼接音频特征长度
        new_embeds = []  # 新嵌入列表
        for i, d in zip(audio_feature_lens, audio_embeds):  # 遍历长度和嵌入
            new_embeds.append(d[: i.item()])  # 按实际长度截取

        return torch.cat(new_embeds, dim=0)  # 拼接并返回

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Qwen2-Audio 前向传播：处理音频输入并生成隐藏状态"""
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入 ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.AUDIO: self.get_audio_feature,  # 音频模态对应的特征提取函数
            },
            positions=positions,  # 位置信息
        )

        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理堆叠参数映射和共享词嵌入"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q 映射
            ("qkv_proj", "k_proj", "k"),  # K 映射
            ("qkv_proj", "v_proj", "v"),  # V 映射
            ("gate_up_proj", "gate_proj", 0),  # gate 映射
            ("gate_up_proj", "up_proj", 1),  # up 映射
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入逆频率
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的余弦/正弦
                # Models trained using ColossalAI may include these tensors in  # ColossalAI 训练的模型可能包含这些张量
                # the checkpoint. Skip them.  # 跳过它们
                continue

            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:  # 如果共享词嵌入
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name or "audio_tower" in name:  # 跳过非匹配或音频塔参数
                    continue
                name_tmp = name.replace(weight_name, param_name)  # 替换为堆叠参数名

                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:  # 如果偏置不在参数字典中
                    continue
                param = params_dict[name_tmp]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数处理
                try:  # 尝试加载参数
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                        continue
                    param = params_dict[name]  # 获取参数
                except KeyError:  # 参数未找到
                    print(params_dict.keys())  # 打印可用参数名
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = Qwen2AudioForConditionalGeneration  # 模型入口类
