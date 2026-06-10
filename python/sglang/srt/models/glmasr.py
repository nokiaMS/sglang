# GLM-ASR 语音识别模型文件
# 本文件实现了仅推理模式的 GLM-ASR-HF 语音识别模型，
# 结合音频编码器、多模态投影器和 Llama 语言模型，
# 兼容 HuggingFace 权重格式。

# Copyright 2023-2025 SGLang Team # 版权所有 2023-2025 SGLang 团队
# Licensed under the Apache License, Version 2.0 (the "License"); # 根据 Apache 许可证 2.0 版本授权
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件。
# You may obtain a copy of the License at # 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不附带任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and # 请参阅许可证以获取管理权限和
# limitations under the License. # 限制的具体条款
# ==============================================================================

# Modeling from:  # 建模参考：
# ./llama.py and  # ./llama.py 和
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glmasr/modular_glmasr.py  # HuggingFace GLM-ASR 模块化实现
"""Inference-only GLM-ASR-HF model compatible with HuggingFace weights."""  # 仅推理的 GLM-ASR-HF 模型，兼容 HuggingFace 权重

import logging  # 导入日志模块
from typing import Any, Iterable, List, Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
from transformers import GlmAsrConfig, GlmAsrEncoderConfig  # 导入 GLM-ASR 配置和编码器配置
from transformers.models.glmasr.modeling_glmasr import (  # 从 GLM-ASR 模型导入
    GlmAsrEncoder,  # GLM-ASR 音频编码器
    GlmAsrMultiModalProjector,  # GLM-ASR 多模态投影器
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.llama import LlamaForCausalLM  # 从 Llama 模型导入因果语言模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class GlmAsrForConditionalGeneration(nn.Module):  # GLM-ASR 条件生成模型类
    # BitandBytes specific attributes  # BitsAndBytes 特定属性
    default_bitsandbytes_target_modules = [  # 默认 BitsAndBytes 目标模块列表
        ".gate_proj.",  # 门控投影
        ".down_proj.",  # 下投影
        ".up_proj.",  # 上投影
        ".q_proj.",  # Q 投影
        ".k_proj.",  # K 投影
        ".v_proj.",  # V 投影
        ".o_proj.",  # O 投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitsAndBytes 堆叠参数映射
        # shard_name, weight_name, index  # 分片名, 权重名, 索引
        "q_proj": ("qkv_proj", 0),  # Q 投影映射到 QKV 投影的第0个分片
        "k_proj": ("qkv_proj", 1),  # K 投影映射到 QKV 投影的第1个分片
        "v_proj": ("qkv_proj", 2),  # V 投影映射到 QKV 投影的第2个分片
        "gate_proj": ("gate_up_proj", 0),  # 门控投影映射到门控上投影的第0个分片
        "up_proj": ("gate_up_proj", 1),  # 上投影映射到门控上投影的第1个分片
    }

    def __init__(  # 初始化方法
        self,
        config: GlmAsrConfig,  # GLM-ASR 配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        if getattr(self.config, "audio_config", None) is None:  # 如果配置中没有音频配置
            self.config.audio_config = GlmAsrEncoderConfig(self.config._name_or_path)  # 使用模型路径创建默认音频编码器配置

        self.audio_tower = GlmAsrEncoder(  # 创建音频编码器（音频塔）
            config.audio_config,  # 音频配置
        )
        self.multi_modal_projector = GlmAsrMultiModalProjector(config)  # 创建多模态投影器
        self.language_model = LlamaForCausalLM(  # 创建 Llama 因果语言模型
            config.text_config, quant_config, prefix=add_prefix("model", prefix)  # 传入文本配置、量化配置和前缀
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态数据填充模式

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID方法，将多模态token插入到输入序列中
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式填充输入token

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取音频特征方法
        # Extract audio features from input items  # 从输入项中提取音频特征
        input_features = torch.cat([item.feature for item in items], dim=0).type(  # 拼接所有音频特征
            self.audio_tower.dtype  # 转换为音频编码器的数据类型
        )

        audio_embeds = self.audio_tower(input_features).last_hidden_state  # 通过音频编码器获取最后一层隐藏状态
        audio_embeds = audio_embeds.reshape(  # 重塑音频嵌入
            -1, self.config.audio_config.intermediate_size  # 展平为 (序列长度, 中间层维度)
        )
        audio_embeds = self.multi_modal_projector(audio_embeds)  # 通过多模态投影器映射到语言模型空间

        return audio_embeds  # 返回音频嵌入

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID 张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        **kwargs: Any,  # 其他关键字参数
    ) -> torch.Tensor:  # 返回隐藏状态张量
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次信息
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数字典
                Modality.AUDIO: self.get_audio_feature,  # 音频模态使用 get_audio_feature 方法
            },
            positions=positions,  # 位置编码
        )

        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # QKV 投影中的 Q
            ("qkv_proj", "k_proj", "k"),  # QKV 投影中的 K
            ("qkv_proj", "v_proj", "v"),  # QKV 投影中的 V
            ("gate_up_proj", "gate_proj", 0),  # 门控上投影中的门控投影
            ("gate_up_proj", "up_proj", 1),  # 门控上投影中的上投影
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取模型参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转嵌入的逆频率
                continue  # 跳过

            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:  # 如果绑定词嵌入且是语言模型头权重
                continue  # 跳过，避免重复加载

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name or "audio_tower" in name:  # 如果分片名不在名称中或是音频塔权重
                    continue  # 跳过
                name_tmp = name.replace(weight_name, param_name)  # 替换分片名为参数名

                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载。
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                param = params_dict[name_tmp]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果没有匹配的堆叠参数
                try:  # 尝试
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载。
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                        continue  # 跳过
                    param = params_dict[name]  # 获取参数
                except KeyError:  # 捕获键错误
                    print(params_dict.keys())  # 打印参数字典的所有键
                    raise  # 重新抛出异常

                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器，默认使用 default_weight_loader
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = GlmAsrForConditionalGeneration  # 入口类，用于模型注册
