# GLM-OCR推测解码（Next-N预测）模型实现
# 本文件实现了GLM-OCR模型的推测解码（Speculative Decoding）功能，
# 通过Next-N预测机制加速推理，包含GlmOcrModelNextN和
# GlmOcrForConditionalGenerationNextN两个核心类。

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

"""Inference-only GLM-OCR Speculative Decoding."""  # 仅推理的GLM-OCR推测解码

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入全局专家分布记录器
from sglang.srt.layers.dp_attention import is_dp_attention_enabled  # 导入判断是否启用DP注意力的函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.glm4 import Glm4DecoderLayer  # 导入GLM4解码器层
from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration  # 导入GLM-OCR条件生成模型
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import add_prefix  # 导入前缀添加工具函数

logger = logging.getLogger(__name__)  # 创建日志记录器


class GlmOcrModelNextN(nn.Module):
    """GLM-OCR Next-N预测模型，用于推测解码。"""

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":  # 如果使用modelopt_fp4量化
            logger.warning(  # 记录警告
                "Overriding GlmOcrModelNextN quant config for modelopt_fp4 GLM-OCR model."  # 覆盖GLM-OCR模型的modelopt_fp4量化配置
            )
            quant_config = None  # 将量化配置设为None，因为不支持

        self.vocab_size = config.vocab_size  # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            enable_tp=not is_dp_attention_enabled(),  # 是否启用张量并行，取决于是否启用DP注意力
            prefix=add_prefix("embed_tokens", prefix),  # 添加前缀
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 嵌入归一化层
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态归一化层

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)  # 嵌入与隐藏状态融合投影层

        self.decoder = Glm4DecoderLayer(  # GLM4解码器层
            config,  # 配置
            0,  # 层ID为0
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("decoder", prefix),  # 添加前缀
        )

        self.shared_head = nn.Module()  # 共享头模块
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 共享头的归一化层

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层获取隐藏状态
        else:  # 否则
            hidden_states = input_embeds  # 直接使用输入嵌入

        if hidden_states.shape[0] > 0:  # 如果有有效token
            hidden_states = self.eh_proj(  # 通过融合投影层
                torch.cat(  # 拼接嵌入和隐藏状态
                    (
                        self.enorm(hidden_states),  # 归一化后的当前嵌入
                        self.hnorm(forward_batch.spec_info.hidden_states),  # 归一化后的推测信息隐藏状态
                    ),
                    dim=-1,  # 在最后一维拼接
                )
            )

        residual = None  # 初始化残差为None
        with get_global_expert_distribution_recorder().disable_this_region():  # 禁用专家分布记录
            hidden_states, residual = self.decoder(  # 通过解码器层
                positions, hidden_states, forward_batch, residual  # 传入位置、隐藏状态、批次和残差
            )

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            if residual is not None:  # 如果有残差
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)  # 带残差的归一化
            else:  # 否则
                hidden_states = self.shared_head.norm(hidden_states)  # 无残差的归一化

        return hidden_states  # 返回隐藏状态


class GlmOcrForConditionalGenerationNextN(GlmOcrForConditionalGeneration):
    """GLM-OCR条件生成Next-N预测模型，继承自GlmOcrForConditionalGeneration。"""

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        nn.Module.__init__(self)  # 直接调用nn.Module的初始化，跳过父类初始化
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.quant_config = quant_config  # 量化配置
        self.model = GlmOcrModelNextN(  # 创建Next-N模型
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置和前缀
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model.shared_head.head", prefix),  # 添加前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力TP组
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

        self.num_fused_shared_experts = (  # 融合共享专家数量
            0 if get_global_server_args().disable_shared_experts_fusion else 1  # 如果禁用融合则为0，否则为1
        )

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)  # 通过模型获取隐藏状态
        return self.logits_processor(  # 通过logits处理器获取预测结果
            input_ids, hidden_states, self.lm_head, forward_batch  # 传入输入ID、隐藏状态、语言模型头和批次
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        super().load_weights(weights, is_nextn=True)  # 调用父类的加载权重方法，标记为Next-N模式


EntryClass = [GlmOcrForConditionalGenerationNextN]  # 入口类列表
