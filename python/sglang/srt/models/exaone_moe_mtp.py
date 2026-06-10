# EXAONE-MoE MTP推测解码模型推理实现文件
# 本文件实现了基于EXAONE-MoE的多token预测(MTP)推测解码模型
# MTP通过预测多个后续token来加速推理，继承自ExaoneMoEForCausalLM

# Copyright 2025 The LG AI Research Team  # LG AI研究团队版权声明
# Copyright 2023-2024 SGLang Team  # SGLang团队版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache 2.0许可证授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache 2.0许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不附带任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 请参阅许可证获取管理权限和
# limitations under the License.  # 限制的具体条款
# ==============================================================================  # 分隔线

# Adapted from the vLLM version of EXAONE-MoE MTP  # 从vLLM版本的EXAONE-MoE MTP适配而来
"""Inference-only ExaoneMoE MTP Speculative Decoding."""  # 仅推理的ExaoneMoE MTP推测解码

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch
from torch import nn  # 从PyTorch导入神经网络模块
from transformers import PretrainedConfig  # 从transformers导入预训练配置类

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.exaone_moe import ExaoneMoEForCausalLM, ExaoneMoEModel  # 导入ExaoneMoE基类和模型
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ExaoneMoEForCausalLMMTP(ExaoneMoEForCausalLM):  # ExaoneMoE MTP推测解码模型，继承自ExaoneMoEForCausalLM
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        nn.Module.__init__(self)  # 直接调用nn.Module初始化（跳过父类）
        self.config = config  # 保存配置
        config.num_hidden_layers = 1  # MTP只使用1层解码器
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.quant_config = quant_config  # 保存量化配置
        self.pp_group = get_pp_group()  # 获取流水线并行组

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)  # 全连接层：拼接嵌入和隐藏状态后投影
        self.pre_fc_norm_embedding = RMSNorm(  # 嵌入的预归一化
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态的预归一化
        self.model = ExaoneMoEModel(  # 模型主体（1层）
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 并行语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 参数前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力张量并行组
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        **kwargs,  # 其他关键字参数
    ):
        if input_embeds is None:  # 如果没有提供输入嵌入
            input_embeds = self.model.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入

        hidden_states = forward_batch.spec_info.hidden_states  # 从推测信息获取隐藏状态

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            input_embeds = self.pre_fc_norm_embedding(input_embeds)  # 对嵌入进行预归一化
            hidden_states = self.pre_fc_norm_hidden(hidden_states)  # 对隐藏状态进行预归一化
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))  # 拼接嵌入和隐藏状态后通过全连接层

        hidden_states = self.model(  # 通过模型主体
            input_ids,  # 输入ID
            positions,  # 位置编码
            forward_batch,  # 前向批次信息
            hidden_states,  # 全连接层输出的隐藏状态
        )

        return self.logits_processor(  # 通过logits处理器
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(  # 加载权重函数
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False  # 权重迭代器，是否为MTP
    ):
        super().load_weights(weights, is_mtp=True)  # 调用父类的权重加载，强制is_mtp=True


EntryClass = ExaoneMoEForCausalLMMTP  # 模型入口类
