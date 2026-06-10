# TeleFLM模型实现文件
# 本文件实现了TeleFLM模型，基于Llama架构，支持Maximal Update Parameterization (µP)缩放
# TeleFLM通过input_mult和output_mult参数实现跨尺度的损失预测和模型缩放

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/teleflm.py

from typing import List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
from transformers import LlamaConfig  # 导入Llama配置类

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel  # 导入Llama模型


class TeleFLMModel(LlamaModel):  # TeleFLM模型类，继承自LlamaModel
    """
    This implementation is based on the µScaling paper presented at
    the ICLR 2025 Workshop:
    NanoLM: An Affordable LLM Study Benchmark \
    via Accurate Loss Prediction across Scales
    by Yiqun Yao et al.
    Available at: https://openreview.net/forum?id=IwaPYg1SCA
    arXiv preprint: https://arxiv.org/abs/2304.06875
    """
    # TeleFLM模型基于µScaling论文，实现跨尺度损失预测

    def __init__(  # 初始化方法
        self,
        config: LlamaConfig,  # Llama配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)  # 调用父类初始化
        self.use_mup = getattr(self.config, "use_mup", False)  # 是否使用Maximal Update Parameterization
        if self.use_mup:  # 如果使用µP
            self.input_mult = self.config.input_mult  # 获取输入乘数

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，可选
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank and input_embeds is None:  # 如果是第一个rank且没有输入嵌入
            input_embeds = self.embed_tokens(input_ids)  # 通过嵌入层获取输入嵌入
            if self.use_mup:  # 如果使用µP
                input_embeds = input_embeds * self.input_mult  # 对输入嵌入应用乘数缩放

        return super().forward(  # 调用父类前向传播
            input_ids=input_ids,  # 传入输入ID
            positions=positions,  # 传入位置编码
            forward_batch=forward_batch,  # 传入前向批次
            input_embeds=input_embeds,  # 传入输入嵌入
            pp_proxy_tensors=pp_proxy_tensors,  # 传入流水线代理张量
        )


class TeleFLMForCausalLM(LlamaForCausalLM):  # TeleFLM因果语言模型类，继承自LlamaForCausalLM
    def __init__(  # 初始化方法
        self,
        config: LlamaConfig,  # Llama配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__(config, quant_config=quant_config, prefix=prefix)  # 调用父类初始化
        self.use_mup = getattr(self.config, "use_mup", False)  # 是否使用µP
        if self.use_mup:  # 如果使用µP
            self.mup_scale_factor = self.config.mup_scale_factor  # 获取µP缩放因子
            self.output_mult = self.config.output_mult / self.mup_scale_factor  # 计算输出乘数
            self.logits_processor.logit_scale = self.output_mult  # 设置logits缩放因子

    def _init_model(  # 初始化模型方法
        self,
        config: LlamaConfig,  # Llama配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ):
        return TeleFLMModel(config, quant_config=quant_config, prefix=prefix)  # 返回TeleFLM模型实例


EntryClass = TeleFLMForCausalLM  # 注册入口类为TeleFLMForCausalLM
