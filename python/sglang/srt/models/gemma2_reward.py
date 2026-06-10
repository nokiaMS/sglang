# Gemma2奖励模型：基于Gemma2架构的序列分类模型，用于奖励评分和嵌入提取
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

from typing import Iterable, Optional, Tuple  # 导入类型提示工具 # import type hints

import torch  # 导入PyTorch库 # import PyTorch
from torch import nn  # 导入神经网络模块 # import neural network module
from transformers import Gemma2Config  # 导入Gemma2配置类 # import Gemma2 config

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType  # 导入池化层组件 # import pooler components
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # import quantization config
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # import forward batch info
from sglang.srt.models.gemma2 import Gemma2ForCausalLM, Gemma2Model  # 导入Gemma2因果语言模型和基础模型 # import Gemma2 causal LM and base model
from sglang.srt.utils import add_prefix  # 导入前缀添加工具 # import prefix utility


class Gemma2ForSequenceClassification(nn.Module):  # Gemma2序列分类模型类 # Gemma2 sequence classification model class
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma2Config,  # Gemma2模型配置 # Gemma2 model config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__()  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置对象 # save config object
        self.quant_config = quant_config  # 保存量化配置 # save quantization config
        self.num_labels = config.num_labels  # 分类标签数量 # number of classification labels
        self.model = Gemma2Model(  # 创建Gemma2基础模型实例 # create Gemma2 base model instance
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)  # 评分线性层，将隐藏状态映射到标签数，无偏置 # score linear layer, maps hidden states to label count, no bias
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)  # 池化器，取最后一个token的隐藏状态，不进行归一化 # pooler, takes last token hidden state, no normalization

        self.eos_token_id = config.eos_token_id  # 序列结束符token ID # end of sequence token ID

    @torch.no_grad()  # 禁用梯度计算 # disable gradient computation
    def forward(  # 前向传播方法 # forward pass method
        self,
        input_ids: torch.Tensor,  # 输入token ID张量 # input token ID tensor
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        input_embeds: torch.Tensor = None,  # 输入嵌入张量，可选 # input embedding tensor, optional
        get_embedding: bool = True,  # 是否获取嵌入表示 # whether to get embedding representation
    ) -> EmbeddingPoolerOutput:  # 返回嵌入池化输出 # return embedding pooler output
        assert (  # 断言检查 # assertion check
            get_embedding
        ), "Gemma2ForSequenceClassification is only used for embedding"  # Gemma2序列分类模型仅用于嵌入提取 # Gemma2 sequence classification model only used for embedding

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过基础模型获取隐藏状态 # get hidden states through base model
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings  # 池化获取最后一个token的隐藏状态 # pool to get last token hidden state
        scores = self.score(last_token_hidden)  # 通过评分层计算分类得分 # compute classification scores through score layer

        return EmbeddingPoolerOutput(  # 返回嵌入池化输出结果 # return embedding pooler output result
            embeddings=scores,  # 分类评分作为嵌入输出 # classification scores as embedding output
            pooled_hidden_states=(  # 池化后的隐藏状态 # pooled hidden states
                last_token_hidden if forward_batch.return_pooled_hidden_states else None  # 根据标志决定是否返回池化隐藏状态 # return pooled hidden states based on flag
            ),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重方法 # load model weights method
        Gemma2ForCausalLM.load_weights(self, weights)  # 复用Gemma2ForCausalLM的权重加载逻辑 # reuse Gemma2ForCausalLM weight loading logic


EntryClass = [Gemma2ForSequenceClassification]  # 模型入口类列表 # model entry class list
