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

# LLaMA 奖励模型实现：用于强化学习人类反馈（RLHF）的奖励评分模型，
# 包含两个变体：
# - LlamaForSequenceClassification：基础版本，直接输出分类评分
# - LlamaForSequenceClassificationWithNormal_Weights：带权重归一化的版本，
#   通过额外的权重网络计算归一化系数，将评分与权重加权求和得到最终奖励值

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    pool_hidden_states,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel
from sglang.srt.utils import add_prefix


# LLaMA 序列分类/奖励模型：在 LlamaModel 基础上添加评分头
class LlamaForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        # 分类标签数（奖励模型通常为 1 或 2）
        self.num_labels = config.num_labels
        # 共享 LlamaModel 作为特征提取器
        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # 评分头：将隐藏状态映射到标签空间
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        # 池化层：取最后一个 token 的隐藏状态，不做归一化
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

        # EOS token ID
        self.eos_token_id = config.eos_token_id

    # 奖励模型前向传播
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert (
            get_embedding
        ), "LlamaForSequenceClassification is only used for embedding"

        # 获取模型的隐藏状态
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        # 池化提取最后一个 token 的隐藏状态
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings
        # 通过评分头计算奖励分数
        scores = self.score(last_token_hidden)

        return EmbeddingPoolerOutput(
            embeddings=scores,
            pooled_hidden_states=(
                last_token_hidden if forward_batch.return_pooled_hidden_states else None
            ),
        )

    # 加载权重：委托给 LlamaForCausalLM
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return LlamaForCausalLM.load_weights(self, weights)


# 带权重归一化的奖励模型：通过额外的权重网络对评分进行加权
class LlamaForSequenceClassificationWithNormal_Weights(LlamaForSequenceClassification):
    # 权重网络：将隐藏状态映射到归一化权重
    class Weights(torch.nn.Module):
        def __init__(self, hidden_size, num_label):
            super().__init__()
            # 三层 MLP 权重网络，输出维度为标签数的一半
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
                torch.nn.SELU(),
                torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
                torch.nn.SELU(),
                torch.nn.Linear(hidden_size, num_label // 2, dtype=torch.float16),
            )

        def forward(self, x):
            return self.fc(x.to(torch.float16))

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix=prefix)
        # 初始化权重网络
        self.weights = self.Weights(config.hidden_size, self.num_labels)

    # 带权重归一化的奖励模型前向传播
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert (
            get_embedding
        ), "LlamaForSequenceClassification is only used for embedding"
        # 获取模型的隐藏状态
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        # 计算评分 logits
        logits = self.score(hidden_states)
        # 计算权重
        weights = self.weights(hidden_states)

        # 池化提取最后一个 token 的评分和权重
        pooled_logits = self.pooler(logits, forward_batch).embeddings
        pooled_weights = self.pooler(weights, forward_batch).embeddings

        # 将 logits 重塑为 (batch, num_labels//2, 2)，取第一个元素作为奖励值
        rews = pooled_logits.view(-1, self.num_labels // 2, 2)[:, :, 0].view(
            -1, self.num_labels // 2
        )
        # 奖励值与权重加权求和，得到最终评分
        scores = (rews * pooled_weights).sum(dim=-1).view(-1, 1)

        pooled_hidden = None
        if forward_batch.return_pooled_hidden_states:
            # 返回池化后的隐藏状态
            pooled_hidden = pool_hidden_states(
                self.pooler.pooling_type, hidden_states, forward_batch
            )

        return EmbeddingPoolerOutput(
            embeddings=scores,
            pooled_hidden_states=pooled_hidden,
        )

    # 加载权重：委托给父类
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(weights)


EntryClass = [
    LlamaForSequenceClassification,
    LlamaForSequenceClassificationWithNormal_Weights,
]
