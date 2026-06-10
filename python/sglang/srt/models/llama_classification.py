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

# LLaMA 分类模型实现：基于 LlamaModel 的文本分类模型，
# 在模型输出之上添加分类头，将隐藏状态映射到分类标签空间，
# 使用评分和池化函数进行分类预测。

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    score_and_pool,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel
from sglang.srt.utils import add_prefix


# LLaMA 分类模型：在 LlamaModel 基础上添加分类头
class LlamaForClassification(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        # 共享 LlamaModel 作为特征提取器
        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # 分类头：将隐藏状态映射到分类输出维度
        self.classification_head = nn.Linear(
            config.hidden_size, config.classification_out_size, bias=False
        )
        # 池化层：取最后一个 token 的隐藏状态，不做归一化
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    # 分类模型前向传播
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
        ), "LlamaForClassification is only used for embedding. Please add --is-embedding when you launch the server."

        # 获取模型的隐藏状态
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        # 使用评分和池化函数进行分类预测
        return score_and_pool(
            self.classification_head,
            self.pooler,
            hidden_states,
            forward_batch,
            input_ids,
        )

    # 加载权重：分别处理分类头和其他权重
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "classification_head" in name:
                # 直接加载分类头权重
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif "lm_head" in name:
                # 跳过语言模型头的权重（分类模型不使用）
                continue
            else:
                # 其他权重委托给 LlamaForCausalLM 加载
                LlamaForCausalLM.load_weights(self, [(name, loaded_weight)])


EntryClass = LlamaForClassification
