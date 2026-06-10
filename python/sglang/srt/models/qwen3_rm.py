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

# Qwen3 奖励模型：用于 RLHF（基于人类反馈的强化学习）和 best-of-N 采样的奖励评分模型
"""Qwen3 Reward Model for RLHF and best-of-N sampling."""

from typing import Optional

from torch import nn
from transformers import Qwen2Config  # Qwen3 uses Qwen2Config

from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen3_classification import Qwen3ForPooledOutput


# Qwen3 奖励模型，继承自 Qwen3ForPooledOutput，使用两层 MLP 作为评分头
class Qwen3ForRewardModel(Qwen3ForPooledOutput):
    """Qwen3 Reward Model with 2-layer MLP scoring head for RLHF."""

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.num_labels = 1  # 奖励模型只输出一个标量分数
        # 两层 MLP 评分头：hidden_size -> hidden_size -> 1
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        )
        # 使用最后一个 token 的隐状态进行池化，不进行归一化
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)


# 入口类列表，框架通过此变量注册模型
EntryClass = [
    Qwen3ForRewardModel,
]
