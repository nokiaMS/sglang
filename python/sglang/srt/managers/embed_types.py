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
"""
Dataclasses for embedding injection.

These are placed in a separate module to avoid circular imports between
io_struct.py and schedule_batch.py.
"""
# 嵌入注入相关的数据类定义。
# 这些类被放置在单独的模块中，以避免 io_struct.py 和 schedule_batch.py 之间的循环导入。

from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class PositionalEmbeds:
    """Embeddings to place at specific token positions.

    Accepts either a list of [1, hidden_dim] tensors or a pre-stacked [N, hidden_dim] tensor.
    In both cases, __post_init__ stacks into a single [N, hidden_dim] tensor to reduce
    ZMQ serialization overhead.

    Attributes:
        embeds: Stacked tensor of shape [N, hidden_dim] after __post_init__.
        positions: List of positions where embeddings should be injected.
    """
    # 位置嵌入数据类：用于在特定 token 位置放置嵌入向量。
    # 接受 [1, hidden_dim] 张量列表或预堆叠的 [N, hidden_dim] 张量。
    # __post_init__ 会将其统一堆叠为单个 [N, hidden_dim] 张量，以减少 ZMQ 序列化开销。

    embeds: Union[List[torch.Tensor], torch.Tensor]  # 嵌入向量，初始化后为 [N, hidden_dim] 的张量
    positions: List[int]  # 需要注入嵌入向量的位置列表

    def __post_init__(self):
        # Normalize list of tensors into a single [N, hidden_dim] tensor.
        # Dispatch by element rank to avoid a per-element unsqueeze.
        # 将张量列表归一化为单个 [N, hidden_dim] 张量。
        # 根据元素的维度进行分发，避免逐元素 unsqueeze 操作。
        if isinstance(self.embeds, list):
            if not self.embeds:
                self.embeds = torch.cat(self.embeds, dim=0)  # raises — empty is invalid  # 空列表会抛出异常——空输入无效
            elif self.embeds[0].dim() == 1:
                # [hidden_dim] elements → stack adds the leading dim.
                # [hidden_dim] 元素 → stack 会添加前导维度
                self.embeds = torch.stack(self.embeds, dim=0)
            else:
                # [1, hidden_dim] (already has the leading dim) → plain concat.
                # [1, hidden_dim]（已有前导维度）→ 直接拼接
                self.embeds = torch.cat(self.embeds, dim=0)
        if self.embeds.shape[0] != len(self.positions):
            # 校验嵌入向量数量与位置数量是否一致
            raise ValueError(
                f"embeds length ({self.embeds.shape[0]}) != "
                f"positions length ({len(self.positions)})"
            )
