# InternLM2 奖励模型实现
# 本文件基于 InternLM2 因果语言模型实现了奖励模型，用于对文本生成质量进行评分，
# 通过在隐藏状态上添加值头来输出奖励分数。

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

from typing import Iterable, Optional, Tuple  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType  # 导入池化层相关组件
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批处理信息
from sglang.srt.models.internlm2 import InternLM2ForCausalLM, InternLM2Model  # 导入 InternLM2 模型类
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class InternLM2ForRewardModel(nn.Module):  # InternLM2 奖励模型类，基于 InternLM2 实现奖励评分
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存模型配置
        self.quant_config = quant_config  # 保存量化配置
        self.vocab_size = config.vocab_size  # 词表大小
        self.model = InternLM2Model(  # 创建 InternLM2 模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置和量化配置
        )
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)  # 值头线性层，将隐藏状态映射到标量奖励分数
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)  # 池化层，取最后一个 token 的输出，不归一化

    @torch.no_grad()  # 禁用梯度计算，用于推理
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批处理信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
        get_embedding: bool = True,  # 是否获取嵌入，默认为 True
    ) -> EmbeddingPoolerOutput:  # 返回嵌入池化输出
        assert get_embedding, "InternLM2ForRewardModel is only used for embedding"  # 断言：奖励模型仅用于嵌入模式
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型主体获取隐藏状态
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings  # 池化获取最后一个 token 的隐藏状态
        scores = self.v_head(last_token_hidden)  # 通过值头计算奖励分数
        return EmbeddingPoolerOutput(  # 返回嵌入池化输出
            embeddings=scores,  # 奖励分数作为嵌入
            pooled_hidden_states=(  # 池化后的隐藏状态
                last_token_hidden if forward_batch.return_pooled_hidden_states else None  # 如果需要则返回，否则为 None
            ),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        return InternLM2ForCausalLM.load_weights(self, weights)  # 复用 InternLM2ForCausalLM 的权重加载逻辑


EntryClass = InternLM2ForRewardModel  # 模型入口类，用于框架自动发现和注册模型
