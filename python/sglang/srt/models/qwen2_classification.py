# Qwen2 序列分类模型实现
# 本文件实现了基于 Qwen2 的序列分类模型，用于文本分类任务。
# 该模型复用 Qwen2Model 作为骨干网络，并添加线性评分头和池化器。
# Copyright 2023-2024 SGLang Team  # SGLang 团队版权
# Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 不得违反许可证使用
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何担保
# See the License for the specific language governing permissions and  # 查看许可证获取权限
# limitations under the License.  # 许可证限制
# ==============================================================================

from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块
from transformers import Qwen2Config  # 导入 Qwen2 配置

from sglang.srt.layers.pooler import (  # 导入池化器相关组件
    EmbeddingPoolerOutput,  # 嵌入池化输出
    Pooler,  # 池化器
    PoolingType,  # 池化类型
    score_and_pool,  # 评分和池化函数
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.qwen2 import Qwen2ForCausalLM, Qwen2Model  # 导入 Qwen2 因果语言模型和模型主体
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class Qwen2ForSequenceClassification(nn.Module):
    """Qwen2 序列分类模型，在 Qwen2Model 基础上添加线性评分头"""

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen2 序列分类模型"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = Qwen2Model(  # 创建 Qwen2 模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        self.score = nn.Linear(config.hidden_size, config.num_labels)  # 线性评分头
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)  # 池化器（取最后一个 token，不归一化）

        self.eos_token_id = config.eos_token_id  # EOS token ID

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        """序列分类前向传播：模型主体 -> 评分头 -> 池化"""
        assert (  # 断言
            get_embedding  # 必须获取嵌入
        ), "Qwen2ForSequenceClassification is only used for embedding"  # Qwen2 序列分类仅用于嵌入

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型主体
        return score_and_pool(  # 评分和池化
            self.score, self.pooler, hidden_states, forward_batch, input_ids  # 评分头、池化器、隐藏状态、批次和输入 ID
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，过滤掉语言模型头的权重"""
        # Filter out lm_head weights of Qwen2ForCausalLM  # 过滤 Qwen2ForCausalLM 的 lm_head 权重
        filtered_weights = [  # 过滤后的权重列表
            (name, w) for name, w in weights if not name.startswith("lm_head")  # 排除 lm_head 开头的权重
        ]
        return Qwen2ForCausalLM.load_weights(self, filtered_weights)  # 使用父类方法加载过滤后的权重


EntryClass = [  # 模型入口类列表
    Qwen2ForSequenceClassification,  # Qwen2 序列分类模型
]
