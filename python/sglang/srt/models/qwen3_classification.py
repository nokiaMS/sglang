# Qwen3 序列分类模型实现
# 本文件实现了基于 Qwen3 的序列分类模型，用于文本分类任务。
# 提供了 Qwen3ForPooledOutput 基类和 Qwen3ForSequenceClassification 子类，
# 支持池化输出和分类评分。
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

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块
from transformers import Qwen2Config  # Qwen3 uses Qwen2Config  # Qwen3 使用 Qwen2Config

from sglang.srt.layers.pooler import (  # 导入池化器相关组件
    EmbeddingPoolerOutput,  # 嵌入池化输出
    Pooler,  # 池化器
    PoolingType,  # 池化类型
    score_and_pool,  # 评分和池化函数
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen3 import Qwen3Model  # 导入 Qwen3 模型主体
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

logger = logging.getLogger(__name__)  # 获取日志记录器


class Qwen3ForPooledOutput(nn.Module):
    """Qwen3 池化输出基类，用于分类和奖励等任务。

    Subclasses should set self.score and self.pooler in their __init__.  # 子类应在 __init__ 中设置 self.score 和 self.pooler
    """

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen3 池化输出基类"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = Qwen3Model(  # 创建 Qwen3 模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        self.eos_token_id = config.eos_token_id  # EOS token ID
        # Subclasses must set self.score and self.pooler  # 子类必须设置 self.score 和 self.pooler

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.model.get_input_embeddings()  # 返回模型主体的输入嵌入层

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        """池化输出前向传播：模型主体 -> 评分头 -> 池化"""
        assert get_embedding, f"{self.__class__.__name__} is only used for embedding"  # 断言必须获取嵌入

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型主体
        return score_and_pool(  # 评分和池化
            self.score, self.pooler, hidden_states, forward_batch, input_ids  # 评分头、池化器、隐藏状态、批次和输入 ID
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理堆叠参数映射"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q 映射
            ("qkv_proj", "k_proj", "k"),  # K 映射
            ("qkv_proj", "v_proj", "v"),  # V 映射
            ("gate_up_proj", "gate_proj", 0),  # gate 映射
            ("gate_up_proj", "up_proj", 1),  # up 映射
        ]

        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            # Skip lm_head weights (pooled output models don't have lm_head)  # 跳过 lm_head 权重
            if name.startswith("lm_head"):  # 如果是语言模型头权重
                continue

            # Skip rotary embeddings and other non-parameter tensors  # 跳过旋转嵌入等非参数张量
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转嵌入逆频率和投影器
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的余弦/正弦
                continue

            # Handle stacked parameters (qkv_proj, gate_up_proj)  # 处理堆叠参数
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models  # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                    continue
                if name not in params_dict:  # 如果参数不在字典中
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数处理
                # Skip loading extra bias for GPTQ models  # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                    continue

                if name in params_dict:  # 如果参数在字典中
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                else:  # 参数不在字典中
                    logger.warning(f"Parameter {name} not found in params_dict")  # 警告参数未找到


class Qwen3ForSequenceClassification(Qwen3ForPooledOutput):
    """Qwen3 序列分类模型，添加线性评分头"""

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 Qwen3 序列分类模型"""
        super().__init__(config, quant_config, prefix)  # 调用父类初始化
        self.score = nn.Linear(config.hidden_size, config.num_labels)  # 线性评分头
        # Use normalize=True for qwen3 embedding based on official implementation  # 根据 Qwen3 官方实现使用归一化
        # Reference: https://github.com/QwenLM/Qwen3-Embedding/blob/main/examples/qwen3_embedding_transformers.py#L55  # 参考 Qwen3-Embedding 官方代码
        # Official code: output = F.normalize(output, p=2, dim=1)  # 官方代码使用 L2 归一化
        normalize = True  # 默认启用归一化

        # We don't want to normalize the embedding if we have a classification head  # 如果有分类头则不归一化
        if config.id2label is not None or config.label2id is not None:  # 如果有分类标签映射
            normalize = False  # 禁用归一化

        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=normalize)  # 创建池化器


EntryClass = [  # 模型入口类列表
    Qwen3ForSequenceClassification,  # Qwen3 序列分类模型
]
