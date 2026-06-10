# Qwen2 EAGLE 推测解码模型实现
# 本文件实现了基于 Qwen2 的 EAGLE 推测解码模型，用于加速推理。
# EAGLE 通过轻量级草稿模型预测多个 token，再由目标模型验证，从而提高推理吞吐量。
"""
Copyright 2023-2024 SGLang Team  # SGLang 团队版权
Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0 许可证
you may not use this file except in compliance with the License.  # 不得违反许可证使用
You may obtain a copy of the License at  # 可在以下地址获取许可证

    http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址

Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
distributed under the License is distributed on an "AS IS" BASIS,  # 按原样分发
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何担保
See the License for the specific language governing permissions and  # 查看许可证获取权限
limitations under the License.  # 许可证限制
"""

from sglang.srt.utils import add_prefix  # 导入前缀添加工具

# Adapted from  # 适配自
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py  # EAGLE 项目中的实现
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""  # 仅推理的 LLaMA-EAGLE 模型

from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块

from sglang.srt.distributed import get_pp_group  # 导入流水线并行组
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM  # 导入 Qwen2 解码器层和因果语言模型

Qwen2Config = None  # Qwen2 配置占位符（由外部注入）


class Qwen2DecoderLayer(Qwen2DecoderLayer):
    """Qwen2 EAGLE 解码器层，第一层跳过输入归一化"""

    def __init__(
        self,
        config: Qwen2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 EAGLE 解码器层"""
        super().__init__(config, layer_id, quant_config=quant_config, prefix=prefix)  # 调用父类初始化

        # Skip the input_layernorm  # 跳过输入层归一化
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427  # EAGLE 参考实现
        if layer_id == 0:  # 如果是第一层
            del self.input_layernorm  # 删除输入层归一化
            setattr(self, "input_layernorm", lambda x: x)  # 替换为恒等函数


class Qwen2Model(nn.Module):
    """Qwen2 EAGLE 模型主体，包含嵌入层、解码器层和特征融合层"""

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 EAGLE 模型主体"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 词表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 嵌入维度
            prefix=add_prefix("embed_tokens", prefix),  # 参数前缀
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                Qwen2DecoderLayer(  # 创建解码器层
                    config,  # 配置
                    i,  # 层 ID
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 参数前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历所有层
            ]
        )
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)  # 特征融合线性层

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        """EAGLE 模型前向传播：嵌入 -> 特征融合 -> 解码器层"""
        if input_embeds is None:  # 如果没有输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层
        else:  # 有输入嵌入
            hidden_states = input_embeds  # 直接使用输入嵌入

        hidden_states = self.fc(  # 通过特征融合层
            torch.cat((hidden_states, forward_batch.spec_info.hidden_states), dim=-1)  # 拼接当前嵌入和推测信息隐藏状态
        )

        residual = None  # 残差初始化为空
        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                positions,  # 位置信息
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
                residual,  # 残差
            )
        return hidden_states + residual  # 返回隐藏状态加残差


class Qwen2ForCausalLMEagle(Qwen2ForCausalLM):
    """Qwen2 EAGLE 因果语言模型，用于推测解码"""

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化 EAGLE 因果语言模型"""
        nn.Module.__init__(self)  # 直接调用 nn.Module 初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.model = Qwen2Model(  # 创建 EAGLE 模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        if self.config.tie_word_embeddings:  # 如果共享词嵌入
            self.lm_head = self.model.embed_tokens  # 共享嵌入层
        else:  # 不共享词嵌入
            self.lm_head = ParallelLMHead(  # 创建并行语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏维度
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
            )
        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载 EAGLE 模型权重，将非 lm_head 权重添加 model 前缀"""
        for name, loaded_weight in weights:  # 遍历所有权重
            if "lm_head" not in name:  # 如果不是语言模型头权重
                name = "model." + name  # 添加 model 前缀
                super().load_weights([(name, loaded_weight)])  # 使用父类方法加载


EntryClass = [Qwen2ForCausalLMEagle]  # 模型入口类列表
