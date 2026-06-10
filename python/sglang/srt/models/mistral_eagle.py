# Mistral EAGLE推测解码草稿模型实现
# 本文件实现了GQA Mistral目标模型的EAGLE推测解码草稿模型，
# 复用LlamaForCausalLMEagle的EAGLE机制，但替换为Mistral特定的草稿模型主体。
# 使用RowParallelLinear处理EAGLE fc融合层以支持FP8量化权重，
# 权重名称重映射参照MistralForCausalLMMistralFormat并增加eagle_linear映射。

# Copyright 2023-2026 SGLang Team  # 版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache 2.0许可证授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 许可证下的软件按"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限和
# limitations under the License.  # 限制的具体语言
# ==============================================================================  # 分隔线
"""EAGLE draft model for GQA Mistral targets (e.g. Mistral Medium 3.5).  # GQA Mistral目标模型的EAGLE草稿模型（如Mistral Medium 3.5）

Reuses ``LlamaForCausalLMEagle`` for the EAGLE machinery (lm_head/embed_tokens  # 复用LlamaForCausalLMEagle的EAGLE机制（lm_head/embed_tokens
construction, optional tied embeddings, capture-aux-hidden-states plumbing) but  # 构建、可选的共享嵌入、捕获辅助隐藏状态管道），但
swaps in a Mistral-specific draft model body that:  # 替换为Mistral特定的草稿模型主体：

- runs through the standard :class:`LlamaDecoderLayer` (GQA), not the layernorm  # 使用标准LlamaDecoderLayer（GQA），而不是无层归一化的
  -less variant ``llama_eagle.LlamaDecoderLayer`` — Mistral's EAGLE checkpoint  # llama_eagle.LlamaDecoderLayer变体——Mistral的EAGLE检查点
  ships ``layers.0.attention_norm.weight``, so layer 0 expects the input  # 包含layers.0.attention_norm.weight，因此第0层期望
  layernorm to be present.  # 输入层归一化存在
- uses ``RowParallelLinear`` for the EAGLE fc fusion layer with a  # 对EAGLE fc融合层使用RowParallelLinear并带
  ``quant_config``, so the FP8-quantized ``eagle_linear`` weights from the  # quant_config，以便Mistral原生检查点中的FP8量化eagle_linear权重
  Mistral native checkpoint load via the standard quant pipeline (``LlamaModel``  # 通过标准量化管道加载（llama_eagle.py中的LlamaModel
  in ``llama_eagle.py`` uses a plain :class:`torch.nn.Linear` which cannot  # 使用普通的torch.nn.Linear，无法
  consume FP8 e4m3 tensors).  # 消费FP8 e4m3张量）

The weight name remapping mirrors :class:`MistralForCausalLMMistralFormat` and  # 权重名称重映射参照MistralForCausalLMMistralFormat
adds the eagle-specific entries for ``eagle_linear`` → ``model.fc``.  # 并增加eagle_linear到model.fc的映射
"""

import logging  # 导入日志模块
from collections.abc import Iterable  # 导入可迭代类型
from typing import Optional, Tuple  # 导入可选和元组类型

import regex as re  # 导入正则表达式库
import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.distributed import get_pp_group  # 导入流水线并行组获取函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import RowParallelLinear  # 导入行并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和流水线代理张量
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM  # 导入Llama解码器层和因果语言模型
from sglang.srt.models.llama_eagle import LlamaForCausalLMEagle  # 导入Llama EAGLE因果语言模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class MistralEagleModel(nn.Module):  # Mistral EAGLE草稿模型主体
    """GQA EAGLE draft body with the input-embed ⊕ target-hidden-state fusion."""  # 带有输入嵌入与目标隐藏状态融合的GQA EAGLE草稿模型主体

    def __init__(  # 初始化方法
        self,  # 自身实例
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认无
        prefix: str = "",  # 前缀字符串，默认空
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 保存词表大小
        assert (  # 断言
            get_pp_group().world_size == 1  # 流水线并行世界大小为1
        ), "MistralForCausalLMEagle currently does not support pipeline parallelism"  # Mistral EAGLE目前不支持流水线并行
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.embed_tokens = VocabParallelEmbedding(  # 词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏维度
            prefix=add_prefix("embed_tokens", prefix),  # 带前缀的嵌入层名称
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                LlamaDecoderLayer(  # Llama解码器层（包含输入层归一化）
                    config=config,  # 配置
                    layer_id=i,  # 层ID
                    prefix=add_prefix(f"layers.{i}", prefix),  # 带前缀的层名称
                    quant_config=quant_config,  # 量化配置
                )
                for i in range(config.num_hidden_layers)  # 遍历隐藏层数量
            ]
        )
        self.start_layer = 0  # 起始层索引
        self.end_layer = config.num_hidden_layers  # 结束层索引
        self.fc = RowParallelLinear(  # EAGLE融合层：将词嵌入和目标模型隐藏状态拼接后投影
            config.hidden_size * 2,  # 输入维度为隐藏维度的两倍
            config.hidden_size,  # 输出维度为隐藏维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc", prefix),  # 带前缀的fc层名称
            input_is_parallel=False,  # 输入不并行
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终RMS归一化层

    def forward(  # 前向传播方法
        self,  # 自身实例
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，默认无
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，默认无
    ) -> torch.Tensor:  # 返回隐藏状态张量
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入
        else:  # 如果提供了输入嵌入
            hidden_states = input_embeds  # 直接使用

        # EAGLE fusion: concat input embedding with target's previous hidden  # EAGLE融合：将输入嵌入与目标模型前一隐藏状态拼接
        # state, project back to hidden_size before going through the draft's  # 投影回隐藏维度后再通过草稿模型的
        # transformer layers.  # transformer层
        hidden_states, _ = self.fc(  # 通过fc融合层投影
            torch.cat(  # 拼接
                (hidden_states, forward_batch.spec_info.hidden_states),  # 嵌入和目标模型隐藏状态
                dim=-1,  # 在最后一个维度上拼接
            )
        )

        residual = None  # 残差初始化为None
        for layer in self.layers:  # 遍历所有解码器层
            hidden_states, residual = layer(  # 通过解码器层
                positions, hidden_states, forward_batch, residual  # 传递位置、隐藏状态、批次和残差
            )
        return hidden_states + residual  # 返回隐藏状态加残差


class MistralForCausalLMEagle(LlamaForCausalLMEagle):  # Mistral EAGLE因果语言模型，继承自LlamaForCausalLMEagle
    """EAGLE draft for GQA Mistral targets.  # GQA Mistral目标模型的EAGLE草稿

    Inherits LlamaForCausalLMEagle for the lm_head/embed_tokens setup and the  # 继承LlamaForCausalLMEagle的lm_head/embed_tokens设置和
    capture-aux-hidden-state hooks, then overrides ``self.model`` with the  # 捕获辅助隐藏状态钩子，然后用支持量化的
    quant-aware :class:`MistralEagleModel` and applies Mistral native-format  # MistralEagleModel覆盖self.model，并在load_weights中
    weight remapping during ``load_weights``.  # 应用Mistral原生格式的权重重映射
    """

    # fmt: off  # 关闭格式化
    remapping = {  # Mistral原生格式到HF/Llama格式的权重名称映射表（含EAGLE专用映射）
        r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",  # 注意力层归一化权重映射
        r"layers\.(\d+)\.attention\.wq\.(\w+)": r"model.layers.\1.self_attn.q_proj.\2",  # 查询投影权重映射
        r"layers\.(\d+)\.attention\.wk\.(\w+)": r"model.layers.\1.self_attn.k_proj.\2",  # 键投影权重映射
        r"layers\.(\d+)\.attention\.wv\.(\w+)": r"model.layers.\1.self_attn.v_proj.\2",  # 值投影权重映射
        r"layers\.(\d+)\.attention\.wo\.(\w+)": r"model.layers.\1.self_attn.o_proj.\2",  # 输出投影权重映射
        r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",  # FFN层归一化权重映射
        r"layers\.(\d+)\.feed_forward\.w1\.(\w+)": r"model.layers.\1.mlp.gate_proj.\2",  # 前馈网络门控投影权重映射
        r"layers\.(\d+)\.feed_forward\.w2\.(\w+)": r"model.layers.\1.mlp.down_proj.\2",  # 前馈网络下投影权重映射
        r"layers\.(\d+)\.feed_forward\.w3\.(\w+)": r"model.layers.\1.mlp.up_proj.\2",  # 前馈网络上投影权重映射
        r"norm\.weight": "model.norm.weight",  # 最终归一化权重映射
        # Eagle-specific: the fc layer that fuses input embeds and target  # EAGLE专用：融合输入嵌入和目标
        # hidden states is named `eagle_linear` in the Mistral checkpoint.  # 隐藏状态的fc层在Mistral检查点中名为eagle_linear
        # Its FP8 weights live alongside per-tensor activation/weight scales.  # 其FP8权重与逐张量激活/权重缩放因子共存
        r"eagle_linear\.weight": r"model.fc.weight",  # EAGLE线性层权重映射到fc权重
        r"eagle_linear\.qscale_act": r"model.fc.input_scale",  # EAGLE线性层激活缩放映射到fc输入缩放
        r"eagle_linear\.qscale_weight": r"model.fc.weight_scale",  # EAGLE线性层权重缩放映射到fc权重缩放
        # tok_embeddings and output are intentionally absent — EAGLE shares  # tok_embeddings和output故意省略——EAGLE共享
        # both with the target model and the framework ties them at runtime.  # 两者与目标模型，框架在运行时绑定它们
    }
    # fmt: on  # 恢复格式化

    def __init__(  # 初始化方法
        self,  # 自身实例
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认无
        prefix: str = "",  # 前缀字符串，默认空
    ) -> None:
        # Run LlamaForCausalLMEagle.__init__ to set up lm_head/embed_tokens/etc.  # 运行LlamaForCausalLMEagle的初始化以设置lm_head/embed_tokens等
        # then replace self.model (which uses a plain torch.nn.Linear for fc and  # 然后用支持量化的草稿模型主体替换self.model
        # cannot consume FP8 weights) with our quant-aware draft body.  # （原模型使用普通torch.nn.Linear作为fc，无法消费FP8权重）
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)  # 调用父类初始化
        self.model = MistralEagleModel(  # 用Mistral EAGLE草稿模型替换默认模型
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 带前缀的模型名称
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        # Bypass LlamaForCausalLMEagle.load_weights' "prepend model." behaviour  # 绕过LlamaForCausalLMEagle.load_weights的"前置model."行为
        # because our remap already emits fully-qualified target names.  # 因为我们的重映射已经生成完全限定的目标名称
        return LlamaForCausalLM.load_weights(  # 调用LlamaForCausalLM的权重加载方法
            self, self._remap_mistral_to_llama(weights)  # 先将Mistral格式重映射为Llama格式
        )

    def _remap_mistral_to_llama(  # 将Mistral原生格式权重名称重映射为HF/Llama格式
        self, weights: Iterable[Tuple[str, torch.Tensor]]  # 权重迭代器
    ) -> Iterable[Tuple[str, torch.Tensor]]:  # 返回重映射后的权重迭代器
        for name, loaded_weight in weights:  # 遍历所有权重
            if name.startswith("model.") or name.startswith("lm_head."):  # 如果名称已是HF/Llama格式
                yield name, loaded_weight  # 直接产出
                continue  # 跳过后续处理
            for k, v in self.remapping.items():  # 遍历映射规则
                match = re.fullmatch(k, name)  # 尝试完整匹配权重名称
                if match:  # 如果匹配成功
                    name = match.expand(v)  # 展开为映射后的名称
                    break  # 跳出映射规则循环
            else:  # 如果没有任何映射规则匹配
                logger.warning(f"Unrecognized weight: {name}. Skipping.")  # 记录无法识别的权重并跳过
                continue  # 跳过此权重
            if name.endswith(".qscale_act"):  # 如果名称以.qscale_act结尾
                name = re.sub(r"\.qscale_act$", ".input_scale", name)  # 替换为.input_scale
            elif name.endswith(".qscale_weight"):  # 如果名称以.qscale_weight结尾
                name = re.sub(r"\.qscale_weight$", ".weight_scale", name)  # 替换为.weight_scale
            yield name, loaded_weight  # 产出重映射后的权重名称和张量


EntryClass = [MistralForCausalLMEagle]  # 模型注册入口类列表
