# Mistral Large 3 EAGLE推测解码模型实现
# 本文件实现了Mistral Large 3的EAGLE推测解码草稿模型，
# 在DeepseekV2架构基础上增加了fc融合层，将词嵌入与目标模型隐藏状态拼接后投影，
# 用于加速Mistral Large 3 MLA模型的推理。

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral_large_3_eagle.py  # 适配自vLLM项目的Mistral Large 3 EAGLE实现
# SPDX-License-Identifier: Apache-2.0  # Apache 2.0许可证
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # vLLM项目贡献者版权声明
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.configs.model_config import is_deepseek_dsa  # 导入Deepseek DSA判断函数
from sglang.srt.distributed import get_pp_group  # 导入流水线并行组获取函数
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp  # 导入DSA预填充上下文并行判断函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import RowParallelLinear  # 导入行并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled  # 导入预填充上下文并行启用判断函数
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和流水线代理张量
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2Model  # 导入DeepseekV2解码器层和模型
from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM  # 导入Mistral Large 3因果语言模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具函数


class MistralLarge3EagleModel(DeepseekV2Model):  # Mistral Large 3 EAGLE草稿模型，继承自DeepseekV2Model
    """EAGLE draft model with an fc layer that fuses token embeddings and  # 带有fc层的EAGLE草稿模型，融合词嵌入和
    target-model hidden states before passing through transformer layers."""  # 目标模型隐藏状态后再通过transformer层

    def __init__(  # 初始化方法
        self,  # 自身实例
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认无
        prefix: str = "",  # 前缀字符串，默认空
    ):
        nn.Module.__init__(self)  # 调用nn.Module的初始化

        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 保存词表大小
        assert get_pp_group().world_size == 1  # 断言流水线并行世界大小为1（不支持PP）
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()  # 判断是否启用DSA预填充上下文并行
        self.mla_enable_prefill_cp = (  # 判断是否启用MLA预填充上下文并行
            is_prefill_context_parallel_enabled() and not is_deepseek_dsa(config)  # 启用上下文并行且非Deepseek DSA
        )

        self.embed_tokens = VocabParallelEmbedding(  # 词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏维度
            prefix=add_prefix("embed_tokens", prefix),  # 带前缀的嵌入层名称
        )

        self.layers = nn.ModuleList(  # 解码器层列表
            [
                DeepseekV2DecoderLayer(  # DeepseekV2解码器层
                    config=config,  # 配置
                    prefix=add_prefix(prefix, f"layers.{i}"),  # 带前缀的层名称
                    quant_config=quant_config,  # 量化配置
                    layer_id=i,  # 层ID
                    dsa_enable_prefill_cp=self.dsa_enable_prefill_cp,  # DSA预填充上下文并行标志
                    mla_enable_prefill_cp=self.mla_enable_prefill_cp,  # MLA预填充上下文并行标志
                )
                for i in range(self.config.num_hidden_layers)  # 遍历隐藏层数量
            ]
        )
        self.start_layer = 0  # 起始层索引
        self.end_layer = self.config.num_hidden_layers  # 结束层索引

        self.fc = RowParallelLinear(  # EAGLE融合层：将词嵌入和目标模型隐藏状态拼接后投影
            self.config.hidden_size * 2,  # 输入维度为隐藏维度的两倍（嵌入+隐藏状态拼接）
            self.config.hidden_size,  # 输出维度为隐藏维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix(prefix, "fc"),  # 带前缀的fc层名称
            input_is_parallel=False,  # 输入不并行
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终RMS归一化层
        self.layers_to_capture = []  # 需要捕获隐藏状态的层列表
        self.llama_4_scaling_config = getattr(config, "llama_4_scaling", None)  # Llama 4缩放配置

    def forward(  # 前向传播方法
        self,  # 自身实例
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，默认无
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，默认无
    ) -> torch.Tensor:  # 返回隐藏状态张量
        if input_embeds is None:  # 如果没有提供输入嵌入
            input_embeds = self.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入
        input_embeds, _ = self.fc(  # 通过fc融合层将嵌入与目标模型隐藏状态拼接后投影
            torch.cat((input_embeds, forward_batch.spec_info.hidden_states), dim=-1)  # 在最后一个维度上拼接嵌入和目标模型隐藏状态
        )
        output = super().forward(  # 调用父类DeepseekV2Model的前向传播
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors  # 传递所有参数
        )
        assert isinstance(output, torch.Tensor)  # 断言输出是张量
        return output  # 返回输出


class MistralLarge3ForCausalLMEagle(MistralLarge3ForCausalLM):  # Mistral Large 3 EAGLE因果语言模型，继承自MistralLarge3ForCausalLM
    remapping = MistralLarge3ForCausalLM.remapping | {  # 合并父类映射规则和EAGLE专用映射规则
        r"eagle_linear\.weight": r"model.fc.weight",  # EAGLE线性层权重映射到fc权重
        r"eagle_linear\.qscale_act": r"model.fc.input_scale",  # EAGLE线性层激活缩放映射到fc输入缩放
        r"eagle_linear\.qscale_weight": r"model.fc.weight_scale",  # EAGLE线性层权重缩放映射到fc权重缩放
    }

    def __init__(  # 初始化方法
        self,  # 自身实例
        *,  # 强制关键字参数
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认无
        prefix: str = "",  # 前缀字符串，默认空
    ):
        # DeepseekV2ForCausalLM.__init__ hardcodes self.model = DeepseekV2Model.  # DeepseekV2ForCausalLM的初始化硬编码了self.model为DeepseekV2Model
        # We let the parent init run (it sets up weight loading attrs, lm_head,  # 让父类初始化运行（设置权重加载属性、lm_head等），
        # etc.), then replace self.model with MistralLarge3EagleModel which has  # 然后用包含EAGLE fc层的MistralLarge3EagleModel替换self.model
        # the EAGLE fc layer. The discarded 2-layer DeepseekV2Model is tiny.  # 被丢弃的2层DeepseekV2Model很小
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)  # 调用父类初始化
        self.model = MistralLarge3EagleModel(  # 用EAGLE草稿模型替换默认模型
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传递配置、量化配置和前缀
        )


EntryClass = [MistralLarge3ForCausalLMEagle]  # 模型注册入口类列表
