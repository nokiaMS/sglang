# GLM-4.5/4.6/4.7 MoE 推测解码(NextN)模型文件
# 本文件实现了仅推理模式的 GLM-4.5、GLM-4.6 和 GLM-4.7 MoE 推测解码(NextN)模型，
# 用于加速推理过程中的草稿 token 生成。

# Copyright 2023-2024 SGLang Team # 版权所有 2023-2024 SGLang 团队
# Licensed under the Apache License, Version 2.0 (the "License"); # 根据 Apache 许可证 2.0 版本授权
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件。
# You may obtain a copy of the License at # 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不附带任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and # 请参阅许可证以获取管理权限和
# limitations under the License. # 限制的具体条款
# ==============================================================================

"""Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 Speculative Decoding."""  # 仅推理的 GLM-4.5、GLM-4.6 和 GLM-4.7 推测解码

import logging  # 导入日志模块
from typing import Iterable, Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入全局专家分布记录器
from sglang.srt.layers.dp_attention import is_dp_attention_enabled  # 导入判断是否启用数据并行注意力的函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.glm4_moe import Glm4MoeDecoderLayer, Glm4MoeForCausalLM  # 从 glm4_moe 模型导入解码器层和因果语言模型
from sglang.srt.server_args import get_global_server_args  # 导入获取全局服务器参数的函数
from sglang.srt.utils import add_prefix, is_npu  # 导入前缀添加和 NPU 判断工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class Glm4MoeModelNextN(nn.Module):  # GLM MoE NextN 模型类
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":  # 如果量化配置为 modelopt_fp4
            logger.warning(  # 记录警告
                "Overriding Glm4MoeForCausalLMNextN quant config for modelopt_fp4 GLM-4.5 / GLM-4.6 / GLM-4.7 model."  # 覆盖 modelopt_fp4 的量化配置，适用于 GLM-4.5/4.6/4.7 模型
            )
            quant_config = None  # 将量化配置设为 None

        self.vocab_size = config.vocab_size  # 保存词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 创建词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            use_attn_tp_group=is_dp_attention_enabled(),  # 是否使用注意力张量并行组
            prefix=add_prefix("embed_tokens", prefix),  # 添加嵌入层前缀
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 嵌入归一化层
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 隐藏状态归一化层

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)  # 嵌入与隐藏状态投影层，将拼接后的2倍隐藏维度映射回隐藏维度

        self.decoder = Glm4MoeDecoderLayer(  # 创建 MoE 解码器层
            config,  # 模型配置
            0,  # 层索引为0
            quant_config=quant_config,  # 量化配置
            is_nextn=True,  # 标记为 NextN 层
            prefix=add_prefix("decoder", prefix),  # 添加解码器前缀
        )

        self.shared_head = nn.Module()  # 创建共享头模块
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 共享头的归一化层

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID 张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
    ) -> torch.Tensor:  # 返回隐藏状态张量
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层获取隐藏状态
        else:  # 否则
            hidden_states = input_embeds  # 直接使用输入嵌入作为隐藏状态

        if hidden_states.shape[0] > 0:  # 如果隐藏状态非空
            hidden_states = self.eh_proj(  # 通过 eh_proj 投影层
                torch.cat(  # 拼接嵌入归一化和隐藏状态归一化的结果
                    (
                        self.enorm(hidden_states),  # 对嵌入进行 RMS 归一化
                        self.hnorm(forward_batch.spec_info.hidden_states),  # 对推测信息的隐藏状态进行 RMS 归一化
                    ),
                    dim=-1,  # 在最后一个维度上拼接
                )
            )

        residual = None  # 初始化残差为 None
        with get_global_expert_distribution_recorder().disable_this_region():  # 在禁用专家分布记录的区域内执行
            hidden_states, residual = self.decoder(  # 通过解码器层
                positions, hidden_states, forward_batch, residual  # 传入位置、隐藏状态、批次信息和残差
            )

        if not forward_batch.forward_mode.is_idle():  # 如果前向模式不是空闲
            if residual is not None:  # 如果残差存在
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)  # 共享头归一化，融合残差
            else:  # 否则
                hidden_states = self.shared_head.norm(hidden_states)  # 仅做共享头归一化

        return hidden_states  # 返回隐藏状态


class Glm4MoeForCausalLMNextN(Glm4MoeForCausalLM):  # GLM MoE 因果语言模型 NextN 类，继承自 Glm4MoeForCausalLM
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
    ) -> None:
        nn.Module.__init__(self)  # 调用 nn.Module 的初始化
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        if (  # 如果
            is_npu()  # 是 NPU 设备
            and get_global_server_args().speculative_draft_model_quantization is None  # 且推测草稿模型量化为 None
        ):
            quant_config = None  # 将量化配置设为 None
        self.quant_config = quant_config  # 保存量化配置

        self.model = Glm4MoeModelNextN(  # 创建 NextN 模型
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        self.lm_head = ParallelLMHead(  # 创建并行语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model.shared_head.head", prefix),  # 添加共享头前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力张量并行组
        )
        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器

        self.num_fused_shared_experts = (  # 融合共享专家数量
            0 if get_global_server_args().disable_shared_experts_fusion else 1  # 如果禁用共享专家融合则为0，否则为1
        )

    @torch.no_grad()  # 禁用梯度计算装饰器
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID 张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:  # 返回 logits 张量
        hidden_states = self.model(input_ids, positions, forward_batch)  # 通过 NextN 模型获取隐藏状态
        return self.logits_processor(  # 通过 logits 处理器计算并返回 logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 传入输入ID、隐藏状态、语言模型头和批次信息
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        super().load_weights(weights, is_nextn=True)  # 调用父类的加载权重方法，标记为 NextN


EntryClass = [Glm4MoeForCausalLMNextN]  # 入口类列表，用于模型注册
