# 百灵MoE NextN（多Token预测）模型推理实现
# 该文件实现了BailingMoE的NextN预测模块，用于推测解码（speculative decoding）
# 主要特点包括：
# - 支持标准MoE和混合线性注意力两种架构
# - 通过eh_proj合并当前隐藏状态和推测信息
# - 支持enorm/hnorm归一化和最终层归一化
# - 复用基础模型的权重加载逻辑
# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved. # 版权归属Antgroup和HuggingFace
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX # 本代码基于EleutherAI的GPT-NeoX库
# and OPT implementations in this library. It has been modified from its # 和OPT实现，已从原始形式修改
# original forms to accommodate minor architectural differences compared # 以适应与GPT-NeoX和OPT的轻微架构差异
# to GPT-NeoX and OPT used by the Meta AI team that trained the model. # 这些差异由训练模型的Meta AI团队引入
#
# Licensed under the Apache License, Version 2.0 (the "License"); # 许可证：Apache 2.0
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at # 您可以在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不附带任何明示或暗示的保证
# See the License for the specific language governing permissions and # 请参阅许可证以了解管理权限和
# limitations under the License. # 限制的具体条款
"""SGLang BailingMoENextN model.""" # SGLang百灵MoE NextN模型

import logging # 导入日志模块
from typing import Iterable, Optional, Tuple # 导入类型提示

import torch # 导入PyTorch
from torch import nn # 导入神经网络模块
from transformers import PretrainedConfig # 导入预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size # 导入TP世界大小获取
from sglang.srt.layers.dp_attention import is_dp_attention_enabled # 导入DP注意力启用检测
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS层归一化
from sglang.srt.layers.linear import ReplicatedLinear # 导入复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息
from sglang.srt.models.bailing_moe import BailingMoEBlock, BailingMoEForCausalLM # 导入百灵MoE块和因果语言模型
from sglang.srt.models.bailing_moe_linear import ( # 导入百灵MoE线性模块
    BailingMoELinearDecoderLayer, # 线性解码器层
    BailingMoeV2_5ForCausalLM, # V2.5因果语言模型
)
from sglang.srt.models.utils import WeightsMapper # 导入权重映射器
from sglang.srt.server_args import get_global_server_args # 导入全局服务器参数
from sglang.srt.utils import BumpAllocator, add_prefix # 导入凸起分配器和前缀添加工具

LoraConfig = None # LoRA配置初始化为空
logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


class BailingMoEModelNextN(nn.Module): # 百灵MoE NextN模型
    def __init__( # NextN模型初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.layer_group_size = 1 # 层分组大小为1
        self.start_layer = 0 # 起始层为0
        self.end_layer = 1 # 结束层为1
        self.total_num_layers = 1 # 总层数为1
        self.vocab_size = config.vocab_size # 词表大小
        config.for_nextn_model = True # 标记为NextN模型

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4": # 如果是modelopt_fp4量化
            logger.warning( # 记录警告
                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model." # 覆盖modelopt_fp4 Deepseek模型的量化配置
            )
            quant_config = None # 重置量化配置

        self.vocab_size = config.vocab_size # 词表大小

        self.word_embeddings = VocabParallelEmbedding( # 词嵌入层
            config.vocab_size, # 词表大小
            config.hidden_size, # 隐藏层大小
            enable_tp=not is_dp_attention_enabled(), # 是否启用TP
            prefix=add_prefix("word_embeddings", prefix), # 参数前缀
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 嵌入归一化层
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 隐藏状态归一化层

        self.eh_proj = ReplicatedLinear( # eh投影层（合并嵌入和隐藏状态）
            2 * config.hidden_size, # 输入维度（两倍隐藏大小）
            config.hidden_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix(f"layers.{config.num_hidden_layers}.eh_proj", prefix), # 参数前缀
        )

        self.is_hybrid = ( # 是否为混合架构
            hasattr(config, "model_type") and config.model_type == "bailing_hybrid"
        )
        if self.is_hybrid: # 如果是混合架构
            config.attention_type = 1 # 设置注意力类型为softmax
            self.decoder = BailingMoELinearDecoderLayer( # 线性解码器层
                config,
                quant_config=quant_config,
                layer_id=0, # 层ID为0
                is_nextn=True, # 标记为NextN层
                prefix=add_prefix(f"layers.{config.num_hidden_layers}", prefix), # 参数前缀
            )
        else: # 否则使用标准MoE架构
            self.decoder = BailingMoEBlock( # 标准MoE解码器块
                config,
                0, # 层ID为0
                quant_config=quant_config,
                # is_nextn=True, # 注释掉的nextn标记
                prefix=add_prefix("decoder", prefix), # 参数前缀
            )

        self.shared_head = nn.Module() # 共享头（占位模块）
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 最终层归一化

    def forward( # NextN模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        input_embeds: torch.Tensor = None, # 输入嵌入
    ) -> torch.Tensor:

        if input_embeds is None: # 如果没有输入嵌入
            hidden_states = self.word_embeddings(input_ids) # 通过词嵌入获取隐藏状态
        else: # 否则
            hidden_states = input_embeds # 直接使用输入嵌入

        if hidden_states.shape[0] > 0: # 如果有token
            hidden_states, _ = self.eh_proj( # 通过eh投影合并嵌入和推测信息
                torch.cat(
                    (
                        self.enorm(hidden_states), # 归一化后的嵌入
                        self.hnorm( # 归一化后的推测隐藏状态
                            forward_batch.spec_info.hidden_states.to(
                                self.hnorm.weight.dtype # 转换为归一化权重的数据类型
                            )
                        ),
                    ),
                    dim=-1, # 在最后一维拼接
                )
            )

        residual = None # 初始化残差为空
        if self.is_hybrid: # 如果是混合架构
            device = input_ids.device # 获取设备
            zero_allocator = BumpAllocator( # 零分配器（用于MLA）
                buffer_size=self.total_num_layers # 缓冲区大小
                * 2
                * (2 if forward_batch.can_run_tbo else 1), # TBO模式时翻倍
                dtype=torch.float32, # FP32精度
                device=device, # 设备
            )
            hidden_states, residual = self.decoder( # 线性解码器前向传播
                hidden_states=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                residual=residual,
                zero_allocator=zero_allocator,
            )
        else: # 否则使用标准MoE解码器
            hidden_states, residual = self.decoder( # 标准解码器前向传播
                positions, hidden_states, forward_batch, residual
            )

        if not forward_batch.forward_mode.is_idle(): # 如果不是空闲模式
            if residual is not None: # 如果有残差
                hidden_states, _ = self.final_layernorm(hidden_states, residual) # 带残差的最终层归一化
            else: # 否则
                hidden_states = self.final_layernorm(hidden_states) # 最终层归一化

        return hidden_states # 返回隐藏状态


class BailingMoeForCausalLMNextN(nn.Module): # 百灵MoE NextN因果语言模型

    packed_modules_mapping = { # 打包模块映射
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"], # 融合QKV投影映射
        "gate_up_proj": ["gate_proj", "up_proj"], # gate和up投影映射
    }
    # To ensure correct weight loading and mapping. # 确保正确的权重加载和映射
    hf_to_sglang_mapper = WeightsMapper( # HuggingFace到SGLang权重映射器
        orig_to_new_substr={
            "attention.dense": "attention.o_proj", # dense到o_proj
        },
    )

    def __init__( # NextN因果语言模型初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        nn.Module.__init__(self) # 调用nn.Module初始化
        self.config = config # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size() # 获取TP大小
        self.quant_config = quant_config # 保存量化配置
        if hasattr(self, "determine_num_fused_shared_experts"): # 如果有确定融合共享专家数的方法
            # Asystem has determine_num_fused_shared_experts but theta does not. # Asystem有该方法但theta没有
            self.determine_num_fused_shared_experts("BailingMoeForCausalLMNextN") # 确定融合共享专家数

        self.model = BailingMoEModelNextN( # NextN模型
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead( # 语言模型头
            config.vocab_size, # 词表大小
            config.hidden_size, # 隐藏层大小
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("model.shared_head.head", prefix), # 参数前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head, # 是否使用注意力TP组
        )
        self.logits_processor = LogitsProcessor(config) # logits处理器
        if hasattr(self.config, "model_type") and config.model_type == "bailing_hybrid": # 如果是混合架构
            self.base_load_weights_func = BailingMoeV2_5ForCausalLM.load_weights # 使用V2.5的权重加载
            self.post_load_weights_func = BailingMoeV2_5ForCausalLM.post_load_weights # 使用V2.5的后处理
        else: # 否则
            self.base_load_weights_func = BailingMoEForCausalLM.load_weights # 使用标准MoE的权重加载
            # V1 BailingMoeAttention is standard QKV (no kv_b_proj), no fixup needed. # V1百灵注意力是标准QKV（无kv_b_proj），无需修复
            self.post_load_weights_func = None # 无需后处理

    @torch.no_grad() # 禁用梯度计算
    def forward( # NextN因果语言模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch) # 模型前向传播
        return self.logits_processor( # 通过logits处理器返回
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def set_embed_and_head(self, embed, head): # 设置嵌入和语言模型头权重
        """Used by the eagle_worker.""" # 由eagle_worker使用
        del self.model.word_embeddings.weight # 删除旧嵌入权重
        del self.lm_head.weight # 删除旧头权重
        self.model.word_embeddings.weight = embed # 设置新嵌入权重
        self.lm_head.weight = head # 设置新头权重
        torch.cuda.empty_cache() # 清空CUDA缓存
        torch.cuda.synchronize() # 同步CUDA

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载权重
        self.base_load_weights_func(self, weights, is_nextn=True) # 调用基础模型的权重加载方法

    def post_load_weights(self, is_nextn=True, weight_names=None): # 权重加载后处理
        # `is_nextn` is pinned to True for the NextN subclass; the parameter is kept # 对于NextN子类，is_nextn固定为True；保留该参数
        # only because the underlying `load_weights` flow calls `self.post_load_weights` # 仅因为底层的load_weights流程调用self.post_load_weights
        # with `is_nextn=...` as a kwarg. # 时会传入is_nextn=...作为关键字参数
        if self.post_load_weights_func is None: # 如果无需后处理
            return # 直接返回
        self.post_load_weights_func(self, is_nextn=True, weight_names=weight_names) # 调用后处理函数


EntryClass = [BailingMoeForCausalLMNextN] # 入口类列表
