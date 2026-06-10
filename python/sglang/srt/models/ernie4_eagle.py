# 文件说明：Ernie4.5 MTP（多Token预测）模型实现，兼容baidu/ERNIE-4.5-*-PT权重
# 基于EAGLE推测解码方法，实现多Token预测以提高推理效率

# Copyright 2023-2025 SGLang Team
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
# limitations under the License. # 详见License限制条件
# ==============================================================================

"""Ernie4.5 MTP model compatible with baidu/ERNIE-4.5-*-PT weights.""" # Ernie4.5 MTP模型，兼容百度ERNIE-4.5-*-PT权重

from typing import Iterable, Optional, Tuple # 导入类型提示工具

import torch # 导入PyTorch库
from torch import nn # 导入神经网络模块
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import (
    Ernie4_5_MoeConfig, # 导入Ernie4.5 MoE配置类
)

from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead, # 导入并行语言模型头
    VocabParallelEmbedding, # 导入词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.ernie4 import Ernie4_5_ForCausalLM, Ernie4DecoderLayer # 导入Ernie4因果语言模型和解码器层
from sglang.srt.utils import add_prefix # 导入添加前缀工具函数


class Ernie4ModelMTP(nn.Module): # Ernie4 MTP模型，实现多Token预测
    def __init__( # Ernie4 MTP模型初始化
        self,
        config: Ernie4_5_MoeConfig, # Ernie4.5 MoE配置
        layer_id: int, # 层ID
        prefix: str, # 参数前缀
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
    ) -> None:
        super().__init__() # 调用父类初始化

        self.embed_tokens = VocabParallelEmbedding( # 创建词表并行嵌入层
            config.vocab_size, # 词表大小
            config.hidden_size, # 隐藏维度
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("embed_tokens", prefix), # 参数前缀
        )
        self.mtp_emb_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # MTP嵌入归一化
        self.mtp_hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # MTP隐藏状态归一化
        self.mtp_linear_proj = nn.Linear( # MTP线性投影层，将嵌入和隐藏状态融合
            config.hidden_size * 2, config.hidden_size, bias=config.use_bias # 输入维度为2倍隐藏维度，输出为隐藏维度
        )
        self.mtp_block = Ernie4DecoderLayer( # MTP解码器块
            config=config, # 模型配置
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("mtp_block", prefix), # 参数前缀
            is_mtp=True, # 标记为MTP层（不使用MoE）
        )

    def forward( # Ernie4 MTP模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
    ) -> torch.Tensor: # 返回隐藏状态
        if input_embeds is None: # 无输入嵌入
            hidden_states = self.embed_tokens(input_ids) # 通过嵌入层获取隐藏状态
        else: # 有输入嵌入
            hidden_states = input_embeds # 直接使用输入嵌入
        # masking inputs at position 0, as not needed by MTP # 将位置0的输入遮蔽，因为MTP不需要
        hidden_states[positions == 0] = 0 # 将位置0的隐藏状态置零

        hidden_states = self.mtp_linear_proj( # 线性投影融合嵌入和主模型隐藏状态
            torch.cat(
                (
                    self.mtp_emb_norm(hidden_states), # 归一化后的MTP嵌入
                    self.mtp_hidden_norm(forward_batch.spec_info.hidden_states), # 归一化后的主模型隐藏状态
                ),
                dim=-1, # 在最后一维拼接
            )
        )
        residual = None # 初始化残差为None
        hidden_states, residual = self.mtp_block( # 通过MTP解码器块
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=residual,
        )
        hidden_states = residual + hidden_states # 残差连接
        return hidden_states # 返回隐藏状态


class Ernie4_5_MoeForCausalLMMTP(nn.Module): # Ernie4.5 MoE因果语言模型MTP版本，用于多Token预测
    def __init__( # Ernie4.5 MoE MTP模型初始化
        self,
        config: Ernie4_5_MoeConfig, # Ernie4.5 MoE配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
        mtp_layer_id: int = 0, # MTP层ID
    ) -> None:
        nn.Module.__init__(self) # 调用nn.Module初始化
        self.config = config # 保存配置
        self.mtp_layer_id = mtp_layer_id # 保存MTP层ID

        self.model = Ernie4ModelMTP( # 创建MTP模型
            config=config,
            layer_id=self.mtp_layer_id,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings: # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens # 语言模型头共享嵌入层
        else: # 否则
            self.lm_head = ParallelLMHead( # 创建并行语言模型头
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏维度
                quant_config=quant_config, # 量化配置
                prefix="lm_head", # 参数前缀
            )
        self.logits_processor = LogitsProcessor(config) # 创建logits处理器

    @torch.no_grad() # 禁用梯度计算
    def forward( # Ernie4.5 MoE MTP模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
    ) -> torch.Tensor: # 返回logits
        hidden_states = self.model(input_ids, positions, forward_batch) # 获取MTP模型隐藏状态
        return self.logits_processor( # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载MTP权重
        mtp_layer_found = False # 标记是否找到MTP层
        mtp_weight_patterns = [ # MTP权重模式列表
            f"mtp_block.{self.mtp_layer_id}", # MTP块模式
            f"mtp_emb_norm.{self.mtp_layer_id}", # MTP嵌入归一化模式
            f"mtp_hidden_norm.{self.mtp_layer_id}", # MTP隐藏归一化模式
            f"mtp_linear_proj.{self.mtp_layer_id}", # MTP线性投影模式
        ]
        params_dict = dict(self.named_parameters()) # 获取参数字典
        for name, loaded_weight in weights: # 遍历权重
            # Only name matched patterns should be loaded # 仅加载匹配模式的权重
            for layer_pattern in mtp_weight_patterns: # 遍历MTP权重模式
                if layer_pattern in name: # 名称匹配模式
                    mtp_layer_found = True # 标记找到MTP层
                    break
            else: # 不匹配任何模式则跳过
                continue
            # But strip mtp_layer_id before loading, because each MTP layer is a MTP model. # 加载前需去掉mtp_layer_id，因为每个MTP层是一个MTP模型
            name = name.replace(f".{self.mtp_layer_id}.", ".") # 去掉层ID后缀
            for ( # 遍历堆叠参数映射
                param_name,
                weight_name,
                shard_id,
            ) in Ernie4_5_ForCausalLM.stacked_params_mapping:
                if weight_name not in name: # 权重名不匹配则跳过
                    continue
                name = name.replace(weight_name, param_name) # 替换为堆叠参数名
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break
            else: # 非堆叠参数
                if name in params_dict.keys(): # 参数名存在于参数字典
                    param = params_dict[name] # 获取参数
                    weight_loader = getattr( # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight) # 加载权重
                else:
                    raise KeyError(f"Parameter '{name}' not found in MTP model.") # 参数未找到则抛出异常
        if not mtp_layer_found: # 未找到MTP层
            raise KeyError( # 抛出异常
                f"MTP layers 'mtp_*.{self.mtp_layer_id}.*' not found in weights."
            )

    def get_embed_and_head(self): # 获取嵌入层和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head): # 设置嵌入层和语言模型头权重
        del self.model.embed_tokens.weight # 删除原有嵌入权重
        self.model.embed_tokens.weight = embed # 设置新的嵌入权重
        if self.config.tie_word_embeddings: # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens # 语言模型头共享嵌入层
        else: # 否则
            del self.lm_head.weight # 删除原有语言模型头权重
            self.lm_head.weight = head # 设置新的语言模型头权重
        torch.cuda.empty_cache() # 清空CUDA缓存
        torch.cuda.synchronize() # 同步CUDA操作


EntryClass = [Ernie4_5_MoeForCausalLMMTP] # 模型入口类列表
