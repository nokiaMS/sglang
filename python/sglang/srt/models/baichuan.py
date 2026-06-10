# 百川（BaiChuan）模型推理实现，兼容HuggingFace权重格式
# 该文件实现了BaiChuan模型的推理专用版本，支持ALIBI和ROPE两种位置编码
# 包括BaiChuan 13B（ALIBI）和BaiChuan2 7B/13B（ROPE）两种变体
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project # 版权归属vLLM项目贡献者
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/baichuan.py # 改编自vLLM项目

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved. # 版权归属EleutherAI和HuggingFace
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
"""Inference-only BaiChuan model compatible with HuggingFace weights.""" # 仅推理的百川模型，兼容HuggingFace权重

import math # 导入数学模块
from typing import Iterable, Optional, Tuple # 导入类型提示

import torch # 导入PyTorch
from torch import nn # 导入神经网络模块
from transformers import PretrainedConfig # 导入预训练配置类

from sglang.srt.distributed import ( # 导入分布式相关模块
    get_tensor_model_parallel_rank, # 获取张量并行排名
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
)
from sglang.srt.layers.activation import SiluAndMul # 导入SiLU与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS层归一化
from sglang.srt.layers.linear import ( # 导入线性层
    MergedColumnParallelLinear, # 合并列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope # 导入旋转位置编码获取工具
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.utils import add_prefix, is_npu # 导入前缀添加和NPU检测工具
from sglang.srt.utils.hf_transformers_utils import get_rope_config # 导入ROPE配置获取工具

_is_npu = is_npu() # 检测是否为NPU环境


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor: # 获取ALIBI斜率
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads)) # 找到最接近的2的幂
    base = torch.tensor( # 计算基数
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32) # 生成幂次
    slopes = torch.pow(base, powers) # 计算斜率

    if closest_power_of_2 != total_num_heads: # 如果头数不是2的幂
        extra_base = torch.tensor( # 计算额外基数
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min( # 计算剩余头数
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange( # 生成额外幂次
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0) # 拼接额外斜率
    return slopes # 返回斜率


class BaiChuanMLP(nn.Module): # 百川MLP模块

    def __init__( # MLP初始化方法
        self,
        hidden_size: int, # 隐藏层大小
        intermediate_size: int, # 中间层大小
        hidden_act: str, # 隐藏层激活函数名称
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear( # 合并的gate和up投影
            hidden_size, # 输入维度
            [intermediate_size] * 2, # 输出维度（gate和up各一份）
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("gate_up_proj", prefix), # 参数前缀
        )
        self.down_proj = RowParallelLinear( # 下投影线性层
            intermediate_size, # 输入维度
            hidden_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("down_proj", prefix), # 参数前缀
        )
        if hidden_act != "silu": # 如果激活函数不是silu
            raise ValueError( # 抛出值错误
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now." # 目前仅支持silu
            )
        self.act_fn = SiluAndMul() # SiLU与乘法激活函数

    def forward(self, x): # MLP前向传播
        gate_up, _ = self.gate_up_proj(x) # gate和up投影
        x = self.act_fn(gate_up) # 应用激活函数
        x, _ = self.down_proj(x) # 下投影
        return x # 返回输出


class BaiChuanAttention(nn.Module): # 百川注意力模块
    """Multi-headed attention from 'Attention Is All You Need' paper""" # 来自"Attention Is All You Need"论文的多头注意力

    def __init__( # 注意力初始化方法
        self,
        hidden_size: int, # 隐藏层大小
        num_heads: int, # 注意力头数
        position_embedding: str, # 位置嵌入类型（ALIBI或ROPE）
        rope_theta: float = 10000, # 旋转位置编码theta
        max_position_embeddings: int = 8192, # 最大位置嵌入数
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        layer_id: int = 0, # 层ID
        dtype: Optional[torch.dtype] = torch.bfloat16, # 数据类型
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.hidden_size = hidden_size # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size() # 获取TP大小
        self.total_num_heads = num_heads # 总注意力头数
        self.total_num_kv_heads = self.total_num_heads # KV头数等于注意力头数（MHA）
        assert self.total_num_heads % tp_size == 0 # 断言总头数可被TP大小整除
        self.head_dim = hidden_size // self.total_num_heads # 头维度
        self.position_embedding = position_embedding # 位置嵌入类型
        self.rope_theta = rope_theta # 旋转theta
        self.max_position_embeddings = max_position_embeddings # 最大位置嵌入数
        if self.total_num_kv_heads >= tp_size: # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition # KV头数大于TP大小，因此分区
            # the KV heads across multiple tensor parallel GPUs. # 将KV头分配到多个TP GPU上
            assert self.total_num_kv_heads % tp_size == 0 # 断言KV头数可被TP大小整除
        else: # 否则
            # Number of KV heads is less than TP size, so we replicate # KV头数小于TP大小，因此复制
            # the KV heads across multiple tensor parallel GPUs. # 将KV头复制到多个TP GPU上
            assert tp_size % self.total_num_kv_heads == 0 # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size) # 每个TP rank的KV头数
        self.num_heads = self.num_kv_heads # 头数等于KV头数

        # pylint: disable=invalid-name # 禁用无效名称检查
        self.W_pack = QKVParallelLinear( # QKV打包线性层（百川特有）
            hidden_size, # 输入维度
            self.head_dim, # 头维度
            self.total_num_heads, # 总Q头数
            self.total_num_heads, # 总KV头数（与Q头数相同）
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("W_pack", prefix), # 参数前缀
        )
        self.o_proj = RowParallelLinear( # 输出投影
            self.total_num_heads * self.head_dim, # 输入维度
            hidden_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("o_proj", prefix), # 参数前缀
        )
        self.scaling = self.head_dim**-0.5 # 缩放因子

        self.attn = RadixAttention( # 基数注意力
            self.num_heads, # 注意力头数
            self.head_dim, # 头维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数前缀
        )

        # Create the alibi slopes and slice them. # 创建ALIBI斜率并切片
        if self.position_embedding == "ALIBI": # 如果使用ALIBI位置编码
            tp_rank = get_tensor_model_parallel_rank() # 获取TP排名
            head_start = tp_rank * self.num_heads # 头起始索引
            head_end = (tp_rank + 1) * self.num_heads # 头结束索引
            alibi_slopes = _get_alibi_slopes(self.total_num_heads) # 获取ALIBI斜率
            alibi_slopes = alibi_slopes[head_start:head_end] # 按TP排名切片
            self.alibi_slopes = torch.tensor( # 转换为张量
                alibi_slopes, dtype=dtype, device="npu" if _is_npu else "cuda"
            )
        else: # 否则使用ROPE
            self.rotary_emb = get_rope( # 获取旋转位置编码
                self.head_dim, # 头维度
                rotary_dim=self.head_dim, # 旋转维度
                max_position=self.max_position_embeddings, # 最大位置
                base=self.rope_theta, # 基础频率
            )

        self.attn_kwargs = {} # 注意力额外参数
        if self.position_embedding == "ALIBI" and _is_npu: # 如果ALIBI且在NPU上
            self.attn_kwargs["slopes"] = self.alibi_slopes # 添加斜率参数

    def forward( # 注意力前向传播
        self,
        positions: torch.Tensor, # 位置张量
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次
    ) -> torch.Tensor:
        qkv, _ = self.W_pack(hidden_states) # QKV打包投影
        q, k, v = qkv.chunk(chunks=3, dim=-1) # 拆分为Q、K、V
        if self.position_embedding != "ALIBI": # 如果不使用ALIBI
            q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch, **self.attn_kwargs) # 计算注意力
        output, _ = self.o_proj(attn_output) # 输出投影
        return output # 返回输出


class BaiChuanDecoderLayer(nn.Module): # 百川解码器层

    def __init__( # 解码器层初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        position_embedding: str, # 位置嵌入类型
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.hidden_size = config.hidden_size # 保存隐藏层大小
        rope_theta, _ = get_rope_config(config) # 获取ROPE配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192) # 获取最大位置嵌入
        self.self_attn = BaiChuanAttention( # 自注意力层
            hidden_size=self.hidden_size, # 隐藏层大小
            num_heads=config.num_attention_heads, # 注意力头数
            position_embedding=position_embedding, # 位置嵌入类型
            rope_theta=rope_theta, # 旋转theta
            layer_id=layer_id, # 层ID
            max_position_embeddings=max_position_embeddings, # 最大位置嵌入
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("self_attn", prefix), # 参数前缀
        )
        self.mlp = BaiChuanMLP( # MLP层
            hidden_size=self.hidden_size, # 隐藏层大小
            intermediate_size=config.intermediate_size, # 中间层大小
            hidden_act=config.hidden_act, # 激活函数
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("mlp", prefix), # 参数前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 输入层归一化
        self.post_attention_layernorm = RMSNorm( # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward( # 解码器层前向传播
        self,
        positions: torch.Tensor, # 位置张量
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次
        residual: Optional[torch.Tensor], # 残差连接
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention # 自注意力
        if residual is None: # 如果没有残差
            residual = hidden_states # 残差等于隐藏状态
            hidden_states = self.input_layernorm(hidden_states) # 层归一化
        else: # 否则
            hidden_states, residual = self.input_layernorm(hidden_states, residual) # 带残差的层归一化
        hidden_states = self.self_attn( # 自注意力计算
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected # 全连接层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual) # 注意力后层归一化
        hidden_states = self.mlp(hidden_states) # MLP前向传播
        return hidden_states, residual # 返回隐藏状态和残差


class BaiChuanModel(nn.Module): # 百川模型主体

    def __init__( # 模型初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        position_embedding: str, # 位置嵌入类型
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.padding_idx = config.pad_token_id # 填充token ID
        self.vocab_size = config.vocab_size # 词表大小

        self.embed_tokens = VocabParallelEmbedding( # 词嵌入层
            config.vocab_size, # 词表大小
            config.hidden_size, # 隐藏层大小
            org_num_embeddings=config.vocab_size, # 原始嵌入数量
            prefix=add_prefix("embed_tokens", prefix), # 参数前缀
        )
        self.layers = nn.ModuleList( # 解码器层列表
            [
                BaiChuanDecoderLayer( # 百川解码器层
                    config,
                    layer_id=i, # 层ID
                    position_embedding=position_embedding, # 位置嵌入类型
                    quant_config=quant_config, # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix), # 参数前缀
                )
                for i in range(config.num_hidden_layers) # 遍历所有层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 最终层归一化

    def forward( # 模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) # 词嵌入
        residual = None # 初始化残差为空
        for i in range(len(self.layers)): # 遍历每一层
            layer = self.layers[i] # 获取当前层
            hidden_states, residual = layer( # 前向传播当前层
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual) # 最终层归一化
        return hidden_states # 返回隐藏状态


class BaiChuanBaseForCausalLM(nn.Module): # 百川因果语言模型基类
    packed_modules_mapping = { # 打包模块映射
        "W_pack": ["W_pack"], # QKV打包映射
        "gate_up_proj": [ # gate和up投影映射
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes # LoRA特定属性
    supported_lora_modules = [ # 支持的LoRA模块
        "W_pack", # QKV打包
        "o_proj", # 输出投影
        "gate_up_proj", # gate和up投影
        "down_proj", # 下投影
    ]
    embedding_modules = { # 嵌入模块
        "embed_tokens": ["embed_tokens"], # 词嵌入
    }
    embedding_padding_modules = [] # 嵌入填充模块

    def __init__( # 因果语言模型初始化方法
        self,
        config: PretrainedConfig, # 模型配置
        position_embedding: str, # 位置嵌入类型
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化

        self.config = config # 保存配置

        self.quant_config = quant_config # 保存量化配置
        self.model = BaiChuanModel( # 百川模型
            config, position_embedding, quant_config, prefix=add_prefix("model", prefix)
        )
        if self.config.tie_word_embeddings: # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens # 语言模型头复用词嵌入
        else: # 否则
            self.lm_head = ParallelLMHead( # 独立的语言模型头
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏层大小
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("lm_head", prefix), # 参数前缀
            )
        self.logits_processor = LogitsProcessor(config) # logits处理器

    def forward( # 因果语言模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch) # 模型前向传播
        return self.logits_processor( # 通过logits处理器返回
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载权重
        stacked_params_mapping = [ # 堆叠参数映射
            # (param_name, shard_name, shard_id) # （参数名，分片名，分片ID）
            ("gate_up_proj", "gate_proj", 0), # gate投影映射
            ("gate_up_proj", "up_proj", 1), # up投影映射
        ]
        params_dict = dict(self.named_parameters()) # 获取参数字典
        for name, loaded_weight in weights: # 遍历所有权重
            if "rotary_emb.inv_freq" in name: # 跳过旋转频率
                continue
            if name == "lm_head.weight": # 如果是语言模型头权重
                # Unlike Baichuan, Baichuan2 normalizes the head weights. # 与BaiChuan不同，BaiChuan2对头权重进行归一化
                # Refer to: # 参考：
                # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/84603cde5ebffb6084e476cfaeceaf0b8b91fe54/modeling_baichuan.py#L508
                # Distinguish between Baichuan and Baichuan2 by checking the # 通过检查词表大小区分BaiChuan和BaiChuan2
                # vocab size. This is suggested by # 这由以下建议
                # https://github.com/vllm-project/vllm/pull/1022#discussion_r1325652704
                is_baichuan2 = self.config.vocab_size == 125696 # 判断是否为BaiChuan2
                if is_baichuan2: # 如果是BaiChuan2
                    loaded_weight = torch.nn.functional.normalize(loaded_weight) # 归一化权重

            for param_name, weight_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射
                if weight_name not in name: # 如果权重名不在参数名中
                    continue # 跳过
                name = name.replace(weight_name, param_name) # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重
                break # 跳出循环
            else: # 如果没有匹配堆叠映射
                # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重


class BaichuanForCausalLM(BaiChuanBaseForCausalLM): # 百川因果语言模型
    """Baichuan 13B and Baichuan2 7B/13B.""" # 百川13B和BaiChuan2 7B/13B

    def __init__( # 百川因果语言模型初始化方法
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        if config.hidden_size == 4096:  # baichuan2 7b # BaiChuan2 7B使用ROPE
            super().__init__(config, "ROPE", quant_config, prefix=prefix) # 使用ROPE位置编码
        else:  # baichuan 13b, baichuan2 13b # 百川13B和BaiChuan2 13B使用ALIBI
            super().__init__(config, "ALIBI", quant_config, prefix=prefix) # 使用ALIBI位置编码


EntryClass = [BaichuanForCausalLM] # 入口类列表
