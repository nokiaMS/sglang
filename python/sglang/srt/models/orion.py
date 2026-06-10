# Orion-14B模型推理实现文件
# 本文件实现了兼容HuggingFace权重的Orion-14B大语言模型推理架构
# 包含MLP、注意力层、解码器层、模型主体及因果语言模型等组件

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/OrionStarAI/Orion-14B-Base/blob/main/modeling_orion.py
# Copyright (c) OrionStar Inc.
# LICENSE: https://huggingface.co/OrionStarAI/Orion-14B-Base/blob/main/LICENSE
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/orion.py
"""Inference-only Orion-14B model compatible with HuggingFace weights."""  # 仅推理的Orion-14B模型，兼容HuggingFace权重

from collections.abc import Iterable  # 导入可迭代类型
from typing import Any, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.distributed.parallel_state import get_pp_group  # 导入获取流水线并行组的函数
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.linear import (  # 导入并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和流水线代理张量
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加和层创建工具
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入获取RoPE配置的工具


class OrionMLP(nn.Module):  # Orion模型的MLP（多层感知机）模块
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数名称
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀，用于命名
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并的并行线性层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小列表，两个中间层大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 添加前缀
        )
        self.down_proj = RowParallelLinear(  # 下投影行并行线性层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 添加前缀
        )
        if hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. "  # 不支持的激活函数
                "Only silu is supported for now."  # 目前仅支持silu
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数

    def forward(self, x):  # 前向传播函数，执行MLP计算
        gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影层，获取门控和上投影结果
        x = self.act_fn(gate_up)  # 应用SiLU激活和门控乘法
        x, _ = self.down_proj(x)  # 通过下投影层
        return x  # 返回输出


class OrionAttention(nn.Module):  # Orion模型的注意力模块
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        rope_theta: float = 10000,  # RoPE的theta参数
        rope_scaling: Optional[dict[str, Any]] = None,  # RoPE缩放配置
        max_position_embeddings: int = 8192,  # 最大位置嵌入数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        layer_id: int = 0,  # 层ID
        prefix: str = "",  # 前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = num_heads  # 保存总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个TP秩的注意力头数
        self.total_num_kv_heads = num_kv_heads  # 保存总KV头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于TP大小
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个TP秩的KV头数
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.rope_theta = rope_theta  # 保存RoPE theta
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置嵌入数

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影层
            hidden_size,  # 输入大小
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 添加前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影行并行线性层
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 添加前缀
        )

        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基础theta
            rope_scaling=rope_scaling,  # RoPE缩放配置
        )
        self.attn = RadixAttention(  # 基数注意力模块
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
        )

    def forward(  # 前向传播函数，执行注意力计算
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影获取查询、键、值
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割QKV
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class OrionDecoderLayer(nn.Module):  # Orion模型的解码器层
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 获取最大位置嵌入数
        self.self_attn = OrionAttention(  # 自注意力模块
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            rope_theta=rope_theta,  # RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
            layer_id=layer_id,  # 层ID
        )
        self.mlp = OrionMLP(  # MLP模块
            hidden_size=self.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 添加前缀
        )
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = nn.LayerNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏层大小和eps
        )

    def forward(  # 前向传播函数，执行解码器层计算
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        # Self Attention  # 自注意力部分
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        hidden_states = self.self_attn(  # 通过自注意力层
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = residual + hidden_states  # 残差连接

        # Fully Connected  # 全连接部分
        residual = hidden_states  # 保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP
        hidden_states = residual + hidden_states  # 残差连接
        return hidden_states  # 返回隐藏状态


class OrionModel(nn.Module):  # Orion模型主体
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            self.embed_tokens = VocabParallelEmbedding(  # 词表嵌入层
                config.vocab_size, config.hidden_size  # 词表大小和隐藏层大小
            )
        else:  # 否则
            self.embed_tokens = PPMissingLayer()  # 使用缺失层占位

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 隐藏层数量
            lambda idx, prefix: OrionDecoderLayer(  # 解码器层构造函数
                config, layer_id=idx, quant_config=quant_config, prefix=prefix  # 传入配置、层ID、量化配置和前缀
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行大小
            prefix=add_prefix("layers", prefix),  # 添加前缀
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化
        else:  # 否则
            self.norm = PPMissingLayer()  # 使用缺失层占位

    def forward(  # 前向传播函数，执行模型主体计算
        self,
        input_ids: torch.Tensor,  # 输入ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，可选
    ):
        if self.pp_group.is_first_rank:  # 如果是第一个秩
            if inputs_embeds is not None:  # 如果提供了输入嵌入
                hidden_states = inputs_embeds  # 使用输入嵌入
            else:  # 否则
                hidden_states = self.embed_tokens(input_ids)  # 通过词表嵌入层
        else:  # 否则
            assert pp_proxy_tensors is not None  # 断言代理张量不为空
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态

        for i in range(self.start_layer, self.end_layer):  # 遍历解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states = layer(positions, hidden_states, forward_batch)  # 通过当前层

        if not self.pp_group.is_last_rank:  # 如果不是最后一个秩
            return PPProxyTensors({"hidden_states": hidden_states})  # 返回代理张量

        hidden_states = self.norm(hidden_states)  # 应用最终层归一化
        return hidden_states  # 返回隐藏状态


class OrionForCausalLM(nn.Module):  # Orion因果语言模型
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.model = OrionModel(  # Orion模型主体
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传入配置
        )

        if self.pp_group.is_last_rank:  # 如果是最后一个秩
            self.lm_head = ParallelLMHead(  # 语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 添加前缀
            )
            if self.config.tie_word_embeddings and self.pp_group.is_first_rank:  # 如果绑定词嵌入且是第一个秩
                self.lm_head.weight = self.model.embed_tokens.weight  # 绑定权重
            self.logits_processor = LogitsProcessor(config)  # logits处理器
        else:  # 否则
            self.lm_head = PPMissingLayer()  # 使用缺失层占位

    def forward(  # 前向传播函数，执行因果语言模型计算
        self,
        input_ids: torch.Tensor,  # 输入ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(  # 通过模型主体
            input_ids=input_ids,  # 输入ID
            positions=positions,  # 位置
            forward_batch=forward_batch,  # 前向批次
            inputs_embeds=inputs_embeds,  # 输入嵌入
        )

        if self.pp_group.is_last_rank:  # 如果是最后一个秩
            logits = self.logits_processor(  # 通过logits处理器
                input_ids, hidden_states, self.lm_head, forward_batch  # 传入参数
            )
            return logits  # 返回logits
        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q投影映射
            ("qkv_proj", "k_proj", "k"),  # K投影映射
            ("qkv_proj", "v_proj", "v"),  # V投影映射
            ("gate_up_proj", "gate_proj", 0),  # 门控投影映射
            ("gate_up_proj", "up_proj", 1),  # 上投影映射
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码的逆频率
                continue  # 跳过

            is_packed = False  # 是否已打包标志
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 继续
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                if name not in params_dict:  # 如果参数名不在参数字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                is_packed = True  # 标记为已打包
                break  # 跳出循环
            if is_packed:  # 如果已打包
                continue  # 继续下一个权重

            # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
            if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                continue  # 跳过
            if name not in params_dict:  # 如果参数名不在参数字典中
                continue  # 跳过
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重


EntryClass = OrionForCausalLM  # 入口类为OrionForCausalLM
