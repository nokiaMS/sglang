# Apertus模型推理实现模块
# 实现仅推理的Apertus模型，兼容HuggingFace权重格式，
# 支持Q/K RMSNorm归一化、xIELU激活函数、流水线并行和KV缓存量化缩放

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The SwissAI Initiative
# Copyright 2023-2024 SGLang Team
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
# limitations under the License.
# ==============================================================================

# Adapted from
# 改编自
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only Apertus model compatible with HuggingFace weights."""  # 仅推理的Apertus模型，兼容HuggingFace权重

import logging  # 导入日志模块
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 从torch导入神经网络模块
from transformers import ApertusConfig  # 从transformers导入Apertus配置

from sglang.srt.distributed import (  # 导入分布式通信函数
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_rank,  # 获取张量模型并行秩
    get_tensor_model_parallel_world_size,  # 获取张量模型并行世界大小
)
from sglang.srt.layers.activation import XIELU  # 导入xIELU激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS层归一化
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入逻辑处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入流水线并行工具
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    kv_cache_scales_loader,  # KV缓存缩放加载器
    maybe_remap_kv_scale_name,  # KV缩放名称重映射
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix, make_layers  # 导入工具函数

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class ApertusMLP(nn.Module):  # Apertus的MLP（多层感知机）模块
    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        bias: bool = False,  # 是否使用偏置
        prefix: str = "",  # 参数前缀
        reduce_results: bool = True,  # 是否归约结果
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.up_proj = ColumnParallelLinear(  # up投影的列并行线性层
            hidden_size,  # 输入大小
            intermediate_size,  # 输出大小
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # down投影的行并行线性层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
            reduce_results=reduce_results,  # 是否归约结果
        )
        if hidden_act != "xielu":  # 如果激活函数不是xIELU
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. "
                "Only xIELU is supported for now."  # 不支持的激活函数，仅支持xIELU
            )
        self.act_fn = XIELU()  # xIELU激活函数

    def forward(  # 前向传播
        self,
        x,  # 输入张量
        forward_batch=None,  # 前向批次信息
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ):
        # note: with xielu, there's no gate_proj
        # 注意：使用xIELU时，没有gate_proj
        x, _ = self.up_proj(x)  # up投影
        x = self.act_fn(x)  # 应用xIELU激活函数
        x, _ = self.down_proj(  # down投影
            x,
            skip_all_reduce=use_reduce_scatter,  # 是否跳过全归约（使用reduce-scatter替代）
        )
        return x  # 返回输出


class ApertusAttention(nn.Module):  # Apertus的注意力模块
    def __init__(  # 初始化方法
        self,
        config: ApertusConfig,  # Apertus配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID
        rope_theta: float = 10000,  # RoPE theta参数
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置
        rope_is_neox_style: bool = True,  # RoPE是否使用Neox风格
        max_position_embeddings: int = 8192,  # 最大位置嵌入数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        bias: bool = False,  # QKV投影是否使用偏置
        bias_o_proj: bool = False,  # 输出投影是否使用偏置
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层ID
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP世界大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个TP秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # KV头数大于TP大小，因此我们在多个TP GPU间划分
            # the KV heads across multiple tensor parallel GPUs.
            # KV头。
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate
            # KV头数小于TP大小，因此我们在多个TP GPU间复制
            # the KV heads across multiple tensor parallel GPUs.
            # KV头。
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个TP秩的KV头数
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        # MistralConfig有一个由Mistral-Nemo引入的可选head_dim
        self.head_dim = getattr(  # 获取每个头的维度
            config, "head_dim", self.hidden_size // self.total_num_heads  # 默认为隐藏大小除以总头数
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)  # 部分旋转因子
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)  # 旋转维度
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 保存RoPE theta
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置嵌入数

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            hidden_size,  # 输入大小
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=bias_o_proj,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.rotary_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基础频率
            rope_scaling=rope_scaling,  # 缩放配置
            is_neox_style=rope_is_neox_style,  # 是否使用Neox风格
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q归一化
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K归一化

    def forward(  # 前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分Q、K、V
        q = self.q_norm(q.contiguous().view(-1, self.head_dim)).view_as(q)  # 归一化Q
        k = self.k_norm(k.contiguous().view(-1, self.head_dim)).view_as(k)  # 归一化K
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力输出
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出


class ApertusDecoderLayer(nn.Module):  # Apertus解码器层
    def __init__(  # 初始化方法
        self,
        config: ApertusConfig,  # Apertus配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        rope_theta = config.rope_parameters["rope_theta"]  # 获取RoPE theta参数
        rope_scaling = config.rope_parameters  # 获取RoPE缩放配置
        if rope_scaling is not None and getattr(  # 如果有缩放配置且有original_max_position_embeddings
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 添加原始最大位置嵌入数
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)  # 获取RoPE风格
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 获取最大位置嵌入数
        # Support llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied with attention_bias
        # 支持llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied的attention_bias
        # Support internlm/internlm-7b with bias
        # 支持internlm/internlm-7b的bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(  # 获取注意力偏置配置
            config, "bias", False  # 兼容不同配置名
        )
        bias_o_proj = attention_bias  # 输出投影偏置与注意力偏置相同
        # support internlm/internlm3-8b with qkv_bias
        # 支持internlm/internlm3-8b的qkv_bias
        if hasattr(config, "qkv_bias"):  # 如果配置有qkv_bias
            attention_bias = config.qkv_bias  # 使用qkv_bias作为注意力偏置
        self.self_attn = ApertusAttention(  # 自注意力模块
            config=config,  # 配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放配置
            rope_is_neox_style=rope_is_neox_style,  # RoPE风格
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
            bias=attention_bias,  # 注意力偏置
            bias_o_proj=bias_o_proj,  # 输出投影偏置
        )
        self.mlp = ApertusMLP(  # MLP模块
            hidden_size=self.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            bias=getattr(config, "mlp_bias", False),  # MLP偏置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )
        self.attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力层归一化
        self.feedforward_layernorm = RMSNorm(  # 前馈层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # 自注意力
        if residual is None:  # 如果没有残差（第一层）
            residual = hidden_states  # 保存残差
            hidden_states = self.attention_layernorm(hidden_states)  # 注意力层归一化
        else:  # 有残差
            hidden_states, residual = self.attention_layernorm(hidden_states, residual)  # 注意力层归一化（含残差融合）
        hidden_states = self.self_attn(  # 自注意力
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次信息
        )

        # Fully Connected
        # 全连接
        hidden_states, residual = self.feedforward_layernorm(hidden_states, residual)  # 前馈层归一化（含残差融合）
        hidden_states = self.mlp(hidden_states)  # MLP
        return hidden_states, residual  # 返回隐藏状态和残差


class ApertusModel(nn.Module):  # Apertus模型主体
    def __init__(  # 初始化方法
        self,
        config: ApertusConfig,  # Apertus配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.quant_config = quant_config  # 保存量化配置
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.org_vocab_size = config.vocab_size  # 原始词表大小
        self.pp_group = get_pp_group()  # 获取流水线并行组
        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 嵌入维度
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("embed_tokens", prefix),  # 参数前缀
            )
        else:  # 非第一个秩
            self.embed_tokens = PPMissingLayer()  # 使用缺失层占位

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 隐藏层数量
            lambda idx, prefix: ApertusDecoderLayer(  # 创建解码器层的lambda函数
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix  # 配置、量化配置、层ID、前缀
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行世界大小
            prefix="model.layers",  # 参数前缀
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化
        else:  # 非最后一个秩
            self.norm = PPMissingLayer(return_tuple=True)  # 使用缺失层占位（返回元组）
        self.layers_to_capture = []  # 需要捕获中间输出的层列表

    def forward(  # 前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线并行代理张量
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            if input_embeds is None:  # 如果没有提供输入嵌入
                hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态
            else:  # 如果提供了输入嵌入
                hidden_states = input_embeds  # 直接使用输入嵌入
            residual = None  # 残差为None
        else:  # 非第一个秩
            assert pp_proxy_tensors is not None  # 断言代理张量不为None
            # FIXME(@ying): reduce the number of proxy tensors by not fusing layer norms
            # FIXME(@ying): 通过不融合层归一化来减少代理张量数量
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理张量获取残差
            deferred_norm = None  # 延迟归一化

        aux_hidden_states = []  # 辅助隐藏状态列表
        for i in range(self.start_layer, self.end_layer):  # 遍历负责的层
            if i in self.layers_to_capture:  # 如果该层需要捕获
                aux_hidden_states.append(hidden_states + residual)  # 捕获该层的隐藏状态
            layer = self.layers[i]  # 获取解码器层
            hidden_states, residual = layer(  # 通过解码器层
                positions,  # 位置编码
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次信息
                residual,  # 残差
            )

        if not self.pp_group.is_last_rank:  # 如果不是最后一个秩
            return PPProxyTensors(  # 返回代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:  # 最后一个秩
            hidden_states, _ = self.norm(hidden_states, residual)  # 最终层归一化

        if len(aux_hidden_states) == 0:  # 如果没有辅助隐藏状态
            return hidden_states  # 返回隐藏状态

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态

    # If this function is called, it should always initialize KV cache scale
    # 如果此函数被调用，应始终初始化KV缓存缩放因子
    # factors (or else raise an exception). Thus, handled exceptions should
    # （否则抛出异常）。因此，处理的异常应
    # make sure to leave KV cache scale factors in a known good (dummy) state
    # 确保将KV缓存缩放因子留在已知的良好（虚拟）状态
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:  # 加载KV缓存缩放因子
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP世界大小
        tp_rank = get_tensor_model_parallel_rank()  # 获取TP秩
        for layer_idx, scaling_factor in kv_cache_scales_loader(  # 遍历每层的缩放因子
            quantization_param_path,  # 量化参数路径
            tp_rank,  # TP秩
            tp_size,  # TP大小
            self.config.num_hidden_layers,  # 隐藏层数量
            self.config.__class__.model_type,  # 模型类型
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):  # 如果该层不是占位层
                layer_self_attn = self.layers[layer_idx].self_attn  # 获取自注意力模块

            if hasattr(layer_self_attn.attn, "k_scale"):  # 如果注意力模块有k_scale属性
                layer_self_attn.attn.k_scale = scaling_factor  # 设置K缩放因子
                layer_self_attn.attn.v_scale = scaling_factor  # 设置V缩放因子
            else:  # 没有缩放因子属性
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"  # 自注意力模块没有KV缓存缩放因子属性
                )


class ApertusForCausalLM(nn.Module):  # Apertus因果语言模型
    # LoRA specific attributes
    # LoRA特定属性
    embedding_modules = {  # 嵌入模块映射
        "embed_tokens": "input_embeddings",  # 输入嵌入
        "lm_head": "output_embeddings",  # 输出嵌入
    }
    embedding_padding_modules = ["lm_head"]  # 嵌入填充模块
    # BitandBytes specific attributes
    # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # bitsandbytes默认目标模块
        ".down_proj.",  # down投影
        ".up_proj.",  # up投影
        ".q_proj.",  # Q投影
        ".k_proj.",  # K投影
        ".v_proj.",  # V投影
        ".o_proj.",  # 输出投影
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    # 在TP中，这些权重沿列维度（dim=-1）分区
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]  # 列并行权重模块
    bitsandbytes_stacked_params_mapping = {  # bitsandbytes堆叠参数映射
        # shard_name, weight_name, index  # 分片名，权重名，索引
        ".q_proj": (".qkv_proj", 0),  # Q投影映射
        ".k_proj": (".qkv_proj", 1),  # K投影映射
        ".v_proj": (".qkv_proj", 2),  # V投影映射
    }

    def __init__(  # 初始化方法
        self,
        config: ApertusConfig,  # Apertus配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))  # 初始化模型主体
        if self.config.tie_word_embeddings:  # 如果共享词嵌入和LM头权重
            self.lm_head = self.model.embed_tokens  # LM头共享词嵌入
        else:  # 不共享
            self.lm_head = ParallelLMHead(  # 独立的LM头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力TP组
            )
        self.logits_processor = LogitsProcessor(config)  # 逻辑处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 池化层（取最后一个token，归一化）
        self.stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # （参数名，分片名，分片ID）
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
        ]

        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    def _init_model(  # 初始化模型主体
        self,
        config: ApertusConfig,  # Apertus配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        return ApertusModel(config, quant_config=quant_config, prefix=prefix)  # 返回ApertusModel实例

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        get_embedding: bool = False,  # 是否获取嵌入（用于池化模型）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线并行代理张量
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(  # 获取模型主体输出
            input_ids,  # 输入ID
            positions,  # 位置编码
            forward_batch,  # 前向批次信息
            input_embeds,  # 输入嵌入
            pp_proxy_tensors=pp_proxy_tensors,  # 代理张量
        )

        aux_hidden_states = None  # 辅助隐藏状态
        if self.capture_aux_hidden_states:  # 如果需要捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包隐藏状态和辅助隐藏状态

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            if not get_embedding:  # 如果不获取嵌入
                return self.logits_processor(  # 通过逻辑处理器计算输出
                    input_ids,  # 输入ID
                    hidden_states,  # 隐藏状态
                    self.lm_head,  # LM头
                    forward_batch,  # 前向批次信息
                    aux_hidden_states,  # 辅助隐藏状态
                )
            else:  # 获取嵌入
                return self.pooler(hidden_states, forward_batch)  # 通过池化层获取嵌入
        else:  # 非最后一个秩
            return hidden_states  # 返回隐藏状态

    @torch.no_grad()  # 禁用梯度计算
    def forward_split_prefill(  # 分割预填充前向传播（用于分段处理）
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        split_interval: Tuple[int, int],  # [start, end) 0-based  # [起始, 结束) 0基索引
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval  # 解包分割区间
        # embed
        # 嵌入
        if start == 0:  # 如果从第0层开始
            if input_embeds is None:  # 如果没有提供输入嵌入
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态
            else:  # 如果提供了输入嵌入
                forward_batch.hidden_states = input_embeds  # 直接使用输入嵌入
        # decoder layer
        # 解码器层
        for i in range(start, end):  # 遍历指定范围的层
            layer = self.model.layers[i]  # 获取解码器层
            forward_batch.hidden_states, forward_batch.residual = layer(  # 通过解码器层
                positions,  # 位置编码
                forward_batch.hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次信息
                forward_batch.residual,  # 残差
            )

        if end == self.model.config.num_hidden_layers:  # 如果处理到最后一层
            # norm
            # 归一化
            hidden_states, _ = self.model.norm(  # 最终层归一化
                forward_batch.hidden_states, forward_batch.residual  # 隐藏状态和残差
            )
            forward_batch.hidden_states = hidden_states  # 更新隐藏状态
            # logits process
            # 逻辑处理
            result = self.logits_processor(  # 通过逻辑处理器计算输出
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch  # 输入ID、隐藏状态、LM头、批次信息
            )
        else:  # 未处理到最后一层
            result = None  # 结果为None

        return result  # 返回结果

    @property
    def start_layer(self):  # 获取起始层
        return self.model.start_layer  # 返回模型主体的起始层

    @property
    def end_layer(self):  # 获取结束层
        return self.model.end_layer  # 返回模型主体的结束层

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.model.embed_tokens  # 返回模型主体中的词嵌入层

    def get_module_name_from_weight_name(self, name):  # 从权重名获取模块名
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:  # 遍历堆叠参数映射
            if weight_name in name:  # 如果权重名在名称中
                return (  # 返回模块名和分片数
                    name.replace(weight_name, param_name)[: -len(".weight")],  # 替换权重名为参数名并移除.weight后缀
                    num_shard,  # 分片数
                )
        return name[: -len(".weight")], 1  # 返回模块名和分片数1

    def get_num_params(self):  # 获取参数数量
        params_dict = dict(self.named_parameters())  # 获取参数字典
        return len(params_dict)  # 返回参数数量

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # （参数名，分片名，分片ID）
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
        ]

        params_dict = dict(self.named_parameters())  # 获取模型参数字典

        for name, buffer in self.named_buffers():  # 遍历命名缓冲区
            if name.endswith(".beta") or name.endswith(".eps"):  # 如果是beta或eps缓冲区
                params_dict[name] = buffer  # 添加到参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            layer_id = get_layer_id(name)  # 获取层ID
            if (  # 如果层ID不在当前进程负责的范围内
                layer_id is not None  # 层ID不为None
                and hasattr(self.model, "start_layer")  # 模型有start_layer属性
                and (
                    layer_id < self.model.start_layer  # 层ID小于起始层
                    or layer_id >= self.model.end_layer  # 层ID大于等于结束层
                )
            ):
                continue  # 跳过此权重
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转位置编码和投影器
                continue  # 跳过
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的位置编码
                # Models trained using ColossalAI may include these tensors in
                # 使用ColossalAI训练的模型可能在检查点中包含这些张量。
                # the checkpoint. Skip them.
                # 跳过它们。
                continue  # 跳过
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过不在参数字典中的视觉塔权重
                continue  # 跳过
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 如果共享词嵌入且是LM头权重
                continue  # 跳过（已通过词嵌入共享）
            # Handle FP8 kv-scale remapping
            # 处理FP8 KV缩放重映射
            if "scale" in name:  # 如果名称中包含scale
                name = maybe_remap_kv_scale_name(name, params_dict)  # 可能重映射KV缩放名称
                if name is None:  # 如果重映射后为None
                    continue  # 跳过

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在当前名称中
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.
                # 跳过加载GPTQ模型的额外偏置。
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                if name not in params_dict:  # 如果参数不存在
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出堆叠参数循环
            else:  # 没有匹配堆叠参数映射
                # Skip loading extra bias for GPTQ models.
                # 跳过加载GPTQ模型的额外偏置。
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                # Skip loading kv_scale from ckpts towards new design.
                # 跳过从检查点加载kv_scale（新设计不再需要）。
                if name.endswith(".kv_scale") and name not in params_dict:  # 如果是kv_scale且不在参数字典中
                    continue  # 跳过
                if name in params_dict.keys():  # 如果权重在参数字典中
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认使用默认加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                else:  # 权重不在参数字典中
                    logger.warning(f"Parameter {name} not found in params_dict")  # 记录警告日志

    def get_embed_and_head(self):  # 获取嵌入层和LM头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回词嵌入权重和LM头权重

    def set_embed_and_head(self, embed, head):  # 设置嵌入层和LM头权重
        del self.model.embed_tokens.weight  # 删除旧的词嵌入权重
        del self.lm_head.weight  # 删除旧的LM头权重
        self.model.embed_tokens.weight = embed  # 设置新的词嵌入权重
        self.lm_head.weight = head  # 设置新的LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA

    def get_embed(self):  # 获取嵌入层权重
        return self.model.embed_tokens.weight  # 返回词嵌入权重

    def set_embed(self, embed):  # 设置嵌入层权重
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        # 注意：如果draft隐藏大小 != 目标隐藏大小，则嵌入权重不能为EAGLE3共享
        if (
            hasattr(self.config, "target_hidden_size")  # 如果配置有target_hidden_size
            and self.config.target_hidden_size != self.config.hidden_size  # 且目标隐藏大小与当前不同
        ):
            return  # 不共享，直接返回
        del self.model.embed_tokens.weight  # 删除旧的词嵌入权重
        self.model.embed_tokens.weight = embed  # 设置新的词嵌入权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:  # 加载KV缓存缩放因子
        self.model.load_kv_cache_scales(quantization_param_path)  # 委托给模型主体

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):  # 设置EAGLE3需要捕获的层
        if not self.pp_group.is_last_rank:  # 如果不是最后一个秩
            return  # 直接返回

        if layer_ids is None:  # 如果没有指定层ID
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            num_layers = self.config.num_hidden_layers  # 获取隐藏层数量
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]  # 默认捕获第2、中间、倒数第3层
        else:  # 指定了层ID
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # 这里加1是因为在sglang中，第i层将
            # of the (i-1)th layer as aux hidden state
            # 第(i-1)层的输出作为辅助隐藏状态
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 每个层ID加1


EntryClass = [ApertusForCausalLM]  # 入口类为ApertusForCausalLM列表
