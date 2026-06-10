# Gemma3因果语言模型：实现Gemma3文本模型的因果语言建模，包含MLP、注意力、解码层、旋转嵌入和完整语言模型
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
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
import copy  # 导入深拷贝模块 # import deep copy module
from typing import Iterable, List, Optional, Set, Tuple  # 导入类型提示工具 # import type hints

import einops  # 导入einops张量操作库 # import einops tensor ops library
import torch  # 导入PyTorch库 # import PyTorch
from torch import nn  # 导入神经网络模块 # import neural network module
from transformers import (  # 从transformers导入相关类 # import related classes from transformers
    ROPE_INIT_FUNCTIONS,
    Gemma3TextConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from sglang.srt.distributed import (  # 导入分布式并行相关函数 # import distributed parallel functions
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import GeluAndMul  # 导入GELU激活函数与乘法组合层 # import GELU activation and multiply layer
from sglang.srt.layers.layernorm import Gemma3RMSNorm  # 导入Gemma3的RMS归一化层 # import Gemma3 RMS norm layer
from sglang.srt.layers.linear import (  # 导入并行线性层 # import parallel linear layers
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器 # import logits processor
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # import quantization config
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention  # 导入注意力类型和Radix注意力层 # import attention type and Radix attention
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb, get_rope  # 导入旋转位置编码相关函数 # import rotary position embedding functions
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入词表并行嵌入层 # import vocab parallel embedding layer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # import forward batch info
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具 # import weight loading utilities
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu, make_layers  # 导入工具函数 # import utility functions

_is_cpu = is_cpu()  # 是否为CPU环境 # whether running on CPU
_is_cpu_amx_available = cpu_has_amx_support()  # CPU是否支持AMX指令集 # whether CPU supports AMX instructions


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
# 与HuggingFace实现对齐，滑动窗口包含最后一个token；SGLang假设不包含（排他）
def get_attention_sliding_window_size(config):  # 获取注意力滑动窗口大小 # get attention sliding window size
    return config.sliding_window - 1  # 返回滑动窗口大小减1，因为SGLang使用排他窗口 # return sliding window minus 1, as SGLang uses exclusive window


# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma3.py
# 适配自vLLM项目的Gemma3实现
def extract_layer_index(prefix: str) -> int:  # 从前缀字符串中提取层索引 # extract layer index from prefix string
    """Extract the layer index from a prefix string."""  # 从前缀字符串中提取层索引 # Extract the layer index from a prefix string
    parts = prefix.split(".")  # 按点号分割前缀 # split prefix by dot
    for part in parts:  # 遍历各部分 # iterate over parts
        if part.startswith("layers."):  # 如果部分以"layers."开头 # if part starts with "layers."
            layer_str = part.split(".")[-1]  # 获取最后的数字部分 # get the last numeric part
            try:
                return int(layer_str)  # 转换为整数返回 # convert to integer and return
            except ValueError:
                continue  # 转换失败则继续 # continue if conversion fails
    return -1  # 未找到则返回-1 # return -1 if not found


class Gemma3MLP(nn.Module):  # Gemma3多层感知机模块 # Gemma3 MLP module
    def __init__(  # 初始化方法 # initialization method
        self,
        hidden_size: int,  # 隐藏层大小 # hidden layer size
        intermediate_size: int,  # 中间层大小 # intermediate layer size
        hidden_activation: str,  # 隐藏层激活函数名 # hidden activation function name
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__()  # 调用父类初始化 # call parent class init
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并的并行线性层 # merged gate and up projection parallel linear layer
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(  # 下投影行并行线性层 # down projection row parallel linear layer
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_activation != "gelu_pytorch_tanh":  # 检查激活函数是否正确 # check if activation function is correct
            raise ValueError(  # 抛出值错误 # raise value error
                f"{self.__class__.__name__} uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul()  # GELU激活与乘法组合函数 # GELU activation and multiply function
        self.prefix = prefix  # 保存前缀 # save prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法 # forward pass method
        gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影层 # through gate up projection layer
        x = self.act_fn(gate_up)  # 应用GELU激活和门控乘法 # apply GELU activation and gate multiply
        x, _ = self.down_proj(x)  # 通过下投影层 # through down projection layer
        return x  # 返回输出 # return output


class Gemma3Attention(nn.Module):  # Gemma3注意力模块 # Gemma3 attention module
    def __init__(  # 初始化方法 # initialization method
        self,
        layer_id: int,  # 层ID # layer ID
        config: Gemma3TextConfig,  # Gemma3文本配置 # Gemma3 text config
        max_position_embeddings: int,  # 最大位置编码数 # max position embeddings
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__()  # 调用父类初始化 # call parent class init
        self.layer_id = layer_id  # 保存层ID # save layer ID
        self.config = config  # 保存配置 # save config
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小 # get tensor parallel world size

        self.total_num_heads = config.num_attention_heads  # 总注意力头数 # total number of attention heads
        assert self.total_num_heads % tp_size == 0  # 断言头数能被并行度整除 # assert heads divisible by parallelism
        self.num_heads = self.total_num_heads // tp_size  # 每个并行单元的头数 # heads per parallel unit
        self.total_num_kv_heads = config.num_key_value_heads  # 总KV头数 # total KV heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个并行单元的KV头数 # KV heads per parallel unit

        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于并行度 # KV heads >= parallelism
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            # KV头数大于TP大小，因此在多个张量并行GPU之间分配KV头
            assert self.total_num_kv_heads % tp_size == 0
        else:  # KV头数小于并行度 # KV heads < parallelism
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            # KV头数小于TP大小，因此在多个张量并行GPU之间复制KV头
            assert tp_size % self.total_num_kv_heads == 0

        hidden_size = config.hidden_size  # 隐藏层大小 # hidden size

        head_dim = getattr(  # 获取头维度 # get head dimension
            config, "head_dim", hidden_size // config.num_attention_heads
        )
        self.head_dim = head_dim  # 保存头维度 # save head dimension
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)  # 部分旋转因子 # partial rotary factor
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)  # 旋转维度 # rotary dimension
        self.q_size = self.num_heads * self.head_dim  # 查询向量大小 # query vector size

        self.kv_size = self.num_kv_heads * self.head_dim  # KV向量大小 # KV vector size
        self.scaling = config.query_pre_attn_scalar**-0.5  # 注意力缩放因子 # attention scaling factor

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影层 # QKV parallel projection layer
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 输出投影层 # output projection layer
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"  # 是否为滑动注意力层 # whether sliding attention layer

        # In transformers v5, rope_parameters is nested per layer type:
        #   {"sliding_attention": {"rope_theta": 10000}, "full_attention": {"rope_theta": 1000000}}
        # In v4 it was flat: {"rope_type": "default", "rope_theta": ...}
        # 在transformers v5中，rope_parameters按层类型嵌套；在v4中是扁平格式
        rope_params = config.rope_parameters  # 获取RoPE参数 # get RoPE parameters
        is_nested = isinstance(rope_params, dict) and "full_attention" in rope_params  # 是否为嵌套格式 # whether nested format

        # Initialize the rotary embedding.
        # 初始化旋转位置编码
        if self.is_sliding:  # 滑动注意力（局部注意力） # sliding attention (local attention)
            # Local attention. Override the values in config.json.
            # 局部注意力，覆盖config.json中的值
            if is_nested:  # 嵌套格式 # nested format
                self.rope_theta = rope_params["sliding_attention"].get(
                    "rope_theta", 10000.0
                )
            else:  # 扁平格式 # flat format
                self.rope_theta = getattr(config, "rope_local_base_freq", 10000.0)
            self.rope_scaling = {"rope_type": "default"}  # RoPE缩放类型为默认 # RoPE scaling type is default
            # FIXME(mick): idk why vllm does this
            # 待修复(mick)：不确定vllm为何这样做
            # self.sliding_window = config.interleaved_sliding_window
            self.sliding_window = get_attention_sliding_window_size(config)  # 获取滑动窗口大小 # get sliding window size
        else:  # 全局注意力 # global attention
            # Global attention. Use the values in config.json.
            # 全局注意力，使用config.json中的值
            if is_nested:  # 嵌套格式 # nested format
                self.rope_theta = rope_params["full_attention"].get(
                    "rope_theta", 1000000.0
                )
            else:  # 扁平格式 # flat format
                self.rope_theta = (
                    rope_params.get("rope_theta", 10000.0) if rope_params else 10000.0
                )
            self.rope_scaling = {"rope_type": "default"}  # RoPE缩放类型为默认 # RoPE scaling type is default
            self.sliding_window = None  # 全局注意力无滑动窗口 # no sliding window for global attention
        self.rotary_emb = get_rope(  # 创建旋转位置编码 # create rotary position embedding
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=getattr(config, "rope_is_neox_style", True),
        )
        self.attn = RadixAttention(  # 创建Radix注意力层 # create Radix attention layer
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            # Module must also define `get_attention_sliding_window_size` to correctly initialize
            # attention backend in `ForwardBatch`.
            # 模块必须定义`get_attention_sliding_window_size`以正确初始化ForwardBatch中的注意力后端
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            attn_type=AttentionType.DECODER_BIDIRECTIONAL,  # 解码器双向注意力类型 # decoder bidirectional attention type
        )

        # Gemma3 adds normalization for q and k
        # Gemma3对q和k添加归一化
        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)  # 查询归一化层 # query normalization layer
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)  # 键归一化层 # key normalization layer

    def forward_cpu(  # CPU前向传播方法 # CPU forward pass method
        self,
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        hidden_states: torch.Tensor,  # 隐藏状态张量 # hidden states tensor
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 位置嵌入元组(cos, sin) # position embeddings tuple (cos, sin)
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影 # compute QKV projection
        # [s, h * head_dim]
        # [序列长度, 头数 * 头维度]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割QKV # split QKV

        # [s, h, head_dim]
        # [序列长度, 头数, 头维度]
        q = q.unflatten(-1, (self.num_heads, self.head_dim)).unsqueeze(0)  # 重塑Q的形状 # reshape Q
        q = self.q_norm(q)  # 对Q进行归一化 # normalize Q
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim)).unsqueeze(0)  # 重塑K的形状 # reshape K
        k = self.k_norm(k)  # 对K进行归一化 # normalize K
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码 # apply rotary position embedding

        attn_output = self.attn(q, k, v, forward_batch=forward_batch)  # 计算注意力输出 # compute attention output

        # Compatible with triton backend which returns [1, s, h, head_dim]
        # 兼容triton后端返回的[1, s, h, head_dim]形状
        if attn_output.dim() == 4 and attn_output.shape[0] == 1:
            attn_output = attn_output.squeeze(0)  # 去除第一维 # remove first dimension
            attn_output = attn_output.flatten(-2, -1)  # 展平最后两维 # flatten last two dimensions
        # [s, h * head_dim]
        # [序列长度, 头数 * 头维度]

        output, _ = self.o_proj(attn_output)  # 通过输出投影层 # through output projection layer
        return output  # 返回输出 # return output

    def forward_native(  # 原生前向传播方法 # native forward pass method
        self,
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        hidden_states: torch.Tensor,  # 隐藏状态张量 # hidden states tensor
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 位置嵌入元组(cos, sin) # position embeddings tuple (cos, sin)
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影 # compute QKV projection
        # [s, h * head_dim]
        # [序列长度, 头数 * 头维度]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割QKV # split QKV

        # [s, h, head_dim]
        # [序列长度, 头数, 头维度]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # 重塑Q的形状 # reshape Q
        # -> [h, s, head_dim]
        # -> [头数, 序列长度, 头维度]
        q = q.transpose(0, 1).unsqueeze(0)  # 转置并增加批次维 # transpose and add batch dim
        q = self.q_norm(q)  # 对Q进行归一化 # normalize Q
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))  # 重塑K的形状 # reshape K
        # -> [h, s, head_dim]
        # -> [头数, 序列长度, 头维度]
        k = k.transpose(0, 1).unsqueeze(0)  # 转置并增加批次维 # transpose and add batch dim
        k = self.k_norm(k)  # 对K进行归一化 # normalize K

        # q, k = self.rotary_emb(positions, q, k)
        # 注释掉的旋转编码方式 # commented out rotary embedding approach
        cos, sin = position_embeddings  # 获取余弦和正弦位置嵌入 # get cos and sin position embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # 应用旋转位置编码 # apply rotary position embedding

        # [b, h, s, head_dim] ->  [b, s, h, head_dim]
        # [批次, 头数, 序列长度, 头维度] -> [批次, 序列长度, 头数, 头维度]
        q = q.permute(0, 2, 1, 3)  # 重排Q的维度 # rearrange Q dimensions
        k = k.permute(0, 2, 1, 3)  # 重排K的维度 # rearrange K dimensions

        attn_output = self.attn(q, k, v, forward_batch=forward_batch)  # 计算注意力输出 # compute attention output

        # Compatible with triton backend which returns [1, s, h, head_dim]
        # 兼容triton后端返回的[1, s, h, head_dim]形状
        if attn_output.dim() == 4 and attn_output.shape[0] == 1:
            attn_output = attn_output.squeeze(0)  # 去除第一维 # remove first dimension
            attn_output = attn_output.flatten(-2, -1)  # 展平最后两维 # flatten last two dimensions
        # [s, h * head_dim]
        # [序列长度, 头数 * 头维度]

        output, _ = self.o_proj(attn_output)  # 通过输出投影层 # through output projection layer
        return output  # 返回输出 # return output

    def forward(  # 前向传播入口方法 # forward pass entry method
        self,
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        hidden_states: torch.Tensor,  # 隐藏状态张量 # hidden states tensor
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 位置嵌入元组 # position embeddings tuple
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        **kwargs,
    ) -> torch.Tensor:
        if _is_cpu and _is_cpu_amx_available:  # 如果CPU支持AMX # if CPU supports AMX
            return self.forward_cpu(  # 使用CPU前向传播 # use CPU forward pass
                positions, hidden_states, position_embeddings, forward_batch, **kwargs
            )
        return self.forward_native(  # 使用原生前向传播 # use native forward pass
            positions, hidden_states, position_embeddings, forward_batch, **kwargs
        )


class Gemma3DecoderLayer(nn.Module):  # Gemma3解码器层 # Gemma3 decoder layer
    def __init__(  # 初始化方法 # initialization method
        self,
        layer_id: int,  # 层ID # layer ID
        config: PretrainedConfig,  # 预训练配置 # pretrained config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__()  # 调用父类初始化 # call parent class init
        self.hidden_size = config.hidden_size  # 隐藏层大小 # hidden size
        self.self_attn = Gemma3Attention(  # 自注意力层 # self attention layer
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.hidden_size = config.hidden_size  # 隐藏层大小 # hidden size
        self.mlp = Gemma3MLP(  # MLP模块 # MLP module
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = Gemma3RMSNorm(  # 输入层归一化 # input layer norm
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma3RMSNorm(  # 注意力后归一化 # post attention norm
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(  # 前馈前归一化 # pre feedforward norm
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(  # 前馈后归一化 # post feedforward norm
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding  # 是否为滑动注意力 # whether sliding attention
        self.layer_id = layer_id  # 保存层ID # save layer ID

    def forward(  # 前向传播方法 # forward pass method
        self,
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        hidden_states: torch.Tensor,  # 隐藏状态张量 # hidden states tensor
        position_embeddings_global: torch.Tensor,  # 全局位置嵌入 # global position embeddings
        position_embeddings_local: torch.Tensor,  # 局部位置嵌入 # local position embeddings
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        **kwargs,
    ) -> tuple[  # 返回元组 # return tuple
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states  # 保存残差连接 # save residual connection
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化 # input layer norm

        # apply global RoPE to non-sliding layer only
        # 仅对非滑动层应用全局RoPE
        if self.self_attn.is_sliding:  # 滑动注意力层 # sliding attention layer
            position_embeddings = position_embeddings_local  # 使用局部位置嵌入 # use local position embeddings
        else:  # 全局注意力层 # global attention layer
            position_embeddings = position_embeddings_global  # 使用全局位置嵌入 # use global position embeddings

        hidden_states = self.self_attn(  # 通过自注意力层 # through self attention layer
            positions=positions,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            forward_batch=forward_batch,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化 # post attention norm
        hidden_states = residual + hidden_states  # 残差连接 # residual connection

        residual = hidden_states  # 更新残差 # update residual
        hidden_states = self.pre_feedforward_layernorm(hidden_states)  # 前馈前归一化 # pre feedforward norm
        hidden_states = self.mlp(hidden_states)  # 通过MLP层 # through MLP layer
        hidden_states = self.post_feedforward_layernorm(hidden_states)  # 前馈后归一化 # post feedforward norm
        hidden_states = residual + hidden_states  # 残差连接 # residual connection

        outputs = (hidden_states,)  # 输出元组 # output tuple

        return outputs  # 返回输出 # return outputs


class Gemma3RotaryEmbedding(nn.Module):  # Gemma3旋转位置编码模块 # Gemma3 rotary position embedding module
    def __init__(self, config: Gemma3TextConfig, device=None):  # 初始化方法 # initialization method
        super().__init__()  # 调用父类初始化 # call parent class init
        # BC: "rope_type" was originally "type"
        # 向后兼容："rope_type"最初为"type"
        rope_scaling = config.rope_parameters  # 获取RoPE缩放参数 # get RoPE scaling parameters
        if rope_scaling is not None:  # 如果缩放参数不为空 # if scaling params not None
            self.rope_type = rope_scaling.get(  # 获取RoPE类型 # get RoPE type
                "rope_type", rope_scaling.get("type", "default")
            )

        else:  # 缩放参数为空 # scaling params is None
            self.rope_type = "default"  # 默认类型 # default type

        if self.rope_type is None:  # 类型为空 # type is None
            self.rope_type = "default"  # 设为默认 # set to default

        self.max_seq_len_cached = config.max_position_embeddings  # 缓存的最大序列长度 # cached max sequence length
        self.original_max_seq_len = config.max_position_embeddings  # 原始最大序列长度 # original max sequence length

        self.config = config  # 保存配置 # save config

        if self.rope_type == "default":  # 默认RoPE类型 # default RoPE type
            self.rope_init_fn = self.compute_default_rope_parameters  # 使用默认参数计算函数 # use default param compute function
        else:  # 其他RoPE类型 # other RoPE type
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]  # 使用注册的初始化函数 # use registered init function

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)  # 计算逆频率和注意力缩放 # compute inverse frequency and attention scaling
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册逆频率缓冲区 # register inverse frequency buffer
        self.original_inv_freq = self.inv_freq  # 保存原始逆频率 # save original inverse frequency

    def _dynamic_frequency_update(self, position_ids, device):  # 动态频率更新方法 # dynamic frequency update method
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        # 动态RoPE层应在以下情况重新计算`inv_freq`：
        # 1 - 超出缓存的序列长度（允许缩放）
        # 2 - 当前序列长度在原始范围内（避免小序列精度损失）
        seq_len = torch.max(position_ids) + 1  # 当前序列长度 # current sequence length
        if seq_len > self.max_seq_len_cached:  # growth # 增长 # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(  # 重新计算逆频率 # recompute inverse frequency
                self.config, device, seq_len=seq_len
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            # 注册新的逆频率缓冲区 # TODO joao: 可能在编译时出错
            self.max_seq_len_cached = seq_len  # 更新缓存长度 # update cached length

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset # 重置 # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            # 如果模型在初始化后被移动到设备上，需要此.to()操作（因为缓冲区会自动移动，但原始副本不会）
            self.original_inv_freq = self.original_inv_freq.to(device)  # 移动到设备 # move to device
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)  # 恢复原始逆频率 # restore original inverse frequency
            self.max_seq_len_cached = self.original_max_seq_len  # 恢复缓存长度 # restore cached length

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):  # 计算默认RoPE参数 # compute default RoPE parameters
        """Standard RoPE: no scaling, just base frequency."""  # 标准RoPE：无缩放，仅使用基础频率 # Standard RoPE: no scaling, just base frequency
        rope_params = config.rope_parameters  # 获取RoPE参数 # get RoPE parameters
        if isinstance(rope_params, dict) and "rope_theta" not in rope_params:  # 嵌套格式且无直接的rope_theta # nested format without direct rope_theta
            # Nested per-layer-type format; pick the first available theta
            # 按层类型嵌套格式；选择第一个可用的theta
            for v in rope_params.values():
                if isinstance(v, dict) and "rope_theta" in v:
                    base = v["rope_theta"]  # 获取基础频率 # get base frequency
                    break
            else:
                base = 10000.0  # 默认基础频率 # default base frequency
        else:  # 扁平格式 # flat format
            base = rope_params.get("rope_theta", 10000.0) if rope_params else 10000.0  # 获取基础频率 # get base frequency
        dim = (  # 计算维度 # compute dimension
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (  # 计算逆频率 # compute inverse frequency
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0  # 返回逆频率和缩放因子1.0 # return inverse frequency and scaling factor 1.0

    @torch.no_grad()  # 禁用梯度计算 # disable gradient computation
    def forward(self, x, position_ids):  # 前向传播方法 # forward pass method
        if "dynamic" in self.rope_type:  # 动态RoPE类型 # dynamic RoPE type
            self._dynamic_frequency_update(position_ids, device=x.device)  # 动态更新频率 # dynamically update frequency

        # Core RoPE block
        # 核心RoPE计算块
        inv_freq_expanded = (  # 扩展逆频率维度 # expand inverse frequency dimensions
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()  # 扩展位置ID维度 # expand position IDs dimensions
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        # 强制使用float32（参见相关PR）
        device_type = x.device.type  # 获取设备类型 # get device type
        device_type = (  # 处理设备类型 # process device type
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # 禁用自动混合精度 # disable automatic mixed precision
            freqs = (  # 计算频率 # compute frequencies
                inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)  # 拼接频率 # concatenate frequencies
            cos = emb.cos()  # 计算余弦 # compute cosine
            sin = emb.sin()  # 计算正弦 # compute sine

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        # 高级RoPE类型（如yarn）应用后处理缩放因子，等价于缩放注意力
        cos = cos * self.attention_scaling  # 缩放余弦 # scale cosine
        sin = sin * self.attention_scaling  # 缩放正弦 # scale sine

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # 返回与输入同类型的余弦和正弦 # return cos and sin in input dtype


class Gemma3TextScaledWordEmbedding(nn.Embedding):  # Gemma3文本缩放词嵌入层 # Gemma3 text scaled word embedding layer
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """
    # 此模块重写nn.Embeddings的forward方法，通过乘以嵌入缩放因子
    def __init__(  # 初始化方法 # initialization method
        self,
        num_embeddings: int,  # 嵌入数量（词表大小） # number of embeddings (vocab size)
        embedding_dim: int,  # 嵌入维度 # embedding dimension
        padding_idx: int,  # 填充索引 # padding index
        embed_scale: Optional[float] = 1.0,  # 嵌入缩放因子 # embedding scale factor
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)  # 调用父类初始化 # call parent class init
        self.embed_scale = embed_scale  # 保存缩放因子 # save scale factor

    def forward(self, input_ids: torch.Tensor):  # 前向传播方法 # forward pass method
        return super().forward(input_ids) * self.embed_scale  # 嵌入输出乘以缩放因子 # multiply embedding output by scale factor


class Gemma3TextModel(PreTrainedModel):  # Gemma3文本模型类 # Gemma3 text model class
    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3TextConfig,  # Gemma3文本配置 # Gemma3 text config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config
        self.quant_config = quant_config  # 保存量化配置 # save quantization config

        self.padding_idx = config.pad_token_id  # 填充token ID # padding token ID
        self.vocab_size = config.vocab_size  # 词表大小 # vocab size

        # Gemma3 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        # Gemma3将以下值向下转换为float16，导致sqrt(3072)=55.4256变为55.5
        self.embed_tokens = Gemma3TextScaledWordEmbedding(  # 缩放词嵌入层 # scaled word embedding layer
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # RMS归一化层 # RMS norm layer

        # In transformers v5, rope_parameters is nested per layer type:
        #   {"sliding_attention": {"rope_type": ..., "rope_theta": 10000},
        #    "full_attention":    {"rope_type": ..., "rope_theta": 1000000}}
        # Flatten into the format Gemma3RotaryEmbedding expects.
        # 在transformers v5中，rope_parameters按层类型嵌套；展平为Gemma3RotaryEmbedding期望的格式
        rope_params = config.rope_parameters  # 获取RoPE参数 # get RoPE parameters
        if isinstance(rope_params, dict) and "full_attention" in rope_params:  # 嵌套格式 # nested format
            global_theta = rope_params["full_attention"].get("rope_theta", 1000000.0)  # 全局theta # global theta
            local_theta = rope_params["sliding_attention"].get("rope_theta", 10000.0)  # 局部theta # local theta
        else:  # 扁平格式 # flat format
            # v4 flat format fallback
            # v4扁平格式回退
            global_theta = (
                rope_params.get("rope_theta", 10000.0) if rope_params else 10000.0
            )
            local_theta = getattr(config, "rope_local_base_freq", 10000.0)

        global_config = copy.deepcopy(config)  # 深拷贝配置用于全局 # deep copy config for global
        global_config.rope_parameters = {  # 设置全局RoPE参数 # set global RoPE parameters
            "rope_theta": global_theta,
            "factor": config.rope_parameters["full_attention"]["factor"],
            "rope_type": "linear",
        }
        self.rotary_emb = Gemma3RotaryEmbedding(config=global_config)  # 全局旋转嵌入 # global rotary embedding
        self.gradient_checkpointing = False  # 梯度检查点标志 # gradient checkpointing flag

        local_config = copy.deepcopy(config)  # 深拷贝配置用于局部 # deep copy config for local
        local_config.rope_parameters = {  # 设置局部RoPE参数 # set local RoPE parameters
            "rope_type": "default",
            "rope_theta": local_theta,
        }
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_config)  # 局部旋转嵌入 # local rotary embedding

        self.layers = make_layers(  # 创建解码器层列表 # create decoder layer list
            config.num_hidden_layers,
            lambda idx, prefix: Gemma3DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层 # final norm layer
        self.layers_to_capture = []  # 需要捕获隐藏状态的层 # layers to capture hidden states
        self.post_init()  # 调用后初始化 # call post init

    def forward(  # 前向传播方法 # forward pass method
        self,
        input_ids: torch.Tensor,  # 输入token ID张量 # input token ID tensor
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选 # input embeddings, optional
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is None:  # 无输入嵌入 # no input embeddings
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态 # get hidden states through embedding layer
        else:  # 有输入嵌入 # has input embeddings
            hidden_states = input_embeds  # 直接使用输入嵌入 # use input embeddings directly

        aux_hidden_states = []  # 辅助隐藏状态列表 # auxiliary hidden states list

        num_layers = len(self.layers)  # 层数 # number of layers
        if _is_cpu and _is_cpu_amx_available:  # CPU且支持AMX # CPU with AMX support
            for i, layer in enumerate(self.layers):  # 遍历每层 # iterate each layer
                if i in self.layers_to_capture:  # 需要捕获隐藏状态 # need to capture hidden states
                    aux_hidden_states.append(hidden_states)  # 添加到辅助列表 # append to aux list
                layer_outputs = layer(  # 获取层输出 # get layer output
                    positions=positions,
                    position_embeddings_global=None,
                    position_embeddings_local=None,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]  # 更新隐藏状态 # update hidden states
        else:  # GPU或CPU不支持AMX # GPU or CPU without AMX
            if positions.dim() == 1:  # 一维位置 # 1D positions
                positions = einops.rearrange(positions, "s -> 1 s")  # 重排为二维 # rearrange to 2D

            position_embeddings_global = self.rotary_emb(hidden_states, positions)  # 全局位置嵌入 # global position embeddings
            position_embeddings_local = self.rotary_emb_local(hidden_states, positions)  # 局部位置嵌入 # local position embeddings
            for i, layer in enumerate(self.layers):  # 遍历每层 # iterate each layer
                if i in self.layers_to_capture:  # 需要捕获隐藏状态 # need to capture hidden states
                    aux_hidden_states.append(hidden_states)  # 添加到辅助列表 # append to aux list
                layer_outputs = layer(  # 获取层输出 # get layer output
                    positions=positions,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]  # 更新隐藏状态 # update hidden states

        # Capture the output of the last layer if requested.
        # layers_to_capture uses +1 offset (captures input of layer i = output of i-1),
        # so index num_layers means the output of the final layer.
        # 如果需要，捕获最后一层的输出。layers_to_capture使用+1偏移（捕获第i层的输入=第i-1层的输出），
        # 所以索引num_layers表示最后一层的输出
        if num_layers in self.layers_to_capture:
            aux_hidden_states.append(hidden_states)  # 添加最后一层输出 # append last layer output

        hidden_states = self.norm(hidden_states)  # 最终归一化 # final normalization

        if len(aux_hidden_states) == 0:  # 无辅助隐藏状态 # no auxiliary hidden states
            return hidden_states  # 仅返回隐藏状态 # return hidden states only

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态 # return hidden states and aux hidden states


class Gemma3ForCausalLM(PreTrainedModel):  # Gemma3因果语言模型类 # Gemma3 causal language model class
    config_class = Gemma3TextConfig  # 配置类 # config class

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 绑定权重的键 # tied weight keys
    _tp_plan = {"lm_head": "colwise_rep"}  # 张量并行计划 # tensor parallel plan
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}  # 流水线并行计划 # pipeline parallel plan
    config_class = Gemma3TextConfig  # 配置类 # config class
    base_model_prefix = "language_model"  # 基础模型前缀 # base model prefix

    # BitandBytes specific attributes
    # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块 # default BitandBytes target modules
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射 # BitandBytes stacked params mapping
        # shard_name, weight_name, index
        # 分片名，权重名，索引
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {  # 打包模块映射 # packed modules mapping
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    # LoRA特定属性
    supported_lora_modules = [  # 支持LoRA的模块 # LoRA supported modules
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    # Gemma不在嵌入层应用LoRA
    embedding_modules = {}  # 嵌入模块映射 # embedding modules mapping
    embedding_padding_modules = []  # 嵌入填充模块 # embedding padding modules
    supports_lora = True  # 支持LoRA # supports LoRA

    def __init__(  # 初始化方法 # initialization method
        self,
        config: Gemma3TextConfig,  # Gemma3文本配置 # Gemma3 text config
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选 # quantization config, optional
        prefix: str = "",  # 参数名前缀 # parameter name prefix
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化 # call parent class init
        self.config = config  # 保存配置 # save config
        self.quant_config = quant_config  # 保存量化配置 # save quantization config
        self.model = Gemma3TextModel(  # 创建Gemma3文本模型 # create Gemma3 text model
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器 # create logits processor

        if self.config.tie_word_embeddings:  # 是否绑定词嵌入权重 # whether to tie word embedding weights
            self.lm_head = self.model.embed_tokens  # lm_head与词嵌入共享 # lm_head shares with embed_tokens
        else:  # 不绑定 # not tied
            self.lm_head = ParallelLMHead(  # 创建并行LM头 # create parallel LM head
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态 # whether to capture aux hidden states
        self.post_init()  # 调用后初始化 # call post init

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层 # get input embedding layer
        return self.model.embed_tokens  # 返回词嵌入层 # return word embedding layer

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小 # get attention sliding window size
        return get_attention_sliding_window_size(self.config)  # 委托给模块级函数 # delegate to module-level function

    def dtype(self) -> torch.dtype:  # 获取模型数据类型 # get model data type
        return next(self.parameters()).dtype  # 返回第一个参数的数据类型 # return dtype of first parameter

    @torch.no_grad()  # 禁用梯度计算 # disable gradient computation
    def forward(  # 前向传播方法 # forward pass method
        self,
        input_ids: torch.Tensor,  # 输入token ID张量 # input token ID tensor
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选 # input embeddings, optional
        **kwargs,
    ) -> LogitsProcessor:
        hidden_states = self.model(  # 通过文本模型获取隐藏状态 # get hidden states through text model
            input_ids, positions, forward_batch, input_embeds, **kwargs
        )

        aux_hidden_states = None  # 辅助隐藏状态初始化 # initialize aux hidden states
        if self.capture_aux_hidden_states:  # 如果需要捕获辅助隐藏状态 # if need to capture aux hidden states
            hidden_states, aux_hidden_states = hidden_states  # 解包隐藏状态 # unpack hidden states

        return self.logits_processor(  # 通过logits处理器返回结果 # return result through logits processor
            input_ids,
            hidden_states,
            self.model.embed_tokens,
            forward_batch,
            aux_hidden_states,
        )

    @torch.no_grad()  # 禁用梯度计算 # disable gradient computation
    def forward_split_prefill(  # 分割预填充前向传播方法 # split prefill forward pass method
        self,
        input_ids: torch.Tensor,  # 输入token ID张量 # input token ID tensor
        positions: torch.Tensor,  # 位置编码张量 # position encoding tensor
        forward_batch: ForwardBatch,  # 前向批次信息 # forward batch info
        split_interval: Tuple[int, int],  # [start, end) 0-based # [起始, 结束) 0基索引
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选 # input embeddings, optional
    ):
        start, end = split_interval  # 解包分割区间 # unpack split interval
        # embed
        # 嵌入
        if start == 0:  # 从第0层开始 # start from layer 0
            if input_embeds is None:  # 无输入嵌入 # no input embeddings
                hidden_states = self.model.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态 # get hidden states through embedding layer
            else:  # 有输入嵌入 # has input embeddings
                hidden_states = input_embeds  # 直接使用输入嵌入 # use input embeddings directly

            if positions.dim() == 1:  # 一维位置 # 1D positions
                positions = einops.rearrange(positions, "s -> 1 s")  # 重排为二维 # rearrange to 2D
            position_embeddings_global = self.model.rotary_emb(hidden_states, positions)  # 全局位置嵌入 # global position embeddings
            position_embeddings_local = self.model.rotary_emb_local(  # 局部位置嵌入 # local position embeddings
                hidden_states, positions
            )

            forward_batch.hidden_states = hidden_states  # 保存隐藏状态到批次 # save hidden states to batch
            forward_batch.model_specific_states = {  # 保存模型特定状态 # save model-specific states
                "positions": positions,
                "position_embeddings_global": position_embeddings_global,
                "position_embeddings_local": position_embeddings_local,
            }

        # decoder layer
        # 解码器层
        for i in range(start, end):  # 遍历分割范围内的层 # iterate layers in split range
            layer = self.model.layers[i]  # 获取当前层 # get current layer
            layer_output = layer(  # 计算层输出 # compute layer output
                positions=forward_batch.model_specific_states["positions"],
                position_embeddings_global=forward_batch.model_specific_states[
                    "position_embeddings_global"
                ],
                position_embeddings_local=forward_batch.model_specific_states[
                    "position_embeddings_local"
                ],
                hidden_states=forward_batch.hidden_states,
                forward_batch=forward_batch,
            )
            forward_batch.hidden_states = layer_output[0]  # 更新隐藏状态 # update hidden states

        if end == self.model.config.num_hidden_layers:  # 到达最后一层 # reached last layer
            # norm
            # 归一化
            forward_batch.hidden_states = self.model.norm(forward_batch.hidden_states)  # 最终归一化 # final normalization

            # logits process
            # logits处理
            result = self.logits_processor(  # 通过logits处理器 # through logits processor
                input_ids,
                forward_batch.hidden_states,
                self.model.embed_tokens,
                forward_batch,
            )
        else:  # 未到达最后一层 # not reached last layer
            result = None  # 结果为空 # result is None

        return result  # 返回结果 # return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法 # load weights method
        stacked_params_mapping = [  # 堆叠参数映射 # stacked params mapping
            # (param_name, shard_name, shard_id)
            # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())  # 参数字典 # parameters dict
        loaded_params: Set[str] = set()  # 已加载参数集合 # loaded params set
        for name, loaded_weight in weights:  # 遍历权重 # iterate weights
            for param_name, shard_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射 # iterate stacked mapping
                # if param_name in name:
                # print(f"{param_name} is already in {name}")
                # 如果param_name在name中：打印信息
                if shard_name not in name:  # 分片名不在权重名中 # shard name not in weight name
                    continue  # 跳过 # skip
                name = name.replace(shard_name, param_name)  # 替换分片名为参数名 # replace shard name with param name
                # Skip loading extra bias for GPTQ models.
                # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 额外偏置 # extra bias
                    continue  # 跳过 # skip
                param = params_dict[name]  # 获取参数 # get parameter
                weight_loader = param.weight_loader  # 获取权重加载器 # get weight loader
                weight_loader(param, loaded_weight, shard_id)  # 加载权重 # load weight
                break  # 跳出内层循环 # break inner loop
            else:  # 非堆叠参数 # non-stacked params
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                # lm_head在vllm中未使用，因为它与embed_token绑定。为防止错误，跳过加载lm_head.weight
                if "lm_head.weight" in name:  # lm_head权重 # lm_head weight
                    continue  # 跳过 # skip
                # Skip loading extra bias for GPTQ models.
                # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 额外偏置 # extra bias
                    continue  # 跳过 # skip
                # Remapping the name of FP8 kv-scale.
                # 重映射FP8 kv-scale的名称
                name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射名称 # remap name
                if name is None:  # 名称无效 # name is None
                    continue  # 跳过 # skip

                param = params_dict[name]  # 获取参数 # get parameter
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器 # get weight loader
                weight_loader(param, loaded_weight)  # 加载权重 # load weight
            loaded_params.add(name)  # 添加到已加载集合 # add to loaded set
        # unloaded_params = params_dict.keys() - loaded_params
        # if unloaded_params:
        #     logger.warning(
        #         "Some weights are not initialized from checkpoints: %s", unloaded_params
        #     )
        # 未加载参数相关日志（已注释）
        return loaded_params  # 返回已加载参数集合 # return loaded params set

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):  # 设置EAGLE3需要捕获的层 # set layers to capture for EAGLE3
        if layer_ids is None:  # 未指定层ID # no layer IDs specified
            self.capture_aux_hidden_states = True  # 开启辅助隐藏状态捕获 # enable aux hidden states capture
            num_layers = self.config.num_hidden_layers  # 层数 # number of layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]  # 默认捕获第2、中间、倒数第3层 # default capture layers 2, middle, last-3
        else:  # 指定了层ID # layer IDs specified
            self.capture_aux_hidden_states = True  # 开启辅助隐藏状态捕获 # enable aux hidden states capture
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            # 这里加1是因为在sglang中，第i层取第(i-1)层的输出作为辅助隐藏状态
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 对每个层ID加1 # add 1 to each layer ID

    def _shard_weight(self, weight: torch.Tensor) -> torch.Tensor:  # 分片权重方法 # shard weight method
        """Shard a full embedding/lm_head weight along vocab dim for the current TP rank.

        Gemma3 uses nn.Embedding (unsharded) but the Eagle3 draft model uses
        VocabParallelEmbedding (sharded). This method extracts the correct
        shard so the weights can be shared.
        """
        # 沿词表维度对完整的embedding/lm_head权重进行分片，用于当前张量并行排名。
        # Gemma3使用nn.Embedding（不分片），但Eagle3草稿模型使用VocabParallelEmbedding（分片）。
        # 此方法提取正确的分片，以便权重可以共享。
        tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小 # tensor parallel size
        if tp_size <= 1:  # 无需分片 # no sharding needed
            return weight  # 返回原始权重 # return original weight
        tp_rank = get_tensor_model_parallel_rank()  # 当前并行排名 # current parallel rank
        shard_size = (weight.shape[0] + tp_size - 1) // tp_size  # 每个分片的大小 # shard size
        return weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]  # 返回当前排名对应的分片 # return shard for current rank

    def get_embed(self):  # 获取嵌入权重 # get embedding weights
        return self._shard_weight(self.model.embed_tokens.weight)  # 返回分片后的嵌入权重 # return sharded embedding weights

    def get_embed_and_head(self):  # 获取嵌入和LM头权重 # get embedding and LM head weights
        embed = self._shard_weight(self.model.embed_tokens.weight)  # 分片嵌入权重 # sharded embedding weights
        head = self._shard_weight(self.lm_head.weight)  # 分片LM头权重 # sharded LM head weights
        return embed, head  # 返回嵌入和头权重 # return embed and head weights


EntryClass = Gemma3ForCausalLM  # 模型入口类 # model entry class
