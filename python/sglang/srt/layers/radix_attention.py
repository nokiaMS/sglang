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
"""Radix attention."""
# 基数注意力机制的实现，包含注意力类型枚举和核心注意力层，支持多种注意力后端和量化配置。

from __future__ import annotations  # 启用延迟注解评估，允许类型注解引用尚未定义的类

from enum import Enum  # 导入枚举基类，用于定义注意力类型枚举
from typing import TYPE_CHECKING, Optional  # 导入类型检查常量和可选类型注解

import torch  # 导入PyTorch核心库
from torch import nn  # 导入神经网络模块

from sglang.srt.compilation.compilation_config import register_split_op  # 导入注册分割算子的装饰器，用于编译时的算子拆分
from sglang.srt.compilation.piecewise_context_manager import get_forward_context  # 导入获取前向上下文的函数，用于编译图中的上下文管理
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,  # 导入在CUDA图中以eager模式执行的装饰器
)
from sglang.srt.model_executor.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,  # 导入判断是否在可中断CUDA图中的函数
)
from sglang.srt.model_executor.forward_context import get_attn_backend  # 导入获取注意力后端的函数
from sglang.srt.utils.custom_op import register_custom_op  # 导入注册自定义算子的装饰器

if TYPE_CHECKING:  # 仅在类型检查时导入，避免运行时循环依赖
    from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息类


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """
    # 注意力类型枚举，使用字符串值以兼容torch.compile的图捕获机制

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"  # 解码器自注意力，查询/键/值来自前一层
    # Decoder bidirectional attention between image tokens
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"  # 解码器双向注意力，用于图像token之间的双向关注
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"  # 仅编码器注意力，查询/键/值来自前一层


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """
    # 基数注意力层实现，整合了多种注意力机制，支持GQA/MQA、滑动窗口、交叉注意力等功能

    def __init__(
        self,
        num_heads: int,  # 查询头数量
        head_dim: int,  # 每个注意力头的维度
        scaling: float,  # 注意力缩放因子（通常为1/sqrt(head_dim)）
        num_kv_heads: int,  # 键值头数量，支持GQA/MQA
        layer_id: int,  # 当前层的ID编号
        logit_cap: float = 0.0,  # 注意力logit的上限值，0表示不启用
        v_head_dim: int = -1,  # 值头的维度，-1表示与head_dim相同
        sliding_window_size: int = -1,  # 滑动窗口大小，-1表示不使用滑动窗口
        is_cross_attention: bool = False,  # 是否为交叉注意力
        pos_encoding_mode: str = "NONE",  # 位置编码模式
        logit_capping_method: str = "tanh",  # logit上限的方法，默认为tanh
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，None表示不量化
        attn_type: AttentionType = AttentionType.DECODER,  # 注意力类型，默认为解码器注意力
        use_irope: bool = False,  # 是否使用irope（交替旋转位置编码）
        prefix: str = "",  # 层名前缀，用于量化时的权重命名
    ):
        super().__init__()  # 调用nn.Module的初始化方法
        self.tp_q_head_num = num_heads  # 张量并行下的查询头数量
        self.tp_k_head_num = num_kv_heads  # 张量并行下的键头数量
        self.tp_v_head_num = num_kv_heads  # 张量并行下的值头数量
        self.head_dim = head_dim  # 注意力头维度
        self.qk_head_dim = head_dim  # 查询键头维度（与head_dim相同）
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim  # 值头维度，-1时使用head_dim
        self.scaling = scaling  # 注意力分数的缩放因子
        self.layer_id = layer_id  # 当前层ID
        self.logit_cap = logit_cap  # 注意力logit上限
        self.sliding_window_size = sliding_window_size or -1  # 滑动窗口大小，0也视为-1（不启用）
        self.is_cross_attention = is_cross_attention  # 是否为交叉注意力标志
        self.use_irope = use_irope  # 是否使用irope标志
        self.k_scale = None  # 键的量化缩放因子
        self.v_scale = None  # 值的量化缩放因子
        self.k_scale_float = None  # 键的浮点量化缩放因子
        self.v_scale_float = None  # 值的浮点量化缩放因子
        self.quant_method = None  # 量化方法实例

        if quant_config is not None:  # 如果提供了量化配置
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)  # 根据配置获取量化方法
        if self.quant_method is not None:  # 如果量化方法存在
            self.quant_method.create_weights(self)  # 创建量化权重
        self.attn_type = attn_type  # 保存注意力类型

        self.pos_encoding_mode = pos_encoding_mode  # 保存位置编码模式
        self.logit_capping_method = logit_capping_method  # 保存logit上限方法
        self.xai_temperature_len = -1  # XAI温度长度，-1表示不使用

    def forward(
        self,
        q,  # 查询张量
        k,  # 键张量
        v,  # 值张量
        forward_batch: ForwardBatch,  # 前向批次信息
        save_kv_cache: bool = True,  # 是否保存键值缓存，默认为True
        **kwargs,  # 其他关键字参数（如k_rope等）
    ):
        if k is not None:  # 如果键张量不为空
            # For cross-layer sharing, kv can be None
            assert v is not None  # 断言值张量也不为空（kv应同时存在）
            if "k_rope" not in kwargs:  # 如果没有使用分离的旋转位置编码键
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)  # 将键重塑为(序列长度, 键头数, 键头维度)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)  # 将值重塑为(序列长度, 值头数, 值头维度)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)  # 使用k_rope时，键重塑为值头维度（MLA场景）

        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:  # 在扩展模式且有前向上下文时
            if self.qk_head_dim != self.v_head_dim:  # 如果查询键维度与值维度不同（如MLA）
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))  # 预分配输出张量，维度为查询头数*值头维度
            else:
                output = torch.empty_like(q)  # 维度相同时，创建与查询相同形状的输出张量
            if is_in_breakable_cuda_graph():  # 如果在可中断CUDA图内执行
                bcg_unified_attention_with_output(  # 调用CUDA图兼容的统一注意力函数
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            else:
                unified_attention_with_output(  # 否则调用普通的统一注意力函数
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            return output  # 返回注意力输出
        else:
            return get_attn_backend().forward(  # 非扩展模式或无前向上下文时，直接调用注意力后端的forward
                q,
                k,
                v,
                self,  # 传入自身作为注意力层参数
                forward_batch,
                save_kv_cache,
                **kwargs,
            )


@register_custom_op(mutates_args=["output"])  # 注册为自定义算子，声明output参数会被修改
@register_split_op()  # 注册为可分割算子，用于编译时的图分割
def unified_attention_with_output(
    query: torch.Tensor,  # 查询张量
    key: Optional[torch.Tensor],  # 键张量，可选
    value: Optional[torch.Tensor],  # 值张量，可选
    output: torch.Tensor,  # 输出张量（预分配）
    save_kv_cache: bool,  # 是否保存键值缓存
    layer_id: int,  # 层ID
    *,
    q_rope: Optional[torch.Tensor] = None,  # 查询的旋转位置编码
    k_rope: Optional[torch.Tensor] = None,  # 键的旋转位置编码
    sinks: Optional[torch.Tensor] = None,  # 注意力汇聚点张量
    # MLA / TRT-LLM / NSA paths pass these through RadixAttention.forward(**kwargs);
    # they must appear in the schema when --enforce-piecewise-cuda-graph is on.
    cos_sin_cache: Optional[torch.Tensor] = None,  # 余弦正弦缓存，用于旋转位置编码
    is_neox: Optional[bool] = None,  # 是否使用Neox风格的位置编码
    llama_4_scaling: Optional[torch.Tensor] = None,  # Llama4缩放因子
    topk_indices: Optional[torch.Tensor] = None,  # TopK索引，用于NSA等稀疏注意力
) -> None:
    context = get_forward_context()  # 获取当前的前向计算上下文
    forward_batch = context.forward_batch  # 从上下文中获取前向批次信息
    attention_layers = context.attention_layers  # 从上下文中获取所有注意力层
    attention_layer = attention_layers[layer_id]  # 根据层ID获取当前注意力层
    real_num_tokens = forward_batch.num_token_non_padded_cpu  # 获取实际非填充token数量

    query = query[:real_num_tokens]  # 截取查询张量到实际token数，去除填充部分
    if key is not None:  # 如果键张量不为空
        key = key[:real_num_tokens]  # 截取键张量到实际token数
    if value is not None:  # 如果值张量不为空
        value = value[:real_num_tokens]  # 截取值张量到实际token数

    kwargs = {}  # 初始化额外参数字典
    if q_rope is not None:  # 如果查询旋转位置编码存在
        kwargs["q_rope"] = q_rope[:real_num_tokens]  # 截取并添加到参数字典
    if k_rope is not None:  # 如果键旋转位置编码存在
        kwargs["k_rope"] = k_rope[:real_num_tokens]  # 截取并添加到参数字典
    if sinks is not None:  # 如果注意力汇聚点存在
        kwargs["sinks"] = sinks  # 添加到参数字典（无需截取，不是按token的）
    if cos_sin_cache is not None:  # 如果余弦正弦缓存存在
        kwargs["cos_sin_cache"] = cos_sin_cache  # 添加到参数字典（全局缓存，无需截取）
    if is_neox is not None:  # 如果Neox标志存在
        kwargs["is_neox"] = is_neox  # 添加到参数字典
    if llama_4_scaling is not None:  # 如果Llama4缩放因子存在
        kwargs["llama_4_scaling"] = llama_4_scaling  # 添加到参数字典
    if topk_indices is not None:  # 如果TopK索引存在
        kwargs["topk_indices"] = topk_indices[:real_num_tokens]  # 截取并添加到参数字典

    original_out_cache_loc = forward_batch.out_cache_loc  # 保存原始的输出缓存位置索引
    # Keep the original ForwardBatch object and only narrow cache locations for
    # this backend call so model/backend state is still written to the same batch.
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]  # 缩窄缓存位置索引到实际token数，保持批次对象不变

    # Store pre-allocated output for FA backend to write directly into.
    # Must slice to real_num_tokens to match the narrowed query shape —
    # the FA kernel validates out.size(0) == q.size(0).
    forward_batch._attn_output = output[:real_num_tokens]  # 将预分配输出切片后存储，供FlashAttention后端直接写入

    ret = get_attn_backend().forward(  # 调用注意力后端的forward方法
        query,
        key,
        value,
        attention_layer,  # 传入当前注意力层
        forward_batch,
        save_kv_cache,
        **kwargs,
    )
    forward_batch.out_cache_loc = original_out_cache_loc  # 恢复原始的输出缓存位置索引

    if ret.data_ptr() != output.data_ptr():  # 如果返回张量与预分配输出的数据指针不同（后端未直接写入预分配输出）
        output[:real_num_tokens].view(ret.shape).copy_(ret)  # 将返回结果拷贝到预分配输出中
    return  # 函数无返回值，结果通过output参数传出


bcg_unified_attention_with_output = eager_on_graph(True)(unified_attention_with_output)  # 创建可中断CUDA图兼容版本的统一注意力函数，在CUDA图中以eager模式执行
