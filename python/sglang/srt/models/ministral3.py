# Ministral 3 模型实现
# 本文件实现了Ministral 3模型，基于Llama架构扩展，
# 添加了Llama 4风格的注意力缩放（scaling beta）和滑动窗口支持。
# Ministral 3使用RoPE位置编码参数来配置注意力缩放。

from typing import Any, Dict, Optional  # 导入类型提示

import torch  # 导入PyTorch
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.models.llama import (  # 导入Llama模型组件
    LlamaAttention,  # Llama注意力层
    LlamaDecoderLayer,  # Llama解码器层
    LlamaForCausalLM,  # Llama因果语言模型
    LlamaModel,  # Llama模型
)
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加工具和层构建工具


def _get_llama_4_attn_scale(  # 计算Llama 4风格的注意力缩放因子
    positions_ids: torch.Tensor, beta: float, max_position_embeddings: int  # 位置ID、缩放beta参数、最大位置嵌入数
) -> torch.Tensor:  # 返回缩放因子张量
    scaling = 1 + beta * torch.log(  # 缩放公式：1 + beta * log(1 + floor(pos / max_pos))
        1 + torch.floor(positions_ids / max_position_embeddings)  # 计算位置除以最大位置嵌入数的向下取整值
    )
    return scaling.unsqueeze(-1)  # 在最后一个维度增加维度以便广播


class Ministral3Attention(LlamaAttention):  # Ministral 3注意力层，继承自LlamaAttention
    def __init__(  # 初始化方法
        self,  # 自身实例
        config: PretrainedConfig,  # 预训练配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID，默认0
        rope_theta: float = 1000000.0,  # RoPE theta参数，默认1000000.0
        rope_scaling: Optional[Dict[str, Any]] = {},  # RoPE缩放配置，默认空字典
        rope_is_neox_style: bool = True,  # RoPE是否使用Neox风格，默认True
        max_position_embeddings: int = 8192,  # 最大位置嵌入数，默认8192
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认无
        prefix: str = "",  # 前缀字符串，默认空
        bias: bool = False,  # 是否使用偏置，默认False
    ) -> None:
        super().__init__(  # 调用父类LlamaAttention的初始化
            config,  # 配置
            hidden_size,  # 隐藏层大小
            num_heads,  # 注意力头数
            num_kv_heads,  # KV头数
            layer_id,  # 层ID
            rope_theta,  # RoPE theta参数
            rope_scaling,  # RoPE缩放配置
            rope_is_neox_style,  # RoPE Neox风格标志
            max_position_embeddings,  # 最大位置嵌入数
            quant_config,  # 量化配置
            prefix,  # 前缀
            bias,  # 偏置标志
        )
        # Ministral3 specific: llama 4 style scaling beta  # Ministral3特有：Llama 4风格缩放beta参数
        self.llama_4_scaling_beta = config.rope_parameters.get("llama_4_scaling_beta")  # 从RoPE参数中获取缩放beta值

        # sliding window  # 滑动窗口
        self.sliding_window = getattr(config, "sliding_window", None)  # 获取滑动窗口大小，默认无
        if self.sliding_window is not None:  # 如果设置了滑动窗口
            # Update RadixAttention with sliding window if needed  # 如果需要，使用滑动窗口更新RadixAttention
            # currently RadixAttention in sglang handles this mostly via logic in forward/flashinfer  # 目前SGLang中RadixAttention主要通过forward/flashinfer中的逻辑处理
            pass  # 暂不处理

    def forward(  # 前向传播方法
        self,  # 自身实例
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:  # 返回输出张量
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影获取查询、键、值
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 将QKV分割为查询、键、值

        # Apply RoPE  # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)  # 对查询和键应用旋转位置编码

        # Ministral3 / Llama 4 scaling  # Ministral3 / Llama 4缩放
        if self.llama_4_scaling_beta is not None:  # 如果配置了Llama 4缩放beta
            scale = _get_llama_4_attn_scale(  # 计算注意力缩放因子
                positions, self.llama_4_scaling_beta, self.max_position_embeddings  # 传入位置、beta和最大位置嵌入数
            ).to(q.dtype)  # 转换为查询的数据类型
            # q shape is [batch_size * seq_len, num_heads * head_dim] or [batch_size * seq_len, num_heads, head_dim]  # q的形状为[批次*序列长度, 头数*头维度]或[批次*序列长度, 头数, 头维度]
            # positions is [batch_size * seq_len]  # positions的形状为[批次*序列长度]
            # scale is [batch_size * seq_len, 1]  # scale的形状为[批次*序列长度, 1]
            # We need to reshape q to apply scale correctly if it's flattened  # 如果q被展平，需要重塑形状以正确应用缩放
            # Assuming q is (total_tokens, num_heads * head_dim)  # 假设q的形状为(总token数, 头数*头维度)
            q = q.view(-1, self.num_heads, self.head_dim)  # 重塑q为(总token数, 头数, 头维度)
            q = q * scale.unsqueeze(1)  # Broadcast over heads  # 在头维度上广播缩放因子
            q = q.view(-1, self.num_heads * self.head_dim)  # 重塑回展平形状

        attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力层计算注意力输出
        output, _ = self.o_proj(attn_output)  # 通过输出投影获取最终输出
        return output  # 返回输出


class Ministral3DecoderLayer(LlamaDecoderLayer):  # Ministral 3解码器层，继承自LlamaDecoderLayer
    def __init__(self, config, layer_id=0, quant_config=None, prefix=""):  # 初始化方法
        super().__init__(config, layer_id, quant_config, prefix)  # 调用父类初始化
        self.self_attn = Ministral3Attention(  # 使用Ministral3注意力层替换默认注意力层
            config=config,  # 配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=config.rope_parameters["rope_theta"],  # 从RoPE参数获取theta
            rope_scaling=config.rope_parameters,  # rope_scaling is rope_parameters in Ministral3Config  # 在Ministral3Config中rope_scaling即为rope_parameters
            max_position_embeddings=getattr(  # 最大位置嵌入数
                config, "original_max_position_embeddings", 16384  # 默认16384
            ),
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 带前缀的注意力层名称
            bias=getattr(config, "attention_bias", False)  # 是否使用注意力偏置
            or getattr(config, "bias", False),  # 或是否使用通用偏置
        )


class Ministral3Model(LlamaModel):  # Ministral 3模型，继承自LlamaModel
    def __init__(  # 初始化方法
        self,  # 自身实例
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认无
        prefix: str = "",  # 前缀字符串，默认空
    ) -> None:
        # Override layer creation to use Ministral3Attention  # 覆盖层创建以使用Ministral3Attention
        super().__init__(config, quant_config, prefix)  # 调用父类初始化

        self.layers, self.start_layer, self.end_layer = make_layers(  # 使用make_layers创建解码器层
            config.num_hidden_layers,  # 隐藏层数量
            lambda idx, prefix: Ministral3DecoderLayer(  # 使用Ministral3解码器层
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix  # 传递配置、量化配置、层ID和前缀
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行排名
            pp_size=self.pp_group.world_size,  # 流水线并行世界大小
            prefix="model.layers",  # 层前缀
        )


class Ministral3ForCausalLM(LlamaForCausalLM):  # Ministral 3因果语言模型，继承自LlamaForCausalLM
    def _init_model(  # 初始化模型方法
        self,  # 自身实例
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认无
        prefix: str = "",  # 前缀字符串，默认空
    ):
        return Ministral3Model(config, quant_config, prefix=prefix)  # 返回Ministral3模型实例


EntryClass = [Ministral3ForCausalLM]  # 模型注册入口类列表
