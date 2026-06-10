# HunYuan MoE模型推理实现 - 基于稀疏专家混合架构的混元模型，兼容HuggingFace权重，仅用于推理
# coding=utf-8  # 编码格式
# Copyright 2024 The HunYuan team.  # 版权所有 2024 混元团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证2.0版链接
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 软件按"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 参见许可证了解管理权限和
# limitations under the License.  # 限制的特定语言
"""Inference-only HunYuan model compatible with HuggingFace weights."""  # 仅推理的混元模型，兼容HuggingFace权重

import re  # 正则表达式模块
from typing import Any, Dict, Iterable, Optional, Tuple  # 类型提示导入

import torch  # PyTorch深度学习框架
from torch import nn  # 神经网络模块
from transformers import PretrainedConfig  # 预训练配置基类

from sglang.srt.distributed import (  # 分布式工具导入
    get_tensor_model_parallel_rank,  # 获取张量并行rank
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import ExpertDistributionRecorder  # 专家分布记录器
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活函数与乘法
from sglang.srt.layers.layernorm import RMSNorm  # RMS归一化层
from sglang.srt.layers.linear import (  # 线性层导入
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 融合MoE Triton内核
from sglang.srt.layers.moe.topk import TopK  # Top-K选择模块
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力机制
from sglang.srt.layers.rotary_embedding import get_rope  # 获取旋转位置编码
from sglang.srt.layers.sampler import create_sampler  # 创建采样器
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 权重加载工具
    default_weight_loader,  # 默认权重加载器
    kv_cache_scales_loader,  # KV缓存缩放加载器
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称
)
from sglang.srt.utils import is_hip  # 是否为HIP（AMD GPU）
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 获取RoPE配置

expert_distribution_recorder = ExpertDistributionRecorder()  # 创建专家分布记录器实例


def _is_moe(config: PretrainedConfig) -> bool:  # 判断模型是否使用MoE
    if getattr(config, "num_experts", None) and (  # 如果配置有专家数量且
        (isinstance(config.num_experts, int) and config.num_experts > 1)  # 是整数且大于1
        or (isinstance(config.num_experts, list) and max(config.num_experts) > 1)  # 或列表中最大值大于1
    ):
        return True  # 是MoE模型
    else:  # 否则
        return False  # 不是MoE模型


def _get_cla_factor(config: PretrainedConfig) -> int:  # 获取CLA（跨层注意力）因子
    if not getattr(config, "use_cla", False):  # 如果不使用CLA
        return 1  # 返回1（不共享）
    return getattr(config, "cla_share_factor", 1)  # 返回CLA共享因子


class HunYuanMLP(nn.Module):  # 混元MLP模块

    def __init__(  # 初始化MLP
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
        self.gate_up_proj = MergedColumnParallelLinear(  # 创建gate和up合并投影层
            input_size=hidden_size,  # 输入大小
            output_sizes=[intermediate_size] * 2,  # 输出大小（gate和up各一个）
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.gate_up_proj",  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # 创建down投影层
            input_size=intermediate_size,  # 输入大小
            output_size=hidden_size,  # 输出大小
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.down_proj",  # 参数前缀
            reduce_results=reduce_results,  # 是否归约结果
        )
        if hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出错误
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."  # 目前只支持silu
            )
        self.act_fn = SiluAndMul()  # 创建SiLU激活与乘法函数

    def forward(self, x):  # MLP前向传播
        gate_up, _ = self.gate_up_proj(x)  # 计算gate和up投影
        x = self.act_fn(gate_up)  # 应用激活函数
        x, _ = self.down_proj(x)  # 计算down投影
        return x  # 返回结果


class HunYuanSparseMoeBlock(nn.Module):  # 混元稀疏MoE块

    def __init__(  # 初始化稀疏MoE块
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        layer_id: int = -1,  # 层ID
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小

        if self.tp_size > config.num_experts:  # 如果TP大小大于专家数量
            raise ValueError(  # 抛出错误
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."  # TP大小不能大于专家数量
            )

        # Get layer_id topk if config.moe_topk is a list  # 如果moe_topk是列表，获取对应层的top-k值
        if isinstance(config.moe_topk, list):  # 如果moe_topk是列表
            assert layer_id >= 0  # 断言层ID有效
            assert len(config.moe_topk) > layer_id  # 断言列表长度足够
            top_k = config.moe_topk[layer_id]  # 获取对应层的top-k值
        else:  # 否则
            top_k = config.moe_topk  # 使用统一的top-k值

        # If it is moe, moe_intermediate_size is preferred  # 如果是MoE，优先使用moe_intermediate_size
        intermediate_size = config.intermediate_size  # 默认中间层大小
        if config.moe_intermediate_size is not None:  # 如果配置了MoE中间层大小
            intermediate_size = (  # 使用MoE中间层大小
                config.moe_intermediate_size  # 如果是整数直接使用
                if isinstance(config.moe_intermediate_size, int)
                else config.moe_intermediate_size[layer_id]  # 否则获取对应层的值
            )

        self.topk = TopK(  # 创建Top-K选择模块
            top_k=top_k,  # top-k值
            layer_id=layer_id,  # 层ID
            renormalize=True if top_k > 1 else False,  # top_k>1时重新归一化
        )

        self.experts = FusedMoE(  # 创建融合MoE专家层
            num_experts=config.num_experts,  # 专家数量
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=intermediate_size,  # 中间层大小
            reduce_results=False,  # 不归约结果（后续手动归约）
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
        )

        self.gate = ReplicatedLinear(  # 创建门控线性层（路由器）
            config.hidden_size, config.num_experts, bias=False, quant_config=None  # 无偏置、不量化
        )
        if config.use_mixed_mlp_moe > 0:  # 如果使用混合MLP+MoE
            # Get layer_id num_shared_expert if config.num_shared_expert is a list  # 如果num_shared_expert是列表，获取对应层的共享专家数
            if isinstance(config.num_shared_expert, list):  # 如果是列表
                assert layer_id >= 0  # 断言层ID有效
                assert len(config.num_shared_expert) > layer_id  # 断言列表长度足够
                num_shared_expert = config.num_shared_expert[layer_id]  # 获取对应层的共享专家数
            else:  # 否则
                num_shared_expert = config.num_shared_expert  # 使用统一值

            self.shared_mlp = HunYuanMLP(  # 创建共享MLP
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=config.intermediate_size * num_shared_expert,  # 共享MLP中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                reduce_results=False,  # 不归约结果
            )
        else:  # 否则不使用共享MLP
            self.shared_mlp = None  # 设为None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 稀疏MoE前向传播
        # NOTE: hidden_states can have either 1D or 2D shape.  # 注意：hidden_states可以是1D或2D形状
        orig_shape = hidden_states.shape  # 保存原始形状
        hidden_dim = hidden_states.shape[-1]  # 获取隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑为2D
        shared_output = None  # 共享输出初始化
        if self.shared_mlp is not None:  # 如果有共享MLP
            shared_output = self.shared_mlp(hidden_states)  # 计算共享MLP输出

        # router_logits: (num_tokens, n_experts)  # 路由器logits形状：(token数, 专家数)
        router_logits, _ = self.gate(hidden_states)  # 计算路由器logits
        topk_output = self.topk(hidden_states, router_logits)  # 获取top-k选择结果
        final_hidden_states = self.experts(hidden_states, topk_output)  # 执行专家计算
        if shared_output is not None:  # 如果有共享MLP输出
            final_hidden_states = final_hidden_states + shared_output  # 加上共享MLP输出
        if self.tp_size > 1:  # 如果TP大小大于1
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 全归约

        return final_hidden_states.view(orig_shape)  # 恢复原始形状并返回


def get_head_dim(config):  # 获取头维度
    if hasattr(config, "head_dim"):  # 如果有head_dim属性
        return int(config.head_dim)  # 返回head_dim
    if hasattr(config, "attention_head_dim"):  # 如果有attention_head_dim属性
        return int(config.attention_head_dim)  # 返回attention_head_dim

    # since some hunyuan model don't follow the self.hidden_size // self.total_num_heads rule
    # wrong setting may cause runtime error, just throw error if this field is missing.
    # 因为一些混元模型不遵循 hidden_size // total_num_heads 的规则，
    # 错误的设置可能导致运行时错误，如果缺少此字段则直接抛出错误。
    raise ValueError("Missing head dim config, try set head_dim in config.json")  # 抛出配置缺失错误


def check_head_dim(config):  # 检查头维度一致性
    # Some models may lack `head_dim` and use `attention_head_dim` instead.
    # This attribute is also used by flashinfer_backend.py, so we check for
    # consistency and raise an error if it's not met to avoid silent failures.
    # Although we could adapt the HunYuan model to use `attention_head_dim`,
    # flashinfer expects `head_dim`, so we enforce its presence for correctness.
    # 一些模型可能缺少head_dim而使用attention_head_dim。
    # 此属性也被flashinfer_backend.py使用，因此我们检查一致性并在不满足时抛出错误以避免静默失败。
    # 虽然我们可以让混元模型使用attention_head_dim，但flashinfer期望head_dim，因此为了正确性我们强制要求其存在。
    calc_head_dim = config.hidden_size // config.num_attention_heads  # 计算头维度

    if hasattr(config, "attention_head_dim"):  # 如果有attention_head_dim
        if calc_head_dim != config.attention_head_dim and not hasattr(  # 如果计算值与配置值不匹配且没有head_dim
            config, "head_dim"
        ):
            # in this case, flash infer(and other components may calculate wrong value.)  # 在这种情况下，flashinfer和其他组件可能计算错误
            raise ValueError(  # 抛出配置错误
                f"HunYuan model config error: calculated head_dim {calc_head_dim} != attention_head_dim {config.attention_head_dim}"
                + f"\nPlease Add head_dim:{config.attention_head_dim} in config.json to make sure correctly inference."  # 请在config.json中添加head_dim
            )

        if hasattr(config, "head_dim") and config.attention_head_dim != config.head_dim:  # 如果两个属性不一致
            raise ValueError(  # 抛出配置错误
                f"HunYuan model config error: head_dim({config.head_dim}) != attention_head_dim({config.attention_head_dim})"
                + f"\nPlease change head_dim:{config.attention_head_dim} in config.json to make sure correctly inference."  # 请修改config.json中的head_dim
            )


class HunYuanAttention(nn.Module):  # 混元注意力模块

    def __init__(  # 初始化注意力
        self,
        config: PretrainedConfig,  # 模型配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数量
        num_kv_heads: int,  # KV头数量
        rope_theta: float = 10000,  # RoPE基数
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置
        max_position_embeddings: int = 8192,  # 最大位置嵌入数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        bias: bool = False,  # 是否使用偏置
        prefix: str = "",  # 参数前缀
        attention_type: str = "self",  # 注意力类型（self或cross）
        layer_id: int = -1,  # 层ID
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个rank的注意力头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.  # KV头数大于TP大小，因此在多个张量并行GPU上划分KV头
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.  # KV头数小于TP大小，因此在多个张量并行GPU上复制KV头
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个rank的KV头数
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        # Prioritize `head_dim` but fall back to `attention_head_dim` for Hunyuan models.
        # MistralConfig有一个可选的head_dim由Mistral-Nemo引入。
        # 优先使用head_dim，但对混元模型回退到attention_head_dim。
        self.head_dim = get_head_dim(config)  # 获取头维度

        check_head_dim(config)  # 检查头维度一致性

        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.rope_theta = rope_theta  # 保存RoPE基数
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置嵌入数
        self.use_qk_norm = getattr(config, "use_qk_norm", False)  # 是否使用QK归一化
        self.attention_type = attention_type  # 保存注意力类型
        self.layer_id = layer_id  # 保存层ID

        if attention_type == "self":  # 如果是自注意力
            self.qkv_proj = QKVParallelLinear(  # 创建QKV并行投影层
                hidden_size=hidden_size,  # 输入大小
                head_size=self.head_dim,  # 每个头的大小
                total_num_heads=self.total_num_heads,  # 总Q头数
                total_num_kv_heads=self.total_num_kv_heads,  # 总KV头数
                bias=bias,  # 是否使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.qkv_proj",  # 参数前缀
            )
        elif attention_type == "cross":  # 如果是交叉注意力
            self.q_proj = ColumnParallelLinear(  # 创建Q投影层
                hidden_size,  # 输入大小
                hidden_size,  # 输出大小
                bias=bias,  # 是否使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.q_proj",  # 参数前缀
            )
        else:  # 否则
            raise RuntimeError("Not support attnention type")  # 抛出不支持的注意力类型错误

        self.o_proj = RowParallelLinear(  # 创建输出投影层
            input_size=self.total_num_heads * self.head_dim,  # 输入大小
            output_size=hidden_size,  # 输出大小
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.o_proj",  # 参数前缀
        )

        is_neox_style = True  # 默认使用Neox风格
        if quant_config is not None and quant_config.get_name() == "gguf":  # 如果是GGUF量化
            is_neox_style = False  # 不使用Neox风格

        self.rotary_emb = get_rope(  # 创建旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # RoPE基数
            rope_scaling=rope_scaling,  # RoPE缩放配置
            is_neox_style=is_neox_style,  # 是否为Neox风格
        )
        self.attn = RadixAttention(  # 创建基数注意力层
            self.num_heads,  # 注意力头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            prefix=f"{prefix}.attn",  # 参数前缀
        )

        if self.use_qk_norm:  # 如果使用QK归一化
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q归一化层
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K归一化层

    def forward(  # 注意力前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        kv_states: Optional[Tuple[torch.Tensor]] = None,  # KV状态（用于交叉注意力）
    ) -> torch.Tensor:
        if self.attention_type == "self":  # 如果是自注意力
            qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
            ori_k = k  # 保存原始K（用于CLA）
            if self.use_qk_norm:  # 如果使用QK归一化
                # q = self.query_layernorm(q.view(-1, self.num_heads, self.head_dim).contiguous())
                # k = self.key_layernorm(k.view(-1, self.num_kv_heads, self.head_dim).contiguous())
                q = self.query_layernorm(q.reshape(-1, self.head_dim).contiguous())  # 对Q进行RMS归一化
                k = self.key_layernorm(k.reshape(-1, self.head_dim).contiguous())  # 对K进行RMS归一化
        elif self.attention_type == "cross":  # 如果是交叉注意力
            assert kv_states is not None  # 断言KV状态不为None
            ori_k, v = kv_states  # use last layer kv,  # 使用上一层的KV
            k = ori_k  # 使用原始K
            q, _ = self.q_proj(hidden_states)  # 计算Q投影
            k_tmp = torch.empty_like(k)  # Todo: reduant rotary embedding  # 待办：冗余的旋转位置编码
            q, _ = self.rotary_emb(positions, q, k_tmp)  # 应用旋转位置编码
            if self.use_qk_norm:  # 如果使用QK归一化
                q = self.query_layernorm(  # 对Q进行RMS归一化
                    q.view(-1, self.num_heads, self.head_dim).contiguous()
                )
                k = self.key_layernorm(  # 对K进行RMS归一化
                    k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
                )
        else:  # 否则
            raise RuntimeError("Not support attnention type")  # 抛出不支持的注意力类型错误

        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output, (ori_k, v)  # 返回输出和KV状态


class HunYuanDecoderLayer(nn.Module):  # 混元解码器层

    def __init__(  # 初始化解码器层
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        layer_id: int = -1,  # 层ID
    ) -> None:
        super().__init__()  # 调用父类初始化
        assert layer_id >= 0  # 断言层ID有效
        self.layer_id = layer_id  # 保存层ID
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.intermediate_size = (  # 中间层大小
            config.intermediate_size  # 如果是整数直接使用
            if isinstance(config.intermediate_size, int)
            else config.intermediate_size[layer_id]  # 否则获取对应层的值
        )
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置
        if rope_scaling is not None and getattr(  # 如果有RoPE缩放且有原始最大位置嵌入数
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 添加原始最大位置嵌入数
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入数
        # Support abacusai/Smaug-72B-v0.1 with attention_bias  # 支持带有attention_bias的abacusai/Smaug-72B-v0.1
        # Support internlm/internlm-7b with bias  # 支持带有bias的internlm/internlm-7b
        attention_bias = getattr(config, "attention_bias", False) or getattr(  # 获取注意力偏置
            config, "bias", False
        )
        cla_factor = _get_cla_factor(config)  # 获取CLA因子
        attention_type = (  # 确定注意力类型
            "cross" if layer_id >= 0 and layer_id % cla_factor != 0 else "self"  # 非CLA共享层使用交叉注意力
        )
        self.self_attn = HunYuanAttention(  # 创建自注意力层
            config=config,  # 配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=getattr(  # KV头数
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,  # RoPE基数
            rope_scaling=rope_scaling,  # RoPE缩放配置
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入数
            quant_config=quant_config,  # 量化配置
            bias=attention_bias,  # 注意力偏置
            prefix=f"{prefix}.self_attn",  # 参数前缀
            attention_type=attention_type,  # 注意力类型
            layer_id=layer_id,  # 层ID
        )
        if _is_moe(config):  # 如果是MoE模型
            self.mlp = HunYuanSparseMoeBlock(  # 创建稀疏MoE块
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                layer_id=layer_id,  # 层ID
            )
        else:  # 否则是Dense模型
            self.mlp = HunYuanMLP(  # 创建MLP
                hidden_size=self.hidden_size,  # 隐藏层大小
                intermediate_size=self.intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                bias=getattr(config, "mlp_bias", False),  # MLP偏置
                prefix=f"{prefix}.mlp",  # 参数前缀
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 解码器层前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差
        kv_states: Optional[Tuple[torch.Tensor]] = None,  # KV状态（用于CLA）
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention  # 自注意力
        if residual is None:  # 如果没有残差
            residual = hidden_states  # 当前隐藏状态作为残差
            hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        else:  # 否则有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 归一化（带残差）
        hidden_states, ori_kv_states = self.self_attn(  # 执行自注意力计算
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
            kv_states=kv_states,  # KV状态
        )

        # Fully Connected  # 全连接层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后归一化
        hidden_states = self.mlp(hidden_states)  # 执行MLP/MoE计算
        return hidden_states, residual, ori_kv_states  # 返回隐藏状态、残差和KV状态


class HunYuanModel(nn.Module):  # 混元模型主体

    def __init__(  # 初始化模型
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.org_vocab_size = config.vocab_size  # 原始词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 创建词嵌入层
            self.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
        )

        self.layers = nn.ModuleList(  # 创建解码器层列表
            [
                HunYuanDecoderLayer(  # 每一层都是混元解码器层
                    config=config,  # 配置
                    layer_id=layer_id,  # 层ID
                    quant_config=quant_config,  # 量化配置
                    # prefix=prefix  # 参数前缀（已注释）
                )
                for layer_id in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.embed_tokens(input_ids)  # 返回词嵌入结果

    def forward(  # 模型前向传播
        self,
        input_ids: Optional[torch.Tensor],  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        if input_embeds is not None:  # 如果提供了输入嵌入
            hidden_states = input_embeds  # 直接使用输入嵌入
        else:  # 否则
            hidden_states = self.get_input_embeddings(input_ids)  # 从token ID获取嵌入
        residual = None  # 残差初始化为None

        prev_kv_states = None  # 上一层的KV状态（用于CLA）
        for i in range(len(self.layers)):  # 遍历所有解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual, kv_states = layer(  # 执行当前层前向传播
                positions,  # 位置编码
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
                residual,  # 残差
                prev_kv_states,  # 上一层的KV状态
            )

            if False:  # (i - self.start_layer) % cla_factor == 0:  # 条件永远为False，CLA功能暂未启用
                prev_kv_states = kv_states  # 保存KV状态
            else:  # 否则
                prev_kv_states = None  # 不保存KV状态

        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class HunYuanMoEV1ForCausalLM(nn.Module):  # 混元MoE V1因果语言模型
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [  # QKV投影打包
            "q_proj",  # Q投影
            "k_proj",  # K投影
            "v_proj",  # V投影
        ],
        "gate_up_proj": [  # gate_up投影打包
            "gate_proj",  # gate投影
            "up_proj",  # up投影
        ],
    }

    embedding_modules = {  # 嵌入模块映射
        "embed_tokens": "input_embeddings",  # 输入嵌入
        "lm_head": "output_embeddings",  # 输出嵌入
    }
    embedding_padding_modules = ["lm_head"]  # 嵌入填充模块
    bitsandbytes_stacked_params_mapping = {  # bitsandbytes堆叠参数映射
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),  # Q投影索引0
        "k_proj": ("qkv_proj", 1),  # K投影索引1
        "v_proj": ("qkv_proj", 2),  # V投影索引2
        "gate_proj": ("gate_up_proj", 0),  # gate投影索引0
        "up_proj": ("gate_up_proj", 1),  # up投影索引1
    }

    def __init__(  # 初始化因果语言模型
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        self.model = HunYuanModel(config, quant_config, prefix="model")  # 创建模型主体
        self.unpadded_vocab_size = config.vocab_size  # 未填充词表大小
        self.lm_head = ParallelLMHead(  # 创建语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
        )
        if config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head.weight = self.model.embed_tokens.weight  # 共享权重

        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        self.head_dim = get_head_dim(config)  # 获取头维度

        check_head_dim(config)  # 检查头维度一致性

        logit_scale = getattr(config, "logit_scale", 1.0)  # 获取logit缩放因子
        self.logits_processor = LogitsProcessor(config, logit_scale=logit_scale)  # 创建logits处理器
        self.sampler = create_sampler()  # 创建采样器

    def forward(  # 因果语言模型前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 模型前向传播
        return self.logits_processor(  # 返回logits处理结果
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def _split_qkv_weight(self, qkv: torch.Tensor):  # 拆分QKV权重
        num_attention_heads = self.config.num_attention_heads  # 注意力头数
        num_kv_heads = getattr(  # KV头数
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        num_key_value_groups = num_attention_heads // num_kv_heads  # 每组KV的头数

        qkv = qkv.reshape(  # 重塑QKV权重
            num_kv_heads, num_key_value_groups + 2, self.head_dim, self.hidden_size
        )
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)  # 拆分Q、K、V
        q = q.reshape(-1, self.hidden_size)  # 重塑Q
        k = k.reshape(-1, self.hidden_size)  # 重塑K
        v = v.reshape(-1, self.hidden_size)  # 重塑V
        return torch.concat((q, k, v))  # 拼接并返回
        # return qkv.reshape((num_kv_heads, num_key_value_groups+2 , attention_head_dim, hidden_size)).permute((1,0,2,3)).reshape((-1, hidden_size)),
        # 备用实现：使用permute重排维度

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        cla_factor = _get_cla_factor(self.config)  # 获取CLA因子
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
            (".gate_up_proj", ".gate_proj", 0),  # gate投影映射
            (".gate_up_proj", ".up_proj", 1),  # up投影映射
        ]

        num_attention_heads = self.config.num_attention_heads  # 注意力头数
        num_kv_heads = getattr(  # KV头数
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        split_params_mapping = [  # 拆分参数映射
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),  # gate和up合并参数
            (
                ".qkv_proj",  # QKV参数名
                ".qkv_proj",  # 权重名
                num_attention_heads + num_kv_heads * 2,  # 总分割数
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],  # 各分片信息
                self._split_qkv_weight,  # 拆分函数
            ),
        ]

        if _is_moe(self.config):  # 如果是MoE模型
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)  # 权重、fp8权重缩放和fp8激活缩放的参数映射
            expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
                ckpt_gate_proj_name="gate_proj",  # 检查点gate投影名称
                ckpt_down_proj_name="down_proj",  # 检查点down投影名称
                ckpt_up_proj_name="up_proj",  # 检查点up投影名称
                num_experts=self.config.num_experts,  # 专家数量
            )
        else:  # 否则不是MoE模型
            expert_params_mapping = {}  # 空映射

        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码逆频率
                continue  # 跳过
            if "gate_proj_bias" in name:  # 如果是gate投影偏置
                name = name.replace("gate_proj_bias", "gate_proj.bias")  # 重命名
            if "up_proj_bias" in name:  # 如果是up投影偏置
                name = name.replace("up_proj_bias", "up_proj.bias")  # 重命名
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 如果是缓存的cos/sin
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.  # 使用ColossalAI训练的模型可能在检查点中包含这些张量，跳过它们
                continue  # 跳过
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            # 当tie_word_embeddings时，可以跳过lm_head.weight。
            # 如果模型经过量化、LoRA、微调等处理，该权重可能不必要地出现在文件中。
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 如果绑定词嵌入且是lm_head权重
                continue  # 跳过

            is_found = False  # 是否找到匹配标志
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 跳过
                if "mlp.experts" in name:  # 如果是专家参数
                    continue  # 跳过（专家参数单独处理）
                # cross layer only have q_proj, skip qkv pack  # 交叉注意力层只有q_proj，跳过QKV打包
                if weight_name == ".q_proj":  # 如果是Q投影
                    match = re.search(r"layers\.\d+", name)  # 查找层号
                    if match:  # 如果匹配到
                        layer_id = int(match.group(0).split(".")[-1])  # 获取层ID
                        if cla_factor > 1 and layer_id % cla_factor != 0:  # 如果是CLA交叉层
                            continue  # 跳过QKV打包
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重

                is_found = True  # 设置找到标志
                break  # 跳出循环
            if is_found:  # 如果找到了匹配
                continue  # 跳到下一个权重

            for param_name, weight_name, den, split_param, func in split_params_mapping:  # 遍历拆分参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过

                assert loaded_weight.shape[0] % den == 0  # 断言权重可被整除
                units = loaded_weight.shape[0] // den  # 计算单位数

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                offset = 0  # 偏移量
                for shard_id, num in split_param:  # 遍历各分片
                    new_offset = offset + num * units  # 计算新偏移量
                    if func:  # 如果有拆分函数
                        weight_loader(  # 使用拆分函数加载
                            param, func(loaded_weight)[offset:new_offset], shard_id
                        )
                    else:  # 否则直接加载
                        weight_loader(param, loaded_weight[offset:new_offset], shard_id)
                    offset = new_offset  # 更新偏移量

                break  # 跳出循环
            else:  # 如果没有匹配拆分参数映射
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不在参数名中
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    # Skip layers on other devices.  # 跳过其他设备上的层
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载专家权重
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break  # 跳出循环
                else:  # 如果没有匹配专家参数映射
                    # Remapping the name of FP8 kv-scale.  # 重映射FP8 KV缩放的名称
                    name = maybe_remap_kv_scale_name(name, params_dict)  # 可能重映射名称
                    if name is None:  # 如果名称为None
                        continue  # 跳过

                    if "mlp.gate.wg." in name:  # 如果是MoE门控权重
                        name = name.replace("wg.", "")  # 移除wg.前缀

                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    # 如果调用此函数，应该始终初始化KV缓存缩放因子（否则抛出异常）。
    # 因此，处理的异常应确保将KV缓存缩放因子保留在已知良好（虚拟）状态
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:  # 加载KV缓存缩放因子
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        tp_rank = get_tensor_model_parallel_rank()  # 获取张量并行rank
        for layer_idx, scaling_factor in kv_cache_scales_loader(  # 加载KV缓存缩放因子
            quantization_param_path,  # 量化参数路径
            tp_rank,  # 张量并行rank
            tp_size,  # 张量并行大小
            self.config.num_hidden_layers,  # 隐藏层数量
            self.config.__class__.model_type,  # 模型类型
        ):
            if not isinstance(self.model.layers[layer_idx], nn.Identity):  # 如果层不是Identity
                layer_self_attn = self.model.layers[layer_idx].self_attn  # 获取自注意力层

            if is_hip():  # 如果是HIP（AMD GPU）
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                # 我们假设的缩放因子约定是：
                # quantized_value * scaling_factor ≈ true_value
                # 这与设置scaling_factor = tensor_amax / FPtype_max的实践一致
                scaling_factor *= 2  # HIP平台缩放因子乘2
            if hasattr(layer_self_attn, "kv_scale"):  # 如果有KV缩放属性
                layer_self_attn.attn._kv_scale = scaling_factor  # 设置KV缩放值
            else:  # 否则
                raise RuntimeError(  # 抛出运行时错误
                    "Self attention has no KV cache scaling " "factor attribute!"  # 自注意力没有KV缓存缩放因子属性
                )


class HunYuanDenseV1ForCausalLM(HunYuanMoEV1ForCausalLM):  # 混元Dense V1因果语言模型（继承MoE V1）
    pass  # 直接继承，无额外实现


EntryClass = [HunYuanMoEV1ForCausalLM, HunYuanDenseV1ForCausalLM]  # 入口类列表，用于模型注册
