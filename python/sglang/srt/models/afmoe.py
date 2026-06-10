# AfMoE模型推理实现模块
# 实现仅推理的AfMoE（注意力门控混合专家）模型，兼容HuggingFace权重格式，
# 支持Sigmoid门控、Q/K RMSNorm归一化、双层归一化、滑动窗口注意力和muP缩放

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
"""Inference-only AfMoE model compatible with HuggingFace weights.
"""仅推理的AfMoE模型，兼容HuggingFace权重。

AfMoE is a Mixture-of-Experts model with:
AfMoE是一个混合专家模型，具有以下特点：
- Gated attention with sigmoid gating
- 使用Sigmoid门控的门控注意力
- Q/K normalization with RMSNorm
- 使用RMSNorm的Q/K归一化
- Dual normalization (pre/post for both attention and MLP)
- 双层归一化（注意力层和MLP层的前/后归一化）
- Sliding window attention for local layers
- 局部层的滑动窗口注意力
- muP (maximal update parameterization) scaling support
- muP（最大更新参数化）缩放支持
"""

from __future__ import annotations  # 启用延迟类型注解求值

import functools  # 导入偏函数工具
from typing import Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入PyTorch函数式接口
from torch import nn  # 从torch导入神经网络模块
from transformers import PretrainedConfig  # 从transformers导入预训练配置类

from sglang.srt.distributed import (  # 导入分布式通信函数
    get_tensor_model_parallel_rank,  # 获取张量模型并行秩
    get_tensor_model_parallel_world_size,  # 获取张量模型并行世界大小
    tensor_model_parallel_all_reduce,  # 张量模型并行全归约
)
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS层归一化
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入逻辑处理器
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # 导入MoE运行器配置
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe  # 导入融合MoE
from sglang.srt.layers.moe.topk import TopK  # 导入Top-K选择
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix, is_npu  # 导入工具函数

_is_npu = is_npu()  # 判断是否为NPU设备

if _is_npu:  # 如果是NPU设备
    from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (  # 导入NPU融合MoE
        fused_moe_npu as fused_moe,  # 使用NPU版本的融合MoE
    )


def get_attention_sliding_window_size(config: PretrainedConfig) -> Optional[int]:  # 获取注意力滑动窗口大小
    sliding_window = getattr(config, "sliding_window", None)  # 获取滑动窗口配置
    if sliding_window is None:  # 如果未配置
        return None  # 返回None
    if sliding_window <= 0:  # 如果滑动窗口值无效
        return None  # 返回None
    # Align with other local attention implementations (see gpt_oss).
    # 与其他局部注意力实现对齐（参见gpt_oss）。
    return sliding_window - 1  # 返回滑动窗口大小减1


class AfmoeMLP(nn.Module):  # AfMoE的MLP（多层感知机）模块

    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # gate和up投影合并的列并行线性层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 两个中间层大小（gate和up）
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # down投影的行并行线性层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 无偏置
            reduce_results=reduce_results,  # 是否归约结果
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
        )
        if hidden_act != "silu":  # 如果激活函数不是SiLU
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."  # 不支持的激活函数
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        gate_up, _ = self.gate_up_proj(x)  # gate和up投影
        x = self.act_fn(gate_up)  # 应用激活函数
        x, _ = self.down_proj(x)  # down投影
        return x  # 返回输出


class AfmoeMoE(nn.Module):  # AfMoE的MoE（混合专家）模块

    @staticmethod
    def _custom_routing_function(  # 自定义路由函数（支持Sigmoid和Softmax两种评分方式）
        hidden_states: torch.Tensor,  # 隐藏状态
        gating_output: torch.Tensor,  # 门控输出
        topk: int,  # Top-K值
        renormalize: bool,  # 是否重归一化
        *,
        score_func: str,  # 评分函数类型
        expert_bias: Optional[torch.Tensor],  # 专家偏置
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = gating_output.to(torch.float32)  # 将门控输出转为float32
        if score_func == "sigmoid":  # 如果使用Sigmoid评分
            scores = torch.sigmoid(logits)  # 计算Sigmoid分数
            if expert_bias is not None:  # 如果有专家偏置
                bias = expert_bias.to(scores.device, dtype=scores.dtype)  # 转换偏置到相同设备和数据类型
                scores_for_choice = scores + bias  # 带偏置的分数用于选择
                topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)[1]  # 选择Top-K专家ID
                topk_weights = scores.gather(dim=-1, index=topk_ids)  # 收集对应的原始分数
            else:  # 无专家偏置
                topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)  # 直接选择Top-K
        else:  # 使用Softmax评分
            if expert_bias is not None:  # 如果有专家偏置
                logits = logits + expert_bias.to(logits.device, dtype=logits.dtype)  # 添加偏置到logits
            probs = F.softmax(logits, dim=-1)  # 计算Softmax概率
            topk_weights, topk_ids = torch.topk(probs, k=topk, dim=-1)  # 选择Top-K

        if renormalize:  # 如果需要重归一化
            denom = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)  # 计算分母（加小值防止除零）
            topk_weights = topk_weights / denom  # 归一化权重

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)  # 返回Top-K权重和ID

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.rank = get_tensor_model_parallel_rank()  # 获取当前TP秩
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP世界大小

        self.n_routed_experts = getattr(config, "num_experts", None)  # 获取路由专家数量
        if self.n_routed_experts is None:  # 如果未定义
            raise ValueError("AfmoeConfig must define `num_experts`.")  # 抛出值错误
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数
        if self.tp_size > self.n_routed_experts:  # 如果TP大小大于专家数量
            raise ValueError(  # 抛出值错误
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."  # TP大小不能大于专家数量
            )

        self.score_func = getattr(config, "score_func", "softmax")  # 评分函数，默认为softmax
        self.route_norm = getattr(config, "route_norm", True)  # 路由归一化，默认为True
        self.route_scale = float(getattr(config, "route_scale", 1.0))  # 路由缩放因子
        self.n_group = getattr(config, "n_group", 1)  # 分组数量
        self.topk_group = getattr(config, "topk_group", 1)  # 每组选择的专家数
        self.use_grouped_topk = self.n_group is not None and self.n_group > 1  # 是否使用分组Top-K
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)  # 共享专家数量

        self.gate = ReplicatedLinear(  # 门控线性层
            config.hidden_size,  # 输入大小
            self.n_routed_experts,  # 输出大小（专家数量）
            bias=False,  # 无偏置
            quant_config=None,  # 不量化门控层
            prefix=add_prefix("gate", prefix),  # 参数前缀
        )

        self.expert_bias = nn.Parameter(  # 专家偏置参数
            torch.zeros(self.n_routed_experts, dtype=torch.float32),  # 初始化为零
            requires_grad=False,  # 不需要梯度
        )

        self.experts = nn.ModuleList(  # 专家模块列表
            [
                AfmoeMLP(  # 每个专家是一个MLP
                    hidden_size=config.hidden_size,  # 隐藏层大小
                    intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
                    hidden_act=config.hidden_act,  # 激活函数
                    quant_config=quant_config,  # 量化配置
                    reduce_results=False,  # 不归约结果（MoE归约在外部完成）
                    prefix=add_prefix(f"experts.{idx}", prefix),  # 参数前缀
                )
                for idx in range(self.n_routed_experts)  # 遍历所有专家
            ]
        )

        self.pack_params()  # 打包专家参数

        if self.num_shared_experts:  # 如果有共享专家
            intermediate_size = config.moe_intermediate_size * self.num_shared_experts  # 共享专家总中间层大小
            self.shared_experts = AfmoeMLP(  # 共享专家MLP
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                reduce_results=False,  # 不归约结果
                prefix=add_prefix("shared_experts", prefix),  # 参数前缀
            )
        else:  # 没有共享专家
            self.shared_experts = None  # 设为None

        custom_routing_fn = None  # 自定义路由函数
        correction_bias = None if not _is_npu else self.expert_bias  # NPU使用专家偏置作为校正偏置
        if self.use_grouped_topk:  # 如果使用分组Top-K
            correction_bias = self.expert_bias  # 使用专家偏置作为校正偏置
        elif self.score_func == "sigmoid":  # 如果使用Sigmoid评分
            custom_routing_fn = functools.partial(  # 使用偏函数创建自定义路由函数
                AfmoeMoE._custom_routing_function,  # 自定义路由函数
                score_func=self.score_func,  # 评分函数
                expert_bias=self.expert_bias,  # 专家偏置
            )

        renormalize = (  # 是否重归一化
            self.route_norm if self.score_func == "sigmoid" and not _is_npu else False  # Sigmoid评分且非NPU时使用route_norm
        )
        self.topk = TopK(  # Top-K选择模块
            top_k=self.top_k,  # Top-K值
            renormalize=renormalize,  # 是否重归一化
            use_grouped_topk=self.use_grouped_topk,  # 是否使用分组Top-K
            num_expert_group=self.n_group if self.use_grouped_topk else None,  # 专家分组数
            topk_group=self.topk_group if self.use_grouped_topk else None,  # 每组Top-K值
            custom_routing_function=custom_routing_fn,  # 自定义路由函数
            correction_bias=correction_bias,  # 校正偏置
            routed_scaling_factor=self.route_scale,  # 路由缩放因子
            **({"scoring_func": self.score_func} if _is_npu else {}),  # NPU额外参数
        )

    def pack_params(self) -> None:  # 打包专家参数（将所有专家权重扁平化为连续张量）
        w1: list[torch.Tensor] = []  # gate_up投影权重列表
        w2: list[torch.Tensor] = []  # down投影权重列表
        for expert in self.experts:  # 遍历所有专家
            w1.append(expert.gate_up_proj.weight)  # 添加gate_up权重
            w2.append(expert.down_proj.weight)  # 添加down权重
        self.w1 = torch._utils._flatten_dense_tensors(w1)  # 扁平化w1权重
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)  # 反扁平化w1权重
        for data, param in zip(w1s, w1):  # 同步扁平化后的数据
            param.data = data  # 更新参数数据
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)  # 重塑为[num_experts, *weight_shape]

        self.w2 = torch._utils._flatten_dense_tensors(w2)  # 扁平化w2权重
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)  # 反扁平化w2权重
        for data, param in zip(w2s, w2):  # 同步扁平化后的数据
            param.data = data  # 更新参数数据
        self.w2 = self.w2.view(len(w2), *w2s[0].shape)  # 重塑为[num_experts, *weight_shape]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播
        num_tokens, hidden_dim = hidden_states.shape  # 获取token数量和隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑隐藏状态形状

        shared_output = None  # 共享专家输出
        if self.shared_experts is not None:  # 如果有共享专家
            shared_output = self.shared_experts(hidden_states)  # 计算共享专家输出

        router_logits, _ = self.gate(hidden_states)  # 计算路由器logits
        topk_output = self.topk(hidden_states, router_logits)  # 计算Top-K选择结果
        final_hidden_states = fused_moe(  # 融合MoE计算
            hidden_states,  # 隐藏状态
            w1=self.w1,  # gate_up权重
            w2=self.w2,  # down权重
            topk_output=topk_output,  # Top-K选择结果
            moe_runner_config=MoeRunnerConfig(  # MoE运行器配置
                inplace=True,  # 原地操作
                routed_scaling_factor=self.route_scale,  # 路由缩放因子
            ),
        )

        if shared_output is not None:  # 如果有共享专家输出
            final_hidden_states = final_hidden_states + shared_output  # 叠加共享专家输出

        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 张量模型并行全归约
        return final_hidden_states.view(num_tokens, hidden_dim)  # 返回重塑后的输出


class AfmoeAttention(nn.Module):  # AfMoE的注意力模块

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP世界大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个TP秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于TP大小
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # KV头数小于TP大小
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个TP秩的KV头数

        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 缩放因子

        rope_theta = config.rope_parameters["rope_theta"]  # RoPE theta参数
        rope_scaling = config.rope_parameters  # RoPE缩放参数
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)  # 部分旋转因子
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)  # 旋转维度
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入数

        layer_types = getattr(config, "layer_types", None)  # 获取层类型列表
        self.is_local_attention = (  # 是否为局部注意力
            layer_types is not None and layer_types[layer_id] == "sliding_attention"  # 层类型为滑动注意力
        )
        sliding_window = (  # 滑动窗口大小
            get_attention_sliding_window_size(config) if self.is_local_attention else -1  # 局部注意力时获取窗口大小
        )

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            hidden_size,  # 输入大小
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )
        self.gate_proj = ColumnParallelLinear(  # 门控投影（用于Sigmoid门控注意力）
            hidden_size,  # 输入大小
            self.total_num_heads * self.head_dim,  # 输出大小
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_proj", prefix),  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.rotary_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基础频率
            rope_scaling=rope_scaling,  # 缩放配置
            is_neox_style=True,  # 使用Neox风格
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            sliding_window_size=sliding_window,  # 滑动窗口大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

        eps = getattr(config, "rms_norm_eps", 1e-5)  # RMSNorm epsilon
        self.q_norm = RMSNorm(self.head_dim, eps=eps)  # Q归一化
        self.k_norm = RMSNorm(self.head_dim, eps=eps)  # K归一化
        self.sliding_window = sliding_window  # 保存滑动窗口大小

    def _apply_qk_norm(  # 应用Q/K归一化
        self, q: torch.Tensor, k: torch.Tensor  # Q和K张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_heads = self.q_norm(q.reshape(-1, self.head_dim))  # 归一化Q（先重塑为每头一维）
        k_heads = self.k_norm(k.reshape(-1, self.head_dim))  # 归一化K（先重塑为每头一维）
        q = q_heads.view(q.shape)  # 恢复Q的原始形状
        k = k_heads.view(k.shape)  # 恢复K的原始形状
        return q, k  # 返回归一化后的Q和K

    def forward(  # 前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分Q、K、V
        q, k = self._apply_qk_norm(q, k)  # 应用Q/K归一化

        if self.is_local_attention:  # 如果是局部注意力
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力输出

        gate_vals, _ = self.gate_proj(hidden_states)  # 计算门控值
        attn_output = attn_output * torch.sigmoid(gate_vals)  # 应用Sigmoid门控
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出


class AfmoeDecoderLayer(nn.Module):  # AfMoE解码器层

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        self.layer_id = layer_id  # 保存层ID

        self.self_attn = AfmoeAttention(  # 自注意力模块
            config=config,  # 配置
            hidden_size=config.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )

        use_moe = False  # 是否使用MoE
        if hasattr(config, "num_dense_layers"):  # 如果配置了密集层数
            use_moe = layer_id >= config.num_dense_layers  # 超过密集层数后使用MoE
        elif (  # 否则通过其他配置判断
            getattr(config, "num_experts", None) is not None  # 有专家数配置
            and hasattr(config, "first_k_dense_replace")  # 有首个密集替换配置
            and hasattr(config, "moe_layer_freq")  # 有MoE层频率配置
        ):
            base = config.first_k_dense_replace  # 前base层为密集层
            freq = config.moe_layer_freq  # MoE层频率
            use_moe = layer_id >= base and (layer_id - base) % freq == 0  # 根据频率决定是否使用MoE

        if use_moe:  # 如果使用MoE
            self.mlp = AfmoeMoE(  # 使用MoE MLP
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 参数前缀
            )
        else:  # 使用密集MLP
            self.mlp = AfmoeMLP(  # 使用密集MLP
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=config.intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 参数前缀
            )

        eps = getattr(config, "rms_norm_eps", 1e-5)  # RMSNorm epsilon
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)  # 注意力后层归一化
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=eps)  # MLP前层归一化
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=eps)  # MLP后层归一化

    def forward(  # 前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        attn_residual = hidden_states  # 保存注意力残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)  # 自注意力
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化
        hidden_states = attn_residual + hidden_states  # 残差连接

        mlp_residual = hidden_states  # 保存MLP残差
        hidden_states = self.pre_mlp_layernorm(hidden_states)  # MLP前层归一化
        hidden_states = self.mlp(hidden_states)  # MLP
        hidden_states = self.post_mlp_layernorm(hidden_states)  # MLP后层归一化
        hidden_states = mlp_residual + hidden_states  # 残差连接

        return hidden_states  # 返回输出


class AfmoeModel(nn.Module):  # AfMoE模型主体

    fall_back_to_pt_during_load = False  # 加载时不回退到PyTorch

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 嵌入维度
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                AfmoeDecoderLayer(  # 解码器层
                    config,  # 配置
                    layer_id,  # 层ID
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{layer_id}", prefix),  # 参数前缀
                )
                for layer_id in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化

    def forward(  # 前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态
        else:  # 如果提供了输入嵌入
            hidden_states = input_embeds  # 直接使用输入嵌入

        if getattr(self.config, "mup_enabled", False):  # 如果启用了muP缩放
            hidden_states = hidden_states * (self.config.hidden_size**0.5)  # 应用muP缩放

        for layer in self.layers:  # 遍历所有解码器层
            hidden_states = layer(positions, hidden_states, forward_batch)  # 通过解码器层
        hidden_states = self.norm(hidden_states)  # 最终层归一化
        return hidden_states  # 返回隐藏状态

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.embed_tokens  # 返回词嵌入层


class AfmoeForCausalLM(nn.Module):  # AfMoE因果语言模型

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = AfmoeModel(  # AfMoE模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 配置和量化配置
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 参数前缀
        )
        self.logits_processor = LogitsProcessor(config)  # 逻辑处理器

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.model.embed_tokens  # 返回模型主体中的词嵌入层

    def forward(  # 前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 获取模型主体输出
        return self.logits_processor(  # 通过逻辑处理器计算最终输出
            input_ids, hidden_states, self.lm_head, forward_batch  # 输入ID、隐藏状态、LM头、批次信息
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:  # 获取注意力滑动窗口大小
        return get_attention_sliding_window_size(self.config)  # 委托给全局函数

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:  # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, weight_name, shard_id)  # （参数名，权重名，分片ID）
            ("qkv_proj", "q_proj", "q"),  # Q投影映射
            ("qkv_proj", "k_proj", "k"),  # K投影映射
            ("qkv_proj", "v_proj", "v"),  # V投影映射
            ("gate_up_proj", "gate_proj", 0),  # gate投影映射
            ("gate_up_proj", "up_proj", 1),  # up投影映射
        ]

        params_dict = dict(self.named_parameters())  # 获取模型参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            # Skip rotary embedding inverse frequencies
            # 跳过旋转位置编码逆频率
            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码逆频率
                continue  # 跳过

            # Remap router gate weights: HF uses .mlp.router.gate., SGLang uses .mlp.gate.
            # 重映射路由器门控权重：HF使用.mlp.router.gate.，SGLang使用.mlp.gate.
            if ".mlp.router.gate." in name:  # 如果包含HF路由器门控路径
                name = name.replace(".mlp.router.gate.", ".mlp.gate.")  # 替换为SGLang路径

            # Handle stacked params (qkv_proj, gate_up_proj)
            # 处理堆叠参数（qkv_proj、gate_up_proj）
            handled = False  # 是否已处理标志
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在当前名称中
                    continue  # 跳过
                # Skip gate_proj/up_proj stacking for self_attn (attention uses separate gate_proj)
                # 跳过self_attn中gate_proj/up_proj的堆叠（注意力使用独立的gate_proj）
                if ".self_attn." in name and weight_name in {"gate_proj", "up_proj"}:  # 注意力层中跳过gate/up
                    continue  # 跳过

                new_name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip if parameter doesn't exist (e.g., bias for layers without bias)
                # 如果参数不存在则跳过（例如无偏置层的偏置）
                if new_name not in params_dict:  # 如果参数不存在
                    handled = True  # 标记为已处理
                    break  # 跳出循环

                param = params_dict[new_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                handled = True  # 标记为已处理
                break  # 跳出循环

            if handled:  # 如果已处理
                continue  # 跳过后续处理

            # Load remaining weights directly
            # 直接加载剩余权重
            if name in params_dict:  # 如果权重在参数字典中
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = AfmoeForCausalLM  # 入口类为AfmoeForCausalLM
