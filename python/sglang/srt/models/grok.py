# Grok1模型推理实现 - 基于大规模稀疏专家混合(MoE)架构的Grok1模型，仅用于推理
# Copyright 2023-2024 SGLang Team  # 版权所有 2023-2024 SGLang团队
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
# ==============================================================================  # 分隔线

import functools  # 函数工具模块
import logging  # 日志模块
import math  # 数学模块
from typing import Iterable, Optional, Tuple  # 类型提示导入

import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch神经网络函数模块
from torch import nn  # 神经网络模块
from transformers import PretrainedConfig  # 预训练配置基类

from sglang.srt.distributed import (  # 分布式工具导入
    get_tensor_model_parallel_rank,  # 获取张量并行rank
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.layers.activation import GeluAndMul  # GELU激活函数与乘法
from sglang.srt.layers.elementwise import (  # 逐元素操作层
    fused_dual_residual_rmsnorm,  # 融合双残差RMS归一化
    fused_rmsnorm,  # 融合RMS归一化
    gelu_and_mul_triton,  # GELU与乘法Triton内核
)
from sglang.srt.layers.layernorm import RMSNorm  # RMS归一化层
from sglang.srt.layers.linear import (  # 线性层导入
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 融合MoE Triton内核
from sglang.srt.layers.moe.router import fused_moe_router_shim  # 融合MoE路由器垫片
from sglang.srt.layers.moe.topk import TopK  # Top-K选择模块
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力机制
from sglang.srt.layers.rotary_embedding import (  # 旋转位置编码导入
    RotaryEmbedding,  # 旋转位置编码基类
    _yarn_find_correction_range,  # YaRN查找校正范围
    _yarn_get_mscale,  # YaRN获取幅度缩放
    get_rope,  # 获取旋转位置编码
)
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 获取是否处于CUDA图捕获模式
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息
from sglang.srt.model_loader.loader import DefaultModelLoader  # 默认模型加载器
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.utils import add_prefix, is_npu  # 工具函数导入

_is_npu = is_npu()  # 是否在NPU上运行

logger = logging.getLogger(__name__)  # 获取当前模块日志记录器


class Grok1MLP(nn.Module):  # Grok1 MLP模块
    def __init__(  # 初始化MLP
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        reduce_results=True,  # 是否归约结果
        use_presharded_weights: bool = False,  # 是否使用预分片权重
        split_gate_up: bool = False,  # 是否拆分gate和up
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.gate_up_proj = MergedColumnParallelLinear(  # 创建gate和up合并投影层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小列表（gate和up各一个）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
            use_presharded_weights=use_presharded_weights,  # 预分片权重
        )
        self.down_proj = RowParallelLinear(  # 创建down投影层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
            reduce_results=reduce_results,  # 是否归约结果
            use_presharded_weights=use_presharded_weights,  # 预分片权重
        )
        self.act_fn = GeluAndMul(approximate="tanh")  # 创建GELU激活与乘法函数（tanh近似）
        self.layer_id = layer_id  # 保存层ID

    def forward(self, x):  # MLP前向传播
        gate_up, _ = self.gate_up_proj(x)  # 计算gate和up投影
        x, _ = gelu_and_mul_triton(gate_up)  # 应用GELU激活与乘法
        x, _ = self.down_proj(x)  # 计算down投影
        return x  # 返回结果


class Grok1MoE(nn.Module):  # Grok1稀疏专家混合模块
    def __init__(  # 初始化MoE
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        num_experts: int,  # 专家数量
        top_k: int,  # 每个token选择的top-k专家数
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        tp_size: Optional[int] = None,  # 张量并行大小
        reduce_results: bool = True,  # 是否归约结果
        use_presharded_weights: bool = False,  # 是否使用预分片权重
        inplace: bool = True,  # 是否原地操作
        no_combine: bool = False,  # 是否不组合
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小

        self.gate = ReplicatedLinear(  # 创建门控线性层（路由器）
            hidden_size,  # 输入大小
            num_experts,  # 输出大小（专家数量）
            bias=False,  # 不使用偏置
            params_dtype=torch.float32,  # 使用float32精度
            quant_config=None,  # 门控不使用量化
        )

        self.router_logit_softcapping = 30.0  # 路由器logit软上限
        custom_routing_function = functools.partial(  # 创建自定义路由函数
            fused_moe_router_shim, self.router_logit_softcapping  # 使用软上限的路由器垫片
        )

        self.topk = TopK(  # 创建Top-K选择模块
            top_k=top_k,  # 选择top-k个专家
            renormalize=False,  # 不对权重重新归一化
            layer_id=layer_id,  # 层ID
            custom_routing_function=None if _is_npu else custom_routing_function,  # NPU不使用自定义路由
        )

        self.experts = FusedMoE(  # 创建融合MoE专家层
            num_experts=num_experts,  # 专家数量
            top_k=top_k,  # top-k值
            layer_id=layer_id,  # 层ID
            hidden_size=hidden_size,  # 隐藏层大小
            intermediate_size=intermediate_size,  # 中间层大小
            params_dtype=params_dtype,  # 参数数据类型
            quant_config=quant_config,  # 量化配置
            activation="gelu",  # 使用GELU激活函数
            reduce_results=reduce_results,  # 是否归约结果
            use_presharded_weights=use_presharded_weights,  # 预分片权重
            inplace=inplace,  # 原地操作
            no_combine=no_combine,  # 不组合
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # MoE前向传播
        if not _is_npu:  # 如果不是NPU
            topk_output = self.topk(hidden_states, self.gate.weight)  # 直接使用门控权重进行top-k选择
            return self.experts(hidden_states, topk_output)  # 执行专家计算
        else:  # 如果是NPU
            orig_shape = hidden_states.shape  # 保存原始形状
            hidden_states = hidden_states.view(-1, self.hidden_size)  # 重塑为2D

            router_logits, _ = self.gate(hidden_states)  # 计算路由器logits
            router_logits = self.router_logit_softcapping * F.tanh(  # 应用软上限（tanh截断）
                router_logits / self.router_logit_softcapping
            )
            topk_output = self.topk(hidden_states, router_logits)  # 获取top-k选择结果

            final_hidden_states = self.experts(hidden_states, topk_output)  # 执行专家计算
            return final_hidden_states.view(orig_shape)  # 恢复原始形状并返回


def _yarn_linear_ramp_mask(  # YaRN线性斜坡掩码函数
    low: float, high: float, dim: int, dtype: torch.dtype  # 低值、高值、维度、数据类型
) -> torch.Tensor:
    if low == high:  # 如果低值等于高值
        low -= 0.001  # Prevent singularity  # 防止奇异性

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)  # 线性函数
    ramp_func = torch.clamp(linear_func, 0, 1)  # 截断到[0, 1]
    return ramp_func  # 返回斜坡函数


def get_rope_scaling(config):  # 获取RoPE缩放配置
    rope_type = getattr(config, "rope_type", None)  # 获取RoPE类型
    if rope_type:  # 如果有RoPE类型
        original_max_position_embeddings = getattr(  # 获取原始最大位置嵌入数
            config, "original_max_position_embeddings", None
        )
        scaling_factor = getattr(config, "scaling_factor", None)  # 获取缩放因子
        extrapolation_factor = getattr(config, "extrapolation_factor", 1.0)  # 获取外推因子
        attn_factor = getattr(config, "attn_factor", 1.0)  # 获取注意力因子
        beta_fast = getattr(config, "beta_fast", 32)  # 获取快beta
        beta_slow = getattr(config, "beta_slow", 1)  # 获取慢beta
        rope_scaling = {  # 构建RoPE缩放配置字典
            "extra_method": rope_type,  # 外推方法
            "max_position_embeddings": original_max_position_embeddings,  # 最大位置嵌入数
            "scaling_factor": scaling_factor,  # 缩放因子
            "extrapolation_factor": extrapolation_factor,  # 外推因子
            "attn_factor": attn_factor,  # 注意力因子
            "beta_fast": beta_fast,  # 快beta
            "beta_slow": beta_slow,  # 慢beta
            "dtype": torch.bfloat16,  # 数据类型
        }
        return rope_scaling  # 返回RoPE缩放配置
    else:  # 否则没有RoPE缩放
        return None  # 返回None


class ScalingRotaryEmbedding(RotaryEmbedding):  # 缩放旋转位置编码，类似YaRN方法
    """Scale the RotaryEmbedding in a way similar to YaRN method. https://arxiv.org/pdf/2309.00071."""  # 以类似YaRN方法的方式缩放旋转位置编码

    def __init__(  # 初始化缩放旋转位置编码
        self,
        head_size: int,  # 头大小
        rotary_dim: int,  # 旋转维度
        max_position_embeddings: int,  # 最大位置嵌入数
        base: int,  # 基数
        is_neox_style: bool,  # 是否为Neox风格
        scaling_factor: float,  # 缩放因子
        dtype: torch.dtype,  # 数据类型
        *,  # 以下为关键字参数
        extra_method: str = "yarn_log",  # 外推方法（默认yarn_log）
        extrapolation_factor: float = 1,  # 外推因子
        attn_factor: float = 1,  # 注意力因子
        beta_fast: int = 32,  # 快beta
        beta_slow: int = 1,  # 慢beta
    ) -> None:
        self.scaling_factor = scaling_factor  # 保存缩放因子
        self.extra_method = extra_method  # 保存外推方法
        self.extrapolation_factor = extrapolation_factor  # 保存外推因子
        self.attn_factor = attn_factor  # 保存注意力因子
        self.beta_fast = beta_fast  # 保存快beta
        self.beta_slow = beta_slow  # 保存慢beta
        if _is_npu:  # 如果是NPU
            dtype = torch.float32  # 使用float32
        # Get n-d magnitude scaling corrected for interpolation  # 获取针对插值校正的n维幅度缩放
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)  # 计算幅度缩放
        super().__init__(  # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:  # 计算逆频率
        pos_freqs = self.base ** (  # 计算位置频率
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs  # 外推逆频率
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)  # 插值逆频率

        low, high = _yarn_find_correction_range(  # 查找校正范围
            self.beta_fast,  # 快beta
            self.beta_slow,  # 慢beta
            self.rotary_dim,  # 旋转维度
            self.base,  # 基数
            self.max_position_embeddings,  # 最大位置嵌入数
        )
        # Get n-d rotational scaling corrected for extrapolation  # 获取针对外推校正的n维旋转缩放
        inv_freq_mask = (  # 逆频率掩码
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor  # 乘以外推因子
        if self.extra_method in ["original"]:  # 如果使用原始方法
            inv_freq = inv_freq_extrapolation  # 使用外推逆频率
        elif self.extra_method in ["yarn", "yarn_linear"]:  # 如果使用YaRN或线性YaRN方法
            inv_freq = (  # 混合插值和外推逆频率
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
        elif self.extra_method == "yarn_log":  # 如果使用对数YaRN方法
            inv_freq = torch.exp(  # 对数空间混合
                torch.log(inv_freq_extrapolation) * inv_freq_mask
                + torch.log(inv_freq_interpolation) * (1.0 - inv_freq_mask)
            )
        elif self.extra_method == "theta_scale":  # 如果使用theta缩放方法
            exponents = torch.arange(0, self.rotary_dim, 2, dtype=torch.float)  # 指数
            theta_scale_exponent = self.base ** (  # theta缩放指数
                math.log(
                    self.max_position_embeddings * self.scaling_factor / (2 * math.pi)
                )
                / math.log(self.max_position_embeddings / (2 * math.pi))
            )
            inv_freq = torch.tensor(  # 计算逆频率
                1.0 / (theta_scale_exponent ** (exponents / self.rotary_dim)),
                dtype=torch.float32,
            )
        else:  # 未知方法
            raise ValueError(f"Unknown extrapolation method: {self.extra_method}")  # 抛出错误
        return inv_freq  # 返回逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算cos/sin缓存
        inv_freq = self._compute_inv_freq(self.scaling_factor)  # 计算逆频率
        t = torch.arange(  # 生成位置索引
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算频率矩阵
        # cos = freqs.cos() * self.mscale
        # sin = freqs.sin() * self.mscale  # 注释掉的幅度缩放cos/sin
        cos = freqs.cos()  # 计算cos
        sin = freqs.sin()  # 计算sin
        cache = torch.cat((cos, sin), dim=-1)  # 拼接cos和sin
        return cache  # 返回缓存


class Grok1Attention(nn.Module):  # Grok1注意力模块
    def __init__(  # 初始化注意力
        self,
        config: PretrainedConfig,  # 模型配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数量
        num_kv_heads: int,  # KV头数量
        layer_id: int = 0,  # 层ID
        max_position: int = 4096 * 32,  # 最大位置编码长度
        rope_theta: float = 10000,  # RoPE基数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
        load_presharded_attn: bool = False,  # 是否加载预分片注意力
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.layer_id = layer_id  # 保存层ID
        self.hidden_size = hidden_size  # 保存隐藏层大小
        attn_tp_rank = get_tensor_model_parallel_rank()  # 获取张量并行rank
        attn_tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % attn_tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 每个rank的注意力头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= attn_tp_size:  # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.  # KV头数大于TP大小，因此在多个张量并行GPU上划分KV头
            assert self.total_num_kv_heads % attn_tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.  # KV头数小于TP大小，因此在多个张量并行GPU上复制KV头
            assert attn_tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 每个rank的KV头数
        self.head_dim = getattr(config, "head_dim", 128)  # 头维度（默认128）
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.rope_theta = rope_theta  # 保存RoPE基数
        rope_scaling = get_rope_scaling(config)  # 获取RoPE缩放配置
        self.load_presharded_attn = load_presharded_attn  # 保存预分片注意力标志
        self.alt_stream = alt_stream or torch.cuda.Stream()  # 备用CUDA流

        self.qkv_proj = QKVParallelLinear(  # 创建QKV并行投影层
            hidden_size,  # 输入大小
            self.head_dim,  # 每个头的大小
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # 张量并行rank
            tp_size=attn_tp_size,  # 张量并行大小
            load_presharded_attn=self.load_presharded_attn,  # 预分片注意力
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 创建输出投影层
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            reduce_results=reduce_results,  # 是否归约结果
            tp_rank=attn_tp_rank,  # 张量并行rank
            tp_size=attn_tp_size,  # 张量并行大小
            use_presharded_weights=self.load_presharded_attn,  # 预分片权重
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )
        self.rotary_emb = get_rope(  # 创建旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position,  # 最大位置
            base=int(self.rope_theta),  # RoPE基数
            is_neox_style=True,  # 使用Neox风格
        )

        self.rope_rotate_half_dims = getattr(config, "rope_rotate_half_dims", False)  # RoPE是否旋转半维度

        if rope_scaling is not None:  # 如果有RoPE缩放
            self.rotary_emb = ScalingRotaryEmbedding(  # 创建缩放旋转位置编码
                self.head_dim,  # 头维度
                rotary_dim=(  # 旋转维度
                    self.head_dim  # 不旋转半维度时使用完整头维度
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2  # 否则使用半维度
                ),
                base=int(self.rope_theta),  # 基数
                is_neox_style=True,  # Neox风格
                **rope_scaling,  # 解包RoPE缩放配置
            )
            pos_encoding_mode = "NONE"  # 位置编码模式
        else:  # 否则没有RoPE缩放
            self.rotary_emb = get_rope(  # 创建标准旋转位置编码
                self.head_dim,  # 头维度
                rotary_dim=(  # 旋转维度
                    self.head_dim  # 不旋转半维度时使用完整头维度
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2  # 否则使用半维度
                ),
                max_position=max_position,  # 最大位置
                base=int(self.rope_theta),  # 基数
                is_neox_style=True,  # Neox风格
                dtype=torch.float32 if _is_npu else None,  # NPU使用float32
            )
            pos_encoding_mode = "NONE"  # 位置编码模式

        logit_cap = max(getattr(config, "attn_logit_softcapping", 30.0), 0.0)  # 注意力logit软上限
        logit_capping_method = getattr(config, "attn_logit_softcapping_method", "tanh")  # logit上限方法

        self.attn = RadixAttention(  # 创建基数注意力层
            self.num_heads,  # 注意力头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            logit_cap=logit_cap,  # logit上限
            quant_config=quant_config,  # 量化配置
            pos_encoding_mode=pos_encoding_mode,  # 位置编码模式
            logit_capping_method=logit_capping_method,  # logit上限方法
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )
        self.attn.xai_temperature_len = getattr(self.config, "attn_temperature_len", -1)  # 设置注意力温度长度

    def forward(  # 注意力前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV
        if not _is_npu:  # 如果不是NPU
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        else:  # 如果是NPU
            odtype = q.dtype  # 保存原始数据类型
            q, k = self.rotary_emb(positions, q.to(torch.float32), k.to(torch.float32))  # 在float32精度下应用RoPE
            q, k = q.to(odtype), k.to(odtype)  # 转回原始数据类型

        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算

        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出


class Grok1DecoderLayer(nn.Module):  # Grok1解码器层
    def __init__(  # 初始化解码器层
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        load_presharded_moe: bool = False,  # 是否加载预分片MoE
        load_presharded_attn: bool = False,  # 是否加载预分片注意力
        load_presharded_mlp: bool = False,  # 是否加载预分片MLP
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
        skip_moe: bool = False,  # 是否跳过MoE
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.num_experts = config.num_local_experts  # 专家数量
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.residual_moe = getattr(config, "residual_moe", False)  # 是否使用残差MoE
        self.layer_id = layer_id  # 保存层ID
        self.alt_stream = alt_stream or torch.cuda.Stream()  # 备用CUDA流

        rope_theta = getattr(config, "rope_theta", None)  # 获取RoPE基数
        if rope_theta is None:  # 如果没有直接配置
            rope_params = getattr(config, "rope_parameters", None)  # 获取RoPE参数
            rope_theta = rope_params["rope_theta"] if rope_params else 10000  # 从参数获取或使用默认值
        self.self_attn = Grok1Attention(  # 创建自注意力层
            config=config,  # 配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            max_position=(  # 最大位置编码
                config.context_len  # 优先使用context_len
                if hasattr(config, "context_len")  # 如果有context_len属性
                else config.max_position_embeddings  # 否则使用max_position_embeddings
            ),
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE基数
            quant_config=quant_config,  # 量化配置
            reduce_results=False,  # 注意力不归约结果
            alt_stream=self.alt_stream,  # 备用CUDA流
            load_presharded_attn=load_presharded_attn,  # 预分片注意力
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

        split_gate_up = not getattr(config, "merge_gate_up", True)  # 是否拆分gate和up
        if self.num_experts > 0:  # 如果有专家
            self.block_sparse_moe = Grok1MoE(  # 创建稀疏MoE层
                config=config,  # 配置
                layer_id=layer_id,  # 层ID
                num_experts=config.num_local_experts,  # 专家数量
                top_k=config.num_experts_per_tok,  # 每个token的top-k专家数
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=getattr(  # 中间层大小
                    config,
                    "moe_intermediate_size",  # 优先使用MoE中间层大小
                    getattr(config, "intermediate_size", None),  # 否则使用通用中间层大小
                ),
                quant_config=quant_config,  # 量化配置
                reduce_results=not self.residual_moe,  # 残差MoE时不归约结果
                use_presharded_weights=load_presharded_moe,  # 预分片权重
                inplace=False,  # not self.residual_moe,  # 不原地操作
                no_combine=False,  # self.residual_moe,  # just a suggestion to not combine topk  # 不跳过组合
                prefix=add_prefix("block_sparse_moe", prefix),  # 参数前缀
            )
            if self.residual_moe:  # 如果使用残差MoE
                self.mlp = Grok1MLP(  # 创建MLP（与MoE并行）
                    hidden_size=config.hidden_size,  # 隐藏层大小
                    intermediate_size=config.intermediate_size,  # 中间层大小
                    quant_config=quant_config,  # 量化配置
                    reduce_results=False,  # 不归约结果
                    use_presharded_weights=load_presharded_mlp,  # 预分片权重
                    layer_id=layer_id,  # 层ID
                    split_gate_up=split_gate_up,  # 是否拆分gate和up
                )
        else:  # 否则没有专家
            raise NotImplementedError()  # 抛出未实现错误

        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力前归一化
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力后归一化
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # MoE前归一化
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # MoE后归一化

        if self.num_experts > 0:  # 如果有专家
            if self.residual_moe:  # 如果使用残差MoE
                # NOTE: self.block_sparse_moe modifies the input in-place,
                # so we have to call it later. Be aware of any possible related errors.
                # 注意：self.block_sparse_moe会原地修改输入，所以必须后调用。请注意可能的相关错误。
                if get_tensor_model_parallel_world_size() > 1:  # 如果TP大小大于1
                    self.ffn = lambda x: tensor_model_parallel_all_reduce(  # FFN使用全归约包装
                        self.moe_with_rmoe(x)
                    )
                else:  # 否则单卡
                    self.ffn = self.moe_with_rmoe  # 直接使用残差MoE函数
            else:  # 否则不使用残差MoE
                self.ffn = self.block_sparse_moe  # FFN使用稀疏MoE
        else:  # 否则没有专家
            raise NotImplementedError()  # 抛出未实现错误

    def forward(  # 解码器层前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor] = None,  # 残差（可选）
        deferred_norm: Optional[RMSNorm] = None,  # 延迟归一化（可选）
    ) -> Tuple[torch.Tensor, torch.Tensor, RMSNorm]:

        hidden_states_original = hidden_states  # 保存原始隐藏状态
        residual_original = residual  # 保存原始残差

        # Self Attention  # 自注意力
        if deferred_norm is not None:  # 如果有延迟归一化
            assert residual is not None  # 断言残差不为None
            # here hidden_states is output of ffn, residual is residual from after previous attn layer
            # 这里hidden_states是ffn的输出，residual是上一个注意力层后的残差
            hidden_states, residual = fused_dual_residual_rmsnorm(  # 融合双残差RMS归一化
                hidden_states,  # FFN输出
                residual,  # 之前的残差
                deferred_norm.weight,  # 延迟归一化权重
                self.pre_attn_norm.weight,  # 注意力前归一化权重
                deferred_norm.variance_epsilon,  # 延迟归一化epsilon
            )
        else:  # 否则没有延迟归一化
            # here hidden_states is the residual  # 这里hidden_states就是残差
            hidden_states, residual = (  # RMS归一化
                fused_rmsnorm(  # 融合RMS归一化
                    hidden_states,  # 隐藏状态
                    self.pre_attn_norm.weight,  # 注意力前归一化权重
                    self.pre_attn_norm.variance_epsilon,  # epsilon
                ),
                hidden_states,  # 原始隐藏状态作为残差
            )

        hidden_states = self.self_attn(  # 执行自注意力计算
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )

        if get_tensor_model_parallel_world_size() > 1:  # 如果TP大小大于1
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)  # 全归约

        hidden_states, residual = fused_dual_residual_rmsnorm(  # 融合双残差RMS归一化
            hidden_states,  # 注意力输出
            residual,  # 残差
            self.post_attn_norm.weight,  # 注意力后归一化权重
            self.pre_moe_norm.weight,  # MoE前归一化权重
            self.post_attn_norm.variance_epsilon,  # epsilon
        )

        # Fully Connected  # 全连接层
        hidden_states = self.ffn(hidden_states)  # 执行FFN/MoE计算
        return hidden_states, residual, self.post_moe_norm  # defer layernorm  # 返回隐藏状态、残差和延迟归一化

    def moe_with_rmoe(self, x):  # 带残差MoE的FFN
        if self.alt_stream is not None and get_is_capture_mode():  # 如果有备用流且在CUDA图捕获模式
            current_stream = torch.cuda.current_stream()  # 获取当前CUDA流
            self.alt_stream.wait_stream(current_stream)  # 等待当前流完成
            mlp_result = self.mlp(x)  # 在当前流上执行MLP
            with torch.cuda.stream(self.alt_stream):  # 在备用流上执行
                moe_result = self.block_sparse_moe(x)  # 执行MoE计算
            current_stream.wait_stream(self.alt_stream)  # 等待备用流完成
        else:  # 否则串行执行
            mlp_result = self.mlp(x)  # 执行MLP
            moe_result = self.block_sparse_moe(x)  # 执行MoE
        return (mlp_result + moe_result) / 1.4142135623730951  # 合并结果并除以sqrt(2)


class Grok1Model(nn.Module):  # Grok1模型主体
    def __init__(  # 初始化模型
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        load_presharded_moe: bool = False,  # 是否加载预分片MoE
        load_presharded_embedding: bool = False,  # 是否加载预分片嵌入
        load_presharded_attn: bool = False,  # 是否加载预分片注意力
        load_presharded_mlp: bool = False,  # 是否加载预分片MLP
        replicate_embedding: bool = False,  # 是否复制嵌入
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 创建词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            use_presharded_weights=load_presharded_embedding,  # 预分片权重
            enable_tp=not replicate_embedding,  # 是否启用张量并行
            prefix=add_prefix("embed_tokens", prefix),  # 参数前缀
        )

        self.alt_stream = torch.cuda.Stream()  # 创建备用CUDA流
        self.layers = nn.ModuleList(  # 创建解码器层列表
            [
                Grok1DecoderLayer(  # 每一层都是Grok1解码器层
                    config,  # 配置
                    i,  # 层ID
                    quant_config=quant_config,  # 量化配置
                    load_presharded_moe=load_presharded_moe,  # 预分片MoE
                    load_presharded_attn=load_presharded_attn,  # 预分片注意力
                    load_presharded_mlp=load_presharded_mlp,  # 预分片MLP
                    alt_stream=self.alt_stream,  # 备用CUDA流
                )
                for i in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def forward(  # 模型前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 从token ID获取嵌入
            hidden_states.mul_(self.config.embedding_multiplier_scale)  # 乘以嵌入缩放因子
        else:  # 否则
            hidden_states = input_embeds  # 直接使用输入嵌入

        residual, deferred_norm = None, None  # 初始化残差和延迟归一化
        for i in range(len(self.layers)):  # 遍历所有解码器层
            hidden_states, residual, deferred_norm = self.layers[i](  # 执行当前层前向传播
                positions, hidden_states, forward_batch, residual, deferred_norm
            )

        hidden_states, _ = fused_dual_residual_rmsnorm(  # 最终融合双残差RMS归一化
            hidden_states,  # 隐藏状态
            residual,  # 残差
            deferred_norm.weight,  # 延迟归一化权重
            self.norm.weight,  # 最终归一化权重
            deferred_norm.variance_epsilon,  # epsilon
        )

        return hidden_states  # 返回隐藏状态


class Grok1ForCausalLM(nn.Module):  # Grok1因果语言模型
    def __init__(  # 初始化因果语言模型
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        # Get presharded weights.  # 获取预分片权重配置
        self.load_presharded_mlp = getattr(config, "load_presharded_mlp", False)  # 是否加载预分片MLP
        self.load_presharded_moe = (  # 是否加载预分片MoE
            getattr(config, "load_presharded_moe", True)  # 默认为True
            and self.config.num_local_experts > 0  # 且有专家
            and get_tensor_model_parallel_world_size() > 1  # 且TP大小大于1
        )
        self.load_presharded_attn = getattr(config, "load_presharded_attn", False)  # 是否加载预分片注意力
        self.load_presharded_embedding = getattr(  # 是否加载预分片嵌入
            config, "load_presharded_embedding", False
        )

        default_replicate_lm_head = False  # 默认不复制语言模型头
        self.replicate_lm_head = getattr(  # 是否复制语言模型头
            config, "replicate_lm_head", default_replicate_lm_head
        )

        if get_tensor_model_parallel_world_size() > 1:  # 如果TP大小大于1
            setattr(DefaultModelLoader, "_prepare_weights", _prepare_presharded_weights)  # 设置预分片权重准备方法

        self.replicate_embedding = getattr(config, "replicate_embedding", False)  # 是否复制嵌入

        self.model = Grok1Model(  # 创建模型主体
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            load_presharded_moe=self.load_presharded_moe,  # 预分片MoE
            load_presharded_embedding=self.load_presharded_embedding,  # 预分片嵌入
            load_presharded_attn=self.load_presharded_attn,  # 预分片注意力
            load_presharded_mlp=self.load_presharded_mlp,  # 预分片MLP
            replicate_embedding=self.replicate_embedding,  # 复制嵌入
            prefix=add_prefix("model", prefix),  # 参数前缀
        )

        lm_head_params_dtype = None  # 语言模型头参数数据类型
        if self.replicate_lm_head:  # 如果复制语言模型头
            self.lm_head = ReplicatedLinear(  # 创建复制线性层作为语言模型头
                config.hidden_size,  # 输入大小
                config.vocab_size,  # 输出大小
                bias=False,  # 不使用偏置
                params_dtype=lm_head_params_dtype,  # 数据类型
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)  # 跳过全收集的logits处理器
        else:  # 否则不复制
            self.lm_head = ParallelLMHead(  # 创建并行语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                use_presharded_weights=self.load_presharded_embedding,  # 预分片权重
                params_dtype=lm_head_params_dtype,  # 数据类型
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
            )
            self.logits_processor = LogitsProcessor(config)  # 创建标准logits处理器

        self.loaded_param_names = set()  # 已加载参数名集合

    @torch.no_grad()  # 禁用梯度计算
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

    def load_weights(  # 加载模型权重
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],  # 权重迭代器
        ignore_parent_name: bool = False,  # 是否忽略父名称
        check_hit_names: bool = True,  # 是否检查命中名称
        model_config: PretrainedConfig | None = None,  # 模型配置（可选）
    ) -> dict[str, torch.Tensor]:
        if model_config is None:  # 如果没有提供模型配置
            model_config = self.config  # 使用自身配置

        stacked_params_mapping = []  # 堆叠参数映射列表
        stacked_params_mapping += [  # 添加QKV堆叠映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),  # Q投影
            ("qkv_proj", "k_proj", "k"),  # K投影
            ("qkv_proj", "v_proj", "v"),  # V投影
        ]
        stacked_params_mapping += [  # 添加gate_up堆叠映射
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),  # gate投影
            ("gate_up_proj", "up_proj", 1),  # up投影
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)  # 权重、fp8权重缩放和fp8激活缩放的参数映射
        num_experts = model_config.num_local_experts  # 专家数量
        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
            ckpt_gate_proj_name="w1",  # 检查点gate投影名称
            ckpt_down_proj_name="w2",  # 检查点down投影名称
            ckpt_up_proj_name="w3",  # 检查点up投影名称
            num_experts=num_experts,  # 专家数量
        )

        params_dict = dict(self.named_parameters())  # 获取参数字典
        all_names = set(params_dict.keys())  # 所有参数名集合
        hit_names = set()  # 命中参数名集合

        def load_weight_wrapper(  # 加载权重包装函数
            name: str, loaded_weight: torch.Tensor, *args, **kwargs
        ):
            # Fuse constant multipliers into the weights  # 将常量乘数融合到权重中
            if "lm_head" in name:  # 如果是语言模型头权重
                loaded_weight = (  # 应用输出乘数缩放
                    loaded_weight.to(torch.float32)
                    * model_config.output_multiplier_scale
                )

            original_name = name  # 保存原始名称
            if ignore_parent_name:  # 如果忽略父名称
                name = name.split(".")[-1]  # 只取最后一部分

            if name not in params_dict:  # 如果名称不在参数字典中
                logger.info(f"Skipping {name=} in load_weights_wrapper")  # 记录跳过信息
                return  # 返回

            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight, *args, **kwargs)  # 加载权重
            hit_names.add(name)  # 添加到命中集合
            self.loaded_param_names.add(original_name)  # 添加到已加载参数名集合

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码逆频率
                continue  # 跳过

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                load_weight_wrapper(name, loaded_weight, shard_id)  # 加载权重
                break  # 跳出循环
            else:  # 如果没有匹配堆叠参数映射
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不在参数名中
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名

                    load_weight_wrapper(  # 加载专家权重
                        name,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break  # 跳出循环
                else:  # 如果没有匹配专家参数映射
                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                        continue  # 跳过
                    if name is None:  # 如果名称为None
                        continue  # 跳过

                    load_weight_wrapper(name=name, loaded_weight=loaded_weight)  # 直接加载权重

        if check_hit_names:  # 如果检查命中名称
            if len(hit_names) > 5:  # 如果命中名称多于5个
                missing = all_names - hit_names  # 计算缺失名称
                missing_exclude_scales = {x for x in missing if "scale" not in x}  # 排除缩放因子
                logger.info(  # 记录命中信息
                    f"#all_names: {len(all_names)}, #hit_names: {len(hit_names)}, #missing_exclude_scales: {len(missing_exclude_scales)}",
                )
                if len(missing_exclude_scales) > 0:  # 如果有缺失（排除缩放因子后）
                    raise ValueError(  # 抛出错误
                        f"load_weights failed because some weights are missing: {missing_exclude_scales=}."
                    )

            elif len(hit_names) == 0:  # 如果没有命中任何名称
                raise ValueError(  # 抛出错误
                    f"load_weights failed because it did not hit any names. {all_names=} {hit_names=}"
                )

        return hit_names  # 返回命中名称集合

    def get_num_params_analytical(self):  # 解析计算模型参数数量
        cfg = self.config  # 获取配置
        moe_intermediate_size = getattr(  # 获取MoE中间层大小
            cfg,
            "moe_intermediate_size",
            getattr(cfg, "intermediate_size", None),
        )
        residual_moe = getattr(cfg, "residual_moe", False)  # 是否使用残差MoE
        if cfg.num_local_experts > 0:  # 如果有专家
            num_experts = cfg.num_local_experts + (1 if residual_moe else 0)  # 计算总专家数
        else:  # 否则
            num_experts = 1  # 只有1个（dense MLP）

        wq = (  # Q权重参数量
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        wkv = (  # KV权重参数量
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_key_value_heads
            * cfg.head_dim
            * 2
        )
        out = (  # 输出投影参数量
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        ffn1 = (  # FFN第一层参数量（gate+up）
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
            * 2
        )
        ffn2 = (  # FFN第二层参数量（down）
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
        )
        embed = cfg.hidden_size * cfg.vocab_size * 2  # 嵌入层参数量（输入+输出）
        return wq + wkv + out + ffn1 + ffn2 + embed  # 返回总参数量

    def get_num_params_torch(self):  # 使用PyTorch计算模型参数数量
        return (  # 返回总参数量（考虑张量并行）
            sum(p.numel() for p in self.parameters())
            * get_tensor_model_parallel_world_size()
        )


old_prepare_weights = getattr(DefaultModelLoader, "_prepare_weights")  # 保存原始权重准备方法


def _prepare_presharded_weights(  # 准备预分片权重
    self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
) -> Tuple[str, list[str], bool]:
    import glob  # 导入glob模块
    import os  # 导入os模块

    if get_tensor_model_parallel_world_size() == 1:  # 如果TP大小为1
        return old_prepare_weights(self, model_name_or_path, revision, fall_back_to_pt)  # 使用原始方法

    if not os.path.isdir(model_name_or_path):  # 如果不是本地目录
        from sglang.srt.model_loader.weight_utils import download_weights_from_hf  # 导入HF下载工具

        allow_patterns = ["*.safetensors", "*.bin"]  # 允许的文件模式
        hf_folder = download_weights_from_hf(  # 从HuggingFace下载权重
            model_name_or_path,  # 模型名称或路径
            self.load_config.download_dir,  # 下载目录
            allow_patterns,  # 允许的模式
            revision,  # 版本
            ignore_patterns=self.load_config.ignore_patterns,  # 忽略的模式
        )
    else:  # 否则是本地目录
        hf_folder = model_name_or_path  # 使用本地路径

    tp_rank = get_tensor_model_parallel_rank()  # 获取张量并行rank

    # The old format  # 旧格式
    allow_patterns = [f"*-{tp_rank:03d}.bin"]  # 旧格式权重文件模式

    # The new format  # 新格式
    allow_patterns += [f"*-TP-{tp_rank:03d}.safetensors", "*-TP-common.safetensors"]  # 新格式权重文件模式

    hf_weights_files = []  # 权重文件列表
    for pattern in allow_patterns:  # 遍历允许的模式
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))  # 查找匹配文件

    if not hf_weights_files:  # 如果没有找到预分片文件
        return old_prepare_weights(self, model_name_or_path, revision, fall_back_to_pt)  # 回退到原始方法

    if hf_weights_files[0].endswith("safetensors"):  # 如果是safetensors格式
        use_safetensors = True  # 使用safetensors
    else:  # 否则
        use_safetensors = False  # 不使用safetensors

    return hf_folder, hf_weights_files, use_safetensors  # 返回路径、文件列表和格式标志


class Grok1ModelForCausalLM(Grok1ForCausalLM):  # Grok1模型别名，向后兼容
    """An alias for backward-compatbility."""  # 向后兼容的别名

    pass  # 直接继承，无额外实现


EntryClass = [Grok1ForCausalLM, Grok1ModelForCausalLM]  # 入口类列表，用于模型注册
