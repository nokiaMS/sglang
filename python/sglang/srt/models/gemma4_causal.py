# Copyright 2025 SGLang Team  # 版权所有2025 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and  # 查看许可证了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================  # 分隔线
# Gemma4因果语言模型实现文件
# 本文件实现了Gemma4文本模型的核心组件，包括注意力机制、MoE路由、解码器层和文本模型，
# 支持流水线并行（PP）、张量并行（TP）、KV共享层和Per-Layer Embedding (PLE)。

import logging  # 导入日志模块
import re  # 导入正则表达式模块
from typing import Iterable, List, Optional, Set, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import (  # 从transformers导入
    Gemma4TextConfig,  # Gemma4文本配置
    PretrainedConfig,  # 预训练配置
    PreTrainedModel,  # 预训练模型基类
)

from sglang.srt.distributed import (  # 导入分布式模块
    get_pp_group,  # 获取PP组
    get_tensor_model_parallel_rank,  # 获取TP排名
    get_tensor_model_parallel_world_size,  # 获取TP世界大小
)
from sglang.srt.layers.gemma4_fused_ops import (  # 导入Gemma4融合操作
    gemma_dual_rmsnorm_residual_scalar,  # 双RMSNorm残差标量融合
    gemma_qkv_rmsnorm,  # QKV RMSNorm融合
    gemma_rmsnorm_residual_scalar,  # RMSNorm残差标量融合
)
from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm  # 导入归一化层
from sglang.srt.layers.linear import (  # 导入线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类获取函数
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE层
from sglang.srt.layers.moe.topk import TopK  # 导入TopK模块
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入工具函数
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称
)
from sglang.srt.models.gemma3_causal import Gemma3MLP, Gemma3TextScaledWordEmbedding  # 导入Gemma3 MLP和缩放嵌入
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import add_prefix, make_layers  # 导入工具函数

logger = logging.getLogger(__name__)  # 创建日志记录器


# Aligned with HF's implementation, using sliding window inclusive with the last token  # 与HF实现对齐，滑动窗口包含最后一个token
# SGLang assumes exclusive  # SGLang假设排他
def get_attention_sliding_window_size(config):  # 获取注意力滑动窗口大小
    return config.sliding_window - 1  # 返回滑动窗口减1（排他）


Gemma4MLP = Gemma3MLP  # Gemma4 MLP使用Gemma3的MLP实现
Gemma4TextScaledWordEmbedding = Gemma3TextScaledWordEmbedding  # Gemma4缩放嵌入使用Gemma3的实现


def pp_filter_load_weight(  # PP过滤加载权重函数
    name,  # 参数名
    loaded_weight,  # 加载的权重
    *,  # 以下为仅关键字参数
    pp_group,  # PP组
    start_layer,  # 起始层
    end_layer,  # 结束层
    params_dict,  # 参数字典
    loaded_params,  # 已加载参数集合
    tie_word_embeddings,  # 是否绑定词嵌入
    embed_weight_name,  # 嵌入权重名称
    first_rank_only_patterns=(),  # 仅第一个rank的模式
    last_rank_only_prefixes=(),  # 仅最后一个rank的前缀
    head_param_name="lm_head.weight",  # 头参数名称
):
    """Shared PP filter for Gemma4 load_weights paths.  # Gemma4 load_weights路径的共享PP过滤器

    Returns True if the caller should ``continue`` (handled or skipped),  # 如果调用者应continue（已处理或跳过）则返回True
    False otherwise.  No-op when ``pp_group.world_size == 1``.  # 否则返回False。pp_group.world_size==1时为空操作

    Handles three concerns in order:  # 按顺序处理三个关注点：
      1. Drop transformer-layer weights outside [start_layer, end_layer).  # 丢弃[start_layer, end_layer)之外的transformer层权重
      2. Route the tied ``embed_tokens.weight`` to ``lm_head`` on the last  # 将绑定的embed_tokens.weight路由到最后一个rank上的lm_head
         rank (under PP, embed and lm_head live on different ranks so they  # （在PP下，embed和lm_head位于不同rank，因此
         can't be tied via module aliasing).  # 无法通过模块别名绑定）
      3. Skip rank-local module weights on the wrong rank.  # 跳过错误rank上的rank本地模块权重
    """
    if pp_group.world_size <= 1:  # 如果PP世界大小<=1
        return False  # 不需要过滤

    layer_id = get_layer_id(name)  # 获取层ID
    if layer_id is not None and (layer_id < start_layer or layer_id >= end_layer):  # 如果层ID在范围外
        return True  # 跳过

    if tie_word_embeddings and pp_group.is_last_rank and name == embed_weight_name:  # 如果绑定词嵌入且是最后rank且是嵌入权重
        head_param = params_dict.get(head_param_name)  # 获取头参数
        if head_param is not None:  # 如果头参数存在
            wl = getattr(head_param, "weight_loader", default_weight_loader)  # 获取权重加载器
            wl(head_param, loaded_weight)  # 加载权重到头参数
            loaded_params.add(head_param_name)  # 添加到已加载集合
        return True  # 已处理

    if not pp_group.is_first_rank and any(p in name for p in first_rank_only_patterns):  # 如果不是第一个rank且匹配仅第一个rank模式
        return True  # 跳过

    if not pp_group.is_last_rank and any(  # 如果不是最后一个rank且匹配仅最后一个rank前缀
        name.startswith(p) for p in last_rank_only_prefixes  # 检查前缀匹配
    ):
        return True  # 跳过

    return False  # 不过滤


class Gemma4Router(nn.Module):  # Gemma4 MoE路由器类
    """Router for Gemma4 MoE that preprocesses input before projection.  # Gemma4 MoE的路由器，在投影前预处理输入

    Applies RMSNorm (no learned weight), root_size scaling  # 应用RMSNorm（无学习权重）、root_size缩放
    (hidden_size^{-0.5}), then a learned per-dimension scale before  # （hidden_size^{-0.5}），然后是学习的每维度缩放，
    projecting to expert logits.  # 最后投影到专家logits

    This preprocessing is applied ONLY to the router's input, not to  # 此预处理仅应用于路由器的输入，
    the expert MLPs' input.  # 不应用于专家MLP的输入
    """

    def __init__(  # 初始化方法
        self,
        config,  # 配置
        quant_config: QuantizationConfig | None = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小

        # RMSNorm without learned weight — pure normalization only  # 无学习权重的RMSNorm——仅纯归一化
        self.norm = Gemma4RMSNorm(  # RMSNorm归一化
            self.hidden_size, eps=config.rms_norm_eps, with_scale=False  # 无缩放参数
        )
        # Per-dimension learned scale, applied after norm + root_size  # 每维度学习缩放，在norm + root_size之后应用
        self.scale = nn.Parameter(torch.ones(self.hidden_size))  # 可学习的每维度缩放参数
        # Constant 1/sqrt(hidden_size) scaling factor  # 常量1/sqrt(hidden_size)缩放因子
        self.register_buffer(  # 注册缓冲区
            "root_size",  # 名称
            torch.tensor(self.hidden_size**-0.5),  # hidden_size的负平方根
            persistent=False,  # 非持久化
        )
        # Project to expert logits; replicated across TP for consistent routing  # 投影到专家logits；跨TP复制以保持一致路由
        self.proj = ReplicatedLinear(  # 复制线性层
            self.hidden_size,  # 输入维度
            config.num_experts,  # 输出维度（专家数）
            bias=False,  # 无偏置
            quant_config=None,  # 无量化
            prefix=add_prefix("proj", prefix),  # 添加前缀
        )
        self._fused_scale: Optional[torch.Tensor] = None  # 融合缩放因子

    def fuse_scale(self):  # 融合缩放方法
        """Pre-compute scale * root_size. Call after weights are loaded."""  # 预计算scale * root_size。在权重加载后调用
        self._fused_scale = (self.scale * self.root_size).to(self.scale.dtype)  # 计算融合缩放并转换数据类型

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        """Returns raw router logits [T, E]."""  # 返回原始路由logits [T, E]
        x = self.norm(x)  # 归一化
        if self._fused_scale is None:  # 如果融合缩放未计算
            self.fuse_scale()  # 计算融合缩放
        x = x * self._fused_scale.to(x.dtype)  # 乘以融合缩放
        router_logits, _ = self.proj(x)  # 投影到专家logits
        return router_logits  # 返回路由logits


class Gemma4MoE(nn.Module):  # Gemma4 MoE类
    """Mixture of Experts for Gemma4.  # Gemma4的混合专家模型

    Wraps MoE implementation with custom routing. The router projection is  # 用自定义路由包装MoE实现。路由器投影是
    external (Gemma4Router) — this class only handles expert dispatch.  # 外部的（Gemma4Router）——此类仅处理专家分发

    Gemma4 routing: softmax over ALL experts → top-k → renormalize.  # Gemma4路由：对所有专家softmax → top-k → 重新归一化
    per_expert_scale is folded into routing weights for mathematical  # per_expert_scale被折叠到路由权重中以确保
    correctness with MoE's fused kernel.  # MoE融合核的数学正确性
    """

    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏层大小
        layer_id: int,  # 层ID
        config: Gemma4TextConfig,  # Gemma4文本配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层ID
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.num_experts = config.num_experts  # 专家数量
        self.tp_size = get_tensor_model_parallel_world_size()  # TP世界大小

        # Per-expert output scale folded into routing weights so that  # 每专家输出缩放折叠到路由权重中，使得
        # MoE's fused kernel computes: Σ_e (expert_e * w_e * scale_e)  # MoE的融合核计算：Σ_e (expert_e * w_e * scale_e)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))  # 每专家缩放参数

        # Capture param directly to avoid closing over self in the routing closure.  # 直接捕获参数以避免在路由闭包中关闭self
        per_expert_scale = self.per_expert_scale  # 局部变量引用

        def routing_function(  # 路由函数
            hidden_states: torch.Tensor,  # 隐藏状态
            gating_output: torch.Tensor,  # 门控输出
            topk: int,  # top-k值
            renormalize: bool,  # always True for Gemma4; softmax identity only holds when renormalizing  # 对Gemma4始终为True；softmax恒等式仅在重新归一化时成立
        ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回权重和索引元组
            # softmax(all)[topk] / sum(softmax(all)[topk]) = softmax(topk_logits),  # softmax(all)[topk] / sum(softmax(all)[topk]) = softmax(topk_logits)，
            # so we softmax only the top-k logits (fewer kernel launches).  # 因此我们仅对top-k logits做softmax（更少的内核启动）
            topk_logits, topk_ids = torch.topk(gating_output, k=topk, dim=-1)  # 获取top-k logits和ID
            topk_weights = torch.nn.functional.softmax(topk_logits, dim=-1)  # 对top-k logits做softmax

            # Fold per_expert_scale into routing weights  # 将per_expert_scale折叠到路由权重中
            topk_weights = topk_weights * per_expert_scale[topk_ids].to(  # 乘以每专家缩放
                topk_weights.dtype  # 转换为相同数据类型
            )

            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)  # 返回float32权重和int32索引

        self.topk = TopK(  # TopK模块
            top_k=config.top_k_experts,  # top-k专家数
            layer_id=layer_id,  # 层ID
            custom_routing_function=routing_function,  # 自定义路由函数
        )

        experts_type = get_moe_impl_class(quant_config)  # 获取MoE实现类

        self.experts = experts_type(  # 创建专家层
            num_experts=config.num_experts  # 专家数量
            + get_global_server_args().ep_num_redundant_experts,  # 加上冗余专家数
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
            layer_id=layer_id,  # 层ID
            top_k=config.top_k_experts,  # top-k值
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("experts", prefix),  # 添加前缀
            activation="gelu",  # 激活函数为GELU
            reduce_results=True,  # 归约结果
        )

    def forward(  # 前向传播方法
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor  # 隐藏状态和路由logits
    ) -> torch.Tensor:  # 返回张量
        num_tokens, hidden_dim = hidden_states.shape  # 获取token数和隐藏维度
        topk_output = self.topk(hidden_states, router_logits)  # TopK选择
        hidden_states = self.experts(hidden_states, topk_output)  # 通过专家层
        return hidden_states.view(num_tokens, hidden_dim)  # 重塑并返回


class Gemma4Attention(nn.Module):  # Gemma4注意力类
    def __init__(  # 初始化方法
        self,
        layer_id: int,  # 层ID
        config: Gemma4TextConfig,  # Gemma4文本配置
        head_dim: int,  # 头维度
        max_position_embeddings: int,  # 最大位置嵌入数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.layer_id = layer_id  # 保存层ID
        self.config = config  # 保存配置
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP世界大小

        layer_type = config.layer_types[layer_id]  # 获取当前层类型
        self.sliding_window = (  # 滑动窗口大小
            config.sliding_window if layer_type == "sliding_attention" else None  # 滑动注意力层使用配置值，否则None
        )

        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个TP的头数

        if layer_type == "sliding_attention":  # 如果是滑动注意力层
            self.total_num_kv_heads = getattr(  # 获取KV头数
                config, "swa_num_key_value_heads", config.num_key_value_heads  # 优先使用swa_kv头数
            )
        else:  # 全注意力层
            self.total_num_kv_heads = config.num_key_value_heads  # 使用配置的KV头数

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个TP的KV头数

        if self.total_num_kv_heads >= tp_size:  # 如果KV头数>=TP大小
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # KV头数<TP大小
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除

        hidden_size = config.hidden_size  # 隐藏层大小
        self.head_dim = head_dim  # 头维度

        self.q_size = self.num_heads * self.head_dim  # Q大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV大小

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性层
            hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=config.attention_bias,  # 是否有偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 添加前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=config.attention_bias,  # 是否有偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 添加前缀
        )

        self.q_norm = Gemma4RMSNorm(  # Q归一化
            self.head_dim,  # 归一化维度
            eps=config.rms_norm_eps,  # epsilon值
        )
        self.k_norm = Gemma4RMSNorm(  # K归一化
            self.head_dim,  # 归一化维度
            eps=config.rms_norm_eps,  # epsilon值
        )
        self.v_norm = Gemma4RMSNorm(  # V归一化
            self.head_dim, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False  # 无缩放参数
        )

        if layer_type in config.rope_parameters:  # 如果层类型在RoPE参数中
            rope_parameters = dict(config.rope_parameters[layer_type])  # 获取该层类型的RoPE参数
        else:  # 使用默认RoPE参数
            rope_parameters = dict(  # 默认参数
                rope_type="default",  # 默认类型
                rope_theta=10000.0,  # 默认theta
            )

        # KV sharing logic  # KV共享逻辑
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)  # KV共享层数
        first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers  # 第一个KV共享层索引
        self.is_kv_shared_layer = (  # 是否是KV共享层
            layer_id >= first_kv_shared_layer_idx and num_kv_shared_layers > 0  # 当前层>=第一个共享层且共享层数>0
        )

        self.kv_shared_layer_index = None  # KV共享层索引
        if num_kv_shared_layers > 0 and self.layer_id >= first_kv_shared_layer_idx:  # 如果有KV共享层且当前层是共享层
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]  # 之前的层类型列表
            current_layer_type = config.layer_types[self.layer_id]  # 当前层类型
            if current_layer_type not in prev_layers:  # 如果当前层类型在之前不存在
                raise ValueError(  # 抛出值错误
                    f"KV sharing layer {self.layer_id} has type '{current_layer_type}' "  # KV共享层类型
                    f"but no matching type found in layers 0..{first_kv_shared_layer_idx - 1}. "  # 但在之前的层中未找到匹配类型
                    f"Available types: {set(prev_layers)}"  # 可用类型
                )
            self.kv_shared_layer_index = (  # 计算KV共享层索引
                len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)  # 反向查找最后一个相同类型的层
            )

        self.rotary_emb = get_rope(  # 创建旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_parameters.get("rope_theta", 10000.0),  # 基础theta
            rope_scaling={"rope_type": rope_parameters.get("rope_type", "default")},  # RoPE缩放
            partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),  # 部分旋转因子
            is_neox_style=True,  # Neox风格
        )

        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            1,  # scaling factor  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=(  # 层ID
                self.kv_shared_layer_index if self.is_kv_shared_layer else self.layer_id  # 共享层使用共享索引
            ),
            logit_cap=0.0,  # logit上限
            sliding_window_size=self.sliding_window,  # 滑动窗口大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
        )

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        **kwargs,  # 其他关键字参数
    ):
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割Q、K、V

        # Fused Q/K/V RMSNorm: replaces three separate norm kernels with one.  # 融合Q/K/V RMSNorm：用一个内核替代三个独立的归一化内核
        # Preconditions for the fused path: tensors on CUDA, q_norm/k_norm use  # 融合路径的前提条件：张量在CUDA上，q_norm/k_norm使用
        # the standard norm*weight (scale_shift==0) and v_norm has weight=ones  # 标准的norm*weight（scale_shift==0），v_norm权重为1
        # (with_scale=False) — the canonical Gemma4 attention configuration.  # （with_scale=False）——标准的Gemma4注意力配置
        is_kv_shared = (  # 是否是KV共享层
            self.is_kv_shared_layer and self.kv_shared_layer_index is not None  # 是KV共享层且索引不为None
        )
        can_fuse_qkv_norm = (  # 是否可以融合QKV归一化
            q.is_cuda  # Q在CUDA上
            and self.q_norm.scale_shift == 0.0  # Q归一化scale_shift为0
            and self.k_norm.scale_shift == 0.0  # K归一化scale_shift为0
            and not self.v_norm.with_scale  # V归一化无缩放
        )
        if can_fuse_qkv_norm:  # 如果可以融合
            if is_kv_shared:  # 如果是KV共享层
                gemma_qkv_rmsnorm(  # 融合QKV RMSNorm
                    q,  # Q张量
                    None,  # K为None（共享层不需要）
                    None,  # V为None（共享层不需要）
                    self.q_norm.weight.data,  # Q归一化权重
                    None,  # K归一化权重为None
                    num_q_heads=self.num_heads,  # Q头数
                    num_kv_heads=self.num_kv_heads,  # KV头数
                    head_dim=self.head_dim,  # 头维度
                    eps=self.q_norm.eps,  # epsilon值
                )
                k = None  # K设为None
                v = None  # V设为None
            else:  # 非KV共享层
                gemma_qkv_rmsnorm(  # 融合QKV RMSNorm
                    q,  # Q张量
                    k,  # K张量
                    v,  # V张量
                    self.q_norm.weight.data,  # Q归一化权重
                    self.k_norm.weight.data,  # K归一化权重
                    num_q_heads=self.num_heads,  # Q头数
                    num_kv_heads=self.num_kv_heads,  # KV头数
                    head_dim=self.head_dim,  # 头维度
                    eps=self.q_norm.eps,  # epsilon值
                )
                # Match the original norm path's output shapes: q stays 2D,  # 匹配原始归一化路径的输出形状：Q保持2D，
                # k/v become 3D so the subsequent `.flatten(-2, -1)` works.  # K/V变为3D以便后续的.flatten(-2, -1)正常工作
                # Use reshape (not view) since k/v are strided slice views of  # 使用reshape（而非view）因为K/V是qkv缓冲区的步幅切片视图
                # the qkv buffer and may not satisfy view's contiguity rules.  # 可能不满足view的连续性规则
                k = k.reshape(-1, self.num_kv_heads, self.head_dim)  # 重塑K为3D
                v = v.reshape(-1, self.num_kv_heads, self.head_dim)  # 重塑V为3D
        else:  # 不能融合
            q = q.unflatten(-1, (self.num_heads, self.head_dim))  # 反扁平化Q
            q = self.q_norm(q)  # Q归一化
            q = q.flatten(-2, -1)  # 扁平化Q
            if is_kv_shared:  # 如果是KV共享层
                k = None  # K设为None
                v = None  # V设为None
            else:  # 非KV共享层
                k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))  # 反扁平化K
                k = self.k_norm(k)  # K归一化
                v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))  # 反扁平化V
                v = self.v_norm(v)  # V归一化

        # Apply rotary embedding  # 应用旋转位置编码
        if k is not None:  # 如果K不为None
            k = k.flatten(-2, -1)  # 扁平化K
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))  # 反扁平化K
        else:  # K为None（KV共享层）
            # Rotary embedding requires a key input; use zeros since KV is shared from another layer  # 旋转位置编码需要键输入；使用零张量因为KV从另一层共享
            dummy_k = torch.zeros_like(q[:, : self.kv_size])  # 创建零张量
            q, _ = self.rotary_emb(positions, q, dummy_k)  # 仅对Q应用旋转位置编码

        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # 反扁平化Q
        attn_output = self.attn(  # 通过基数注意力
            q,  # Q张量
            k,  # K张量
            v,  # V张量
            forward_batch=forward_batch,  # 前向批次
            save_kv_cache=not self.is_kv_shared_layer,  # 非共享层保存KV缓存
        )
        if attn_output.dim() == 3:  # 如果注意力输出是3维
            attn_output = attn_output.flatten(-2, -1)  # 扁平化最后两维
        output, _ = self.o_proj(attn_output)  # 通过输出投影

        return output  # 返回输出


class Gemma4DecoderLayer(nn.Module):  # Gemma4解码器层类
    def __init__(  # 初始化方法
        self,
        layer_id: int,  # 层ID
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.hidden_size_per_layer_input = (  # 每层输入隐藏大小
            getattr(config, "hidden_size_per_layer_input", None) or 0  # 从配置获取或默认0
        )

        self.layer_id = layer_id  # 保存层ID

        # Gemma 4 uses different head dimensions for sliding vs full attention  # Gemma 4对滑动注意力和全注意力使用不同的头维度
        layer_type = config.layer_types[layer_id]  # 获取层类型
        self.is_full_attention = layer_type == "full_attention"  # 是否是全注意力层
        if self.is_full_attention:  # 如果是全注意力
            head_dim = config.head_dim  # following sglang naming  # 遵循sglang命名
        else:  # 滑动注意力
            head_dim = getattr(config, "swa_head_dim", config.head_dim)  # 使用swa头维度或默认头维度

        self.self_attn = Gemma4Attention(  # 自注意力层
            layer_id=layer_id,  # 层ID
            config=config,  # 配置
            max_position_embeddings=config.max_position_embeddings,  # 最大位置嵌入
            head_dim=head_dim,  # 头维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
        )

        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(  # 第一个KV共享层索引
            config, "num_kv_shared_layers", 0  # 减去KV共享层数
        )
        is_kv_shared_layer = self.layer_id >= first_kv_shared_layer_idx > 0  # 是否是KV共享层
        use_double_wide_mlp = (  # 是否使用双宽MLP
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer  # 配置启用且是KV共享层
        )
        layer_intermediate_size = config.intermediate_size * (  # 层中间大小
            2 if use_double_wide_mlp else 1  # 双宽MLP则乘2
        )

        self.mlp = Gemma4MLP(  # MLP层
            hidden_size=self.hidden_size,  # 隐藏大小
            intermediate_size=layer_intermediate_size,  # 中间大小
            hidden_activation=config.hidden_activation,  # 隐藏激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 添加前缀
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )
        self.pre_feedforward_layernorm = RMSNorm(  # 前馈前归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )
        self.post_feedforward_layernorm = RMSNorm(  # 前馈后归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )

        # Per-Layer Embedding (PLE) components — present in each decoder layer  # 每层嵌入（PLE）组件——存在于每个解码器层
        if self.hidden_size_per_layer_input > 0:  # 如果有每层输入
            # Gate: projects hidden_states → per-layer dim for gating  # 门控：投影隐藏状态 → 每层维度用于门控
            self.per_layer_input_gate = ReplicatedLinear(  # 每层输入门控
                self.hidden_size,  # 输入维度
                self.hidden_size_per_layer_input,  # 输出维度
                bias=False,  # 无偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("per_layer_input_gate", prefix),  # 添加前缀
            )
            # Projection: projects gated per-layer input back → hidden size  # 投影：将门控后的每层输入投影回 → 隐藏大小
            self.per_layer_projection = ReplicatedLinear(  # 每层投影
                self.hidden_size_per_layer_input,  # 输入维度
                self.hidden_size,  # 输出维度
                bias=False,  # 无偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("per_layer_projection", prefix),  # 添加前缀
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(  # 每层输入后归一化
                config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
            )
        else:  # 没有每层输入
            self.per_layer_input_gate = None  # 门控为None
            self.per_layer_projection = None  # 投影为None
            self.post_per_layer_input_norm = None  # 归一化为None

        # Parallel MoE  # 并行MoE
        self.enable_moe_block = getattr(config, "enable_moe_block", False)  # 是否启用MoE块
        if self.enable_moe_block:  # 如果启用MoE
            self.router = Gemma4Router(  # MoE路由器
                config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("router", prefix),  # 添加前缀
            )
            self.moe = Gemma4MoE(  # MoE模块
                hidden_size=self.hidden_size,  # 隐藏大小
                layer_id=layer_id,  # 层ID
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("moe", prefix),  # 添加前缀
            )

            self.post_feedforward_layernorm_1 = RMSNorm(  # 前馈后归一化1（MLP分支）
                config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
            )
            self.post_feedforward_layernorm_2 = RMSNorm(  # 前馈后归一化2（MoE分支）
                config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(  # 前馈前归一化2（MoE分支）
                config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
            )
        else:  # 未启用MoE
            self.router = None  # 路由器为None
            self.moe = None  # MoE为None
            self.post_feedforward_layernorm_1 = None  # 归一化1为None
            self.post_feedforward_layernorm_2 = None  # 归一化2为None
            self.pre_feedforward_layernorm_2 = None  # 前归一化2为None

        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)  # 层标量
        self.has_ple = self.hidden_size_per_layer_input > 0  # 是否有PLE
        self.prefix = prefix  # 保存前缀

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        per_layer_input: torch.Tensor,  # 每层输入
        forward_batch: ForwardBatch,  # 前向批次
        **kwargs,  # 其他关键字参数
    ) -> tuple[  # 返回元组
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]  # 隐藏状态和可选的额外输出
    ]:
        # Gemma4 residual pattern following JAX implementation:  # Gemma4残差模式，遵循JAX实现：
        # 1. input_norm(x) -> attn -> post_attn_norm -> ADD residual  # 1. 输入归一化(x) -> 注意力 -> 注意力后归一化 -> 加残差
        # 2. pre_ff_norm -> mlp -> post_ff_norm -> ADD residual  # 2. 前馈前归一化 -> MLP -> 前馈后归一化 -> 加残差
        #
        # Optimization: fuse "post_attn_norm(h) + residual; pre_ff_norm(...)"  # 优化：融合"注意力后归一化(h) + 残差; 前馈前归一化(...)"
        # into "post_attn_norm(h); pre_ff_norm(h, residual)" using  # 为"注意力后归一化(h); 前馈前归一化(h, residual)"，
        # gemma_fused_add_rmsnorm which computes:  # 使用gemma_fused_add_rmsnorm，它计算：
        #   residual = h + residual (in-place)  #   residual = h + residual（原地操作）
        #   h = gemma_norm(residual)  #   h = gemma_norm(residual)
        residual = hidden_states  # 保存残差

        # Apply input layernorm  # 应用输入层归一化
        hidden_states = self.input_layernorm(hidden_states)  # 输入归一化
        hidden_states = self.self_attn(  # 自注意力
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化

        if self.enable_moe_block:  # 如果启用MoE块
            # Fuse: hidden_states + residual -> residual; pre_ff_norm(residual) -> hidden_states  # 融合：隐藏状态 + 残差 -> 残差；前馈前归一化(残差) -> 隐藏状态
            # Also need raw (unfused) residual for router and pre_ff_norm_2  # 路由器和前馈前归一化2还需要原始（未融合的）残差
            hidden_states, residual = self.pre_feedforward_layernorm(  # 前馈前归一化（融合残差加法）
                hidden_states, residual  # 隐藏状态和残差
            )
            # For MoE: router and pre_ff_norm_2 need the unfused residual  # MoE：路由器和前馈前归一化2需要未融合的残差
            # (which is now updated to post_attn_out + old_residual)  # （现在已更新为注意力输出 + 旧残差）
            moe_input = residual  # MoE输入

            # Dense MLP branch  # 稠密MLP分支
            hidden_states_1 = self.mlp(hidden_states)  # 通过MLP

            # MoE branch: router sees residual (= post_attn_out + old_residual)  # MoE分支：路由器看到残差（= 注意力输出 + 旧残差）
            router_logits = self.router(moe_input)  # 路由器计算
            hidden_states_2 = self.pre_feedforward_layernorm_2(moe_input)  # MoE前归一化
            hidden_states_2 = self.moe(hidden_states_2, router_logits)  # 通过MoE

            # Fused: (rmsnorm(rmsnorm(h1,w1) + rmsnorm(h2,w2), w3) + residual) * scalar  # 融合：(rmsnorm(rmsnorm(h1,w1) + rmsnorm(h2,w2), w3) + 残差) * 标量
            if (  # 如果可以融合
                not self.has_ple  # 无PLE
                and hidden_states_1.is_cuda  # 在CUDA上
                and hidden_states_1.dim() == 2  # 是2维
            ):
                norm1 = self.post_feedforward_layernorm_1  # MLP后归一化
                norm2 = self.post_feedforward_layernorm_2  # MoE后归一化
                norm3 = self.post_feedforward_layernorm  # 最终后归一化
                hidden_states = gemma_dual_rmsnorm_residual_scalar(  # 双RMSNorm残差标量融合
                    hidden_states_1,  # MLP输出
                    norm1.weight.data,  # 归一化1权重
                    hidden_states_2,  # MoE输出
                    norm2.weight.data,  # 归一化2权重
                    norm3.weight.data,  # 归一化3权重
                    residual,  # 残差
                    self.layer_scalar,  # 层标量
                    norm1.variance_epsilon,  # 方差epsilon1
                    norm2.variance_epsilon,  # 方差epsilon2
                    norm3.variance_epsilon,  # 方差epsilon3
                )
                return hidden_states, None  # 返回隐藏状态和None

            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_1)  # MLP后归一化
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)  # MoE后归一化

            # Combine branches  # 合并两个分支
            hidden_states = hidden_states_1 + hidden_states_2  # 相加
        else:  # 未启用MoE
            # Fuse: hidden_states + residual -> residual; pre_ff_norm(residual) -> hidden_states  # 融合：隐藏状态 + 残差 -> 残差；前馈前归一化(残差) -> 隐藏状态
            hidden_states, residual = self.pre_feedforward_layernorm(  # 前馈前归一化（融合残差加法）
                hidden_states, residual  # 隐藏状态和残差
            )
            hidden_states = self.mlp(hidden_states)  # 通过MLP

        if not self.has_ple and hidden_states.is_cuda and hidden_states.dim() == 2:  # 无PLE且在CUDA且2维
            # Fused: (post_ff_norm(h) + residual) * layer_scalar in one kernel  # 融合：(前馈后归一化(h) + 残差) * 层标量在一个内核中
            norm = self.post_feedforward_layernorm  # 前馈后归一化
            hidden_states = gemma_rmsnorm_residual_scalar(  # RMSNorm残差标量融合
                hidden_states,  # 隐藏状态
                norm.weight.data,  # 归一化权重
                residual,  # 残差
                self.layer_scalar,  # 层标量
                norm.variance_epsilon,  # 方差epsilon
            )
        else:  # 非融合路径
            hidden_states = self.post_feedforward_layernorm(hidden_states)  # 前馈后归一化
            hidden_states = hidden_states + residual  # 加残差

            if self.has_ple and per_layer_input is not None:  # 如果有PLE且提供了每层输入
                gate, _ = self.per_layer_input_gate(hidden_states)  # 门控投影
                gate = torch.nn.functional.gelu(gate, approximate="tanh")  # GELU激活
                gated_per_layer = gate * per_layer_input  # 门控后的每层输入
                per_layer_contribution, _ = self.per_layer_projection(gated_per_layer)  # 投影回隐藏空间
                per_layer_contribution = self.post_per_layer_input_norm(  # 每层输入后归一化
                    per_layer_contribution  # 每层贡献
                )
                hidden_states = hidden_states + per_layer_contribution  # 加每层贡献

            hidden_states = hidden_states * self.layer_scalar  # 乘以层标量
        return hidden_states, None  # 返回隐藏状态和None


class Gemma4TextModel(PreTrainedModel):  # Gemma4文本模型类
    def __init__(  # 初始化方法
        self,
        config: Gemma4TextConfig,  # Gemma4文本配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.vocab_size = config.vocab_size  # 词表大小
        self.padding_idx = getattr(config, "pad_token_id", None)  # 填充token ID
        self.pp_group = get_pp_group()  # 获取PP组

        # Token / per-layer embedding tables and the per-layer projection only  # token/每层嵌入表和每层投影仅
        # produce activations consumed at the model entry, so they live on the  # 产生在模型入口处消费的激活，因此它们存在于
        # first PP rank only.  Other ranks substitute PPMissingLayer so that  # 仅第一个PP rank上。其他rank用PPMissingLayer替代
        # parameter iteration still works (load_weights skips them explicitly).  # 以便参数迭代仍可工作（load_weights显式跳过它们）
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.hidden_size_per_layer_input = (  # 每层输入隐藏大小
            getattr(config, "hidden_size_per_layer_input", None) or 0  # 从配置获取或默认0
        )
        self.vocab_size_per_layer_input = (  # 每层输入词表大小
            getattr(config, "vocab_size_per_layer_input", None) or config.vocab_size  # 从配置获取或使用主词表大小
        )

        # PLE-enabled variants (E2B/E4B) forward `per_layer_inputs` through  # 启用PLE的变体（E2B/E4B）通过
        # the PP proxy, but cuda_graph_runner hardcodes the proxy schema to  # PP代理转发per_layer_inputs，但cuda_graph_runner将代理模式硬编码为
        # {hidden_states, residual} and silently drops any extra keys at  # {hidden_states, residual}，在重放时静默丢弃任何额外的键
        # replay time.  Empirically this corrupts E4B output to garbage on  # 经验上这会将非第一PP rank上的E4B输出损坏为垃圾
        # non-first PP ranks (eager path produces correct output and  # （eager路径产生正确输出且
        # GSM8K ~0.92, cuda-graph path emits token soup).  Refuse the  # GSM8K ~0.92，cuda-graph路径产生token汤）。拒绝
        # combination until the runner becomes schema-aware; users can run  # 此组合直到runner变为模式感知；用户可以
        # PP + PLE eagerly with --disable-cuda-graph.  # 使用--disable-cuda-graph以eager模式运行PP + PLE
        if self.pp_group.world_size > 1 and self.hidden_size_per_layer_input > 0:  # 如果PP>1且有PLE
            sa = get_global_server_args()  # 获取全局服务器参数
            if sa is not None and not sa.disable_cuda_graph:  # 如果未禁用CUDA图
                raise ValueError(  # 抛出值错误
                    "Pipeline parallelism is currently incompatible with "  # 流水线并行当前不兼容
                    "per-layer-input (PLE) embeddings under CUDA graph: "  # CUDA图下的每层输入（PLE）嵌入：
                    "the runner's PP proxy schema is hardcoded to "  # runner的PP代理模式被硬编码为
                    "{hidden_states, residual} and silently drops "  # {hidden_states, residual}并静默丢弃
                    "per_layer_inputs, corrupting per-layer contributions on "  # per_layer_inputs，损坏非第一PP rank上的
                    "non-first PP ranks. Workarounds: (a) pass "  # 每层贡献。变通方法：(a) 传递
                    "--disable-cuda-graph to fall back to eager replay, or "  # --disable-cuda-graph回退到eager重放，或
                    "(b) use tensor parallelism (--tp-size) instead of PP."  # (b) 使用张量并行（--tp-size）替代PP
                )

        if self.pp_group.is_first_rank:  # 如果是第一个PP rank
            self.embed_tokens = Gemma4TextScaledWordEmbedding(  # 词嵌入层
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 嵌入维度
                self.padding_idx,  # 填充索引
                embed_scale=self.config.hidden_size**0.5,  # embedded normalizer  # 嵌入归一化缩放
            )
        else:  # 非第一个PP rank
            self.embed_tokens = PPMissingLayer()  # 使用PP缺失层占位

        if (  # 如果
            self.pp_group.is_first_rank  # 是第一个PP rank
            and self.hidden_size_per_layer_input  # 有每层输入隐藏大小
            and self.hidden_size_per_layer_input > 0  # 且大于0
        ):
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(  # 每层嵌入层
                self.vocab_size_per_layer_input,  # 每层词表大小
                config.num_hidden_layers * self.hidden_size_per_layer_input,  # 总每层嵌入维度
                self.padding_idx,  # 填充索引
                embed_scale=self.hidden_size_per_layer_input**0.5,  # 每层嵌入缩放
            )

            self.per_layer_model_projection = ReplicatedLinear(  # 每层模型投影
                self.hidden_size,  # 输入维度
                config.num_hidden_layers * self.hidden_size_per_layer_input,  # 输出维度
                bias=False,  # 无偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("per_layer_model_projection", prefix),  # 添加前缀
            )

            self.per_layer_projection_norm = RMSNorm(  # 每层投影归一化
                self.hidden_size_per_layer_input,  # 归一化维度
                config.rms_norm_eps,  # epsilon值
            )
            self.per_layer_input_scale = torch.rsqrt(torch.tensor(2.0))  # 每层输入缩放因子 1/sqrt(2)
            self.per_layer_projection_scale = torch.tensor(  # 每层投影缩放因子
                config.hidden_size**-0.5,  # hidden_size的负平方根
            )
        else:  # 无PLE
            self.embed_tokens_per_layer = None  # 每层嵌入为None
            self.per_layer_model_projection = None  # 每层模型投影为None
            self.per_layer_projection_norm = None  # 每层投影归一化为None
            self.per_layer_input_scale = None  # 每层输入缩放为None
            self.per_layer_projection_scale = None  # 每层投影缩放为None

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 隐藏层数
            lambda idx, prefix: Gemma4DecoderLayer(  # 每层是一个Gemma4DecoderLayer
                layer_id=idx,  # 层ID
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 前缀
            ),
            pp_rank=self.pp_group.rank_in_group,  # PP排名
            pp_size=self.pp_group.world_size,  # PP世界大小
            prefix=add_prefix("layers", prefix),  # 添加前缀
        )

        if self.pp_group.is_last_rank:  # 如果是最后一个PP rank
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化
        else:  # 非最后一个PP rank
            self.norm = PPMissingLayer()  # 使用PP缺失层占位
        self.layers_to_capture = []  # 要捕获的层列表
        self.post_init()  # 后初始化

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.embed_tokens  # 返回词嵌入层

    def dtype(self) -> torch.dtype:  # 获取数据类型
        return next(self.parameters()).dtype  # 返回第一个参数的数据类型

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:  # 获取每层输入
        if self.embed_tokens_per_layer is None:  # 如果无每层嵌入
            return None  # 返回None

        # Handle out-of-vocab tokens for PLE (vocab_size_per_layer_input may  # 处理PLE的超出词表token（vocab_size_per_layer_input可能
        # be smaller than the main vocab_size). Following Gemma3n pattern.  # 小于主vocab_size）。遵循Gemma3n模式
        per_layer_inputs_mask = torch.logical_and(  # 创建有效掩码
            input_ids >= 0,  # ID >= 0
            input_ids < self.vocab_size_per_layer_input,  # ID < 每层词表大小
        )
        per_layer_inputs_tokens = torch.where(  # 获取有效token
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)  # 无效位置填0
        )

        # Get packed per-layer embeddings: (num_tokens, total_ple_dim)  # 获取打包的每层嵌入：(token数, 总PLE维度)
        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)  # 通过每层嵌入层

        # Apply embed_scale (sqrt of per-layer hidden dim)  # 应用嵌入缩放（每层隐藏维度的平方根）
        # Already done in embedding layer  # 已在嵌入层中完成
        # per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer  # per_layer_embeds * 每层嵌入缩放

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)  # 重塑为(token数, 层数, 每层输入隐藏大小)
        per_layer_embeds = per_layer_embeds.reshape(  # 重塑
            *input_ids.shape,  # 保持输入形状
            self.config.num_hidden_layers,  # 层数
            self.hidden_size_per_layer_input,  # 每层输入隐藏大小
        )
        return per_layer_embeds  # 返回每层嵌入

    def project_per_layer_inputs(  # 投影每层输入方法
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入
        per_layer_inputs: Optional[torch.Tensor] = None,  # 每层输入（可选）
    ) -> torch.Tensor:  # 返回张量
        """Project inputs_embeds and combine with per_layer_inputs.  # 投影inputs_embeds并与per_layer_inputs合并

        Following HF/Gemma3n reference:  # 遵循HF/Gemma3n参考：
        1. Project inputs_embeds: hidden_size → total_ple_dim  # 1. 投影inputs_embeds：hidden_size → 总PLE维度
        2. Scale by hidden_size^{-0.5} (Gemma4ScaledLinear w_scale)  # 2. 乘以hidden_size^{-0.5}缩放（Gemma4ScaledLinear w_scale）
        3. Reshape to (num_tokens, num_layers, per_layer_dim)  # 3. 重塑为(token数, 层数, 每层维度)
        4. Normalize with per_layer_projection_norm  # 4. 用per_layer_projection_norm归一化
        5. Combine: (projection + per_layer_inputs) * 1/sqrt(2)  # 5. 合并：(投影 + 每层输入) * 1/sqrt(2)
        """
        if self.per_layer_model_projection is None:  # 如果无每层模型投影
            return None  # 返回None

        # Project from hidden_size to total_ple_dim  # 从hidden_size投影到总PLE维度
        per_layer_projection, _ = self.per_layer_model_projection(inputs_embeds)  # 通过投影层

        # Apply w_scale (HF: Gemma4ScaledLinear with w_scale=hidden_size^{-0.5})  # 应用w_scale（HF：Gemma4ScaledLinear的w_scale=hidden_size^{-0.5}）
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale  # 乘以投影缩放

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)  # 重塑为(token数, 层数, 每层输入隐藏大小)
        per_layer_projection = per_layer_projection.reshape(  # 重塑
            *inputs_embeds.shape[:-1],  # 保持除最后一维外的形状
            self.config.num_hidden_layers,  # 层数
            self.hidden_size_per_layer_input,  # 每层输入隐藏大小
        )

        # Normalize  # 归一化
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)  # 通过归一化层

        if per_layer_inputs is None:  # 如果无每层输入
            return per_layer_projection  # 仅返回投影

        # Combine: (projection + per_layer_inputs) * scale  # 合并：(投影 + 每层输入) * 缩放
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale  # 返回合并结果

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        per_layer_inputs: Optional[torch.Tensor] = None,  # 每层输入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量（可选）
        **kwargs,  # 其他关键字参数
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:  # 返回类型
        if self.pp_group.is_first_rank:  # 如果是第一个PP rank
            if (input_ids is None) ^ (input_embeds is not None):  # 如果恰好有一个为None
                raise ValueError(  # 抛出值错误
                    "You must specify exactly one of input_ids or inputs_embeds"  # 必须指定input_ids或inputs_embeds中的一个
                )

            if input_ids is not None:  # 如果提供了输入ID
                input_embeds = self.embed_tokens(input_ids)  # 获取词嵌入
                per_layer_inputs = self.get_per_layer_inputs(input_ids)  # 获取每层输入
            per_layer_inputs = self.project_per_layer_inputs(  # 投影每层输入
                input_embeds, per_layer_inputs  # 输入嵌入和每层输入
            )
            hidden_states = input_embeds  # 隐藏状态初始化为输入嵌入
        else:  # 非第一个PP rank
            assert (  # 断言
                pp_proxy_tensors is not None  # PP代理张量不为None
            ), "pp_proxy_tensors is required on non-first PP ranks"  # 非第一PP rank需要pp_proxy_tensors
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理获取隐藏状态
            # PLE inputs were computed on rank 0 and forwarded along the  # PLE输入在rank 0上计算并沿
            # pipeline; non-PLE models simply omit the key.  # 流水线转发；非PLE模型简单地省略该键
            per_layer_inputs = pp_proxy_tensors.tensors.get("per_layer_inputs", None)  # 获取每层输入

        aux_hidden_states = []  # 辅助隐藏状态列表
        num_layers = self.config.num_hidden_layers  # 总层数

        for layer_idx in range(self.start_layer, self.end_layer):  # 遍历本rank负责的层
            if layer_idx in self.layers_to_capture:  # 如果需要捕获该层
                aux_hidden_states.append(hidden_states)  # 添加到辅助隐藏状态列表

            if per_layer_inputs is not None:  # 如果有每层输入
                per_layer_input = per_layer_inputs[:, layer_idx, :]  # 获取当前层的每层输入
            else:  # 无每层输入
                per_layer_input = None  # 设为None
            layer = self.layers[layer_idx]  # 获取当前层
            layer_outputs = layer(  # 通过当前层
                positions=positions,  # 位置编码
                hidden_states=hidden_states,  # 隐藏状态
                per_layer_input=per_layer_input,  # 每层输入
                forward_batch=forward_batch,  # 前向批次
                **kwargs,  # 其他关键字参数
            )
            hidden_states = layer_outputs[0]  # 更新隐藏状态
            # Gemma4DecoderLayer.forward always returns (hidden_states, None);  # Gemma4DecoderLayer.forward始终返回(hidden_states, None)；
            # the residual is fused inside the layer, so nothing to thread.  # 残差在层内融合，无需传递

        if not self.pp_group.is_last_rank:  # 如果不是最后一个PP rank
            # cuda_graph_runner allocates a fixed PP-proxy schema of  # cuda_graph_runner分配固定的PP代理模式
            # {hidden_states, residual} and KeyErrors if a model omits a key.  # {hidden_states, residual}，如果模型省略键则KeyError
            # Gemma4 fuses the residual inside each layer so we don't have a  # Gemma4在每层内融合残差，因此我们没有
            # standalone tensor to forward; emit a zero placeholder instead so  # 独立的张量可转发；改为发射零占位符，
            # graph replay can still copy it.  The receiving stage never reads  # 以便图重放仍可复制它。接收阶段从不读取
            # this key.  # 此键
            proxy = {  # 代理字典
                "hidden_states": hidden_states,  # 隐藏状态
                "residual": torch.zeros_like(hidden_states),  # 零占位残差
            }
            if per_layer_inputs is not None:  # 如果有每层输入
                proxy["per_layer_inputs"] = per_layer_inputs  # 添加到代理
            return PPProxyTensors(proxy)  # 返回PP代理张量

        # Capture the output of the last layer if requested.  # 如果请求，捕获最后一层的输出
        # layers_to_capture uses +1 offset, so num_layers means  # layers_to_capture使用+1偏移，因此num_layers表示
        # "output of the last layer" which is only available after the loop.  # "最后一层的输出"，仅在循环后可用
        if num_layers in self.layers_to_capture:  # 如果需要捕获最后一层
            aux_hidden_states.append(hidden_states)  # 添加到辅助隐藏状态列表

        hidden_states = self.norm(hidden_states)  # 最终归一化

        if len(aux_hidden_states) == 0:  # 如果无辅助隐藏状态
            return hidden_states  # 仅返回隐藏状态

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态


class Gemma4ForCausalLM(PreTrainedModel):  # Gemma4因果语言模型类
    config_class = Gemma4TextConfig  # 配置类
    base_model_prefix = "language_model"  # 基础模型前缀
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 绑定权重键
    _tp_plan = {"lm_head": "colwise_rep"}  # TP计划

    # BitandBytes specific attributes  # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块
        ".gate_proj.",  # 门投影
        ".down_proj.",  # 下投影
        ".up_proj.",  # 上投影
        ".q_proj.",  # Q投影
        ".k_proj.",  # K投影
        ".v_proj.",  # V投影
        ".o_proj.",  # O投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射
        # shard_name, weight_name, index  # 分片名，权重名，索引
        "q_proj": ("qkv_proj", 0),  # Q投影映射
        "k_proj": ("qkv_proj", 1),  # K投影映射
        "v_proj": ("qkv_proj", 2),  # V投影映射
        "gate_proj": ("gate_up_proj", 0),  # 门投影映射
        "up_proj": ("gate_up_proj", 1),  # 上投影映射
    }

    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [  # QKV投影
            "q_proj",  # Q投影
            "k_proj",  # K投影
            "v_proj",  # V投影
        ],
        "gate_up_proj": [  # gate_up投影
            "gate_proj",  # 门投影
            "up_proj",  # 上投影
        ],
    }

    # Gemma does not apply LoRA to the embedding layer.  # Gemma不在嵌入层上应用LoRA
    embedding_modules = {}  # 嵌入模块为空
    embedding_padding_modules = []  # 嵌入填充模块为空
    supports_lora = False  # 不支持LoRA

    def __init__(  # 初始化方法
        self,
        config: Gemma4TextConfig,  # Gemma4文本配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置（可选）
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__(config=config)  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取PP组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = Gemma4TextModel(  # 文本模型
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 配置、量化配置和前缀
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

        # tie_word_embeddings ties lm_head to embed_tokens, but with PP those  # tie_word_embeddings将lm_head绑定到embed_tokens，但在PP下这些
        # tensors live on opposite ranks (first vs last).  In the PP > 1 case  # 张量位于相反的rank上（第一个vs最后一个）。在PP>1的情况下
        # we materialize a real ParallelLMHead on the last rank and route the  # 我们在最后一个rank上实例化真正的ParallelLMHead，并在
        # checkpoint's embed_tokens.weight into it during load_weights.  # load_weights期间将检查点的embed_tokens.weight路由到其中
        if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:  # PP=1且绑定词嵌入
            self.lm_head = self.model.embed_tokens  # lm_head共享嵌入层
        elif self.pp_group.is_last_rank:  # 是最后一个PP rank
            self.lm_head = ParallelLMHead(  # 并行语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏大小
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 添加前缀
            )
        else:  # 其他rank
            self.lm_head = PPMissingLayer()  # 使用PP缺失层占位

        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态
        self.post_init()  # 后初始化

    def tie_weights(self, *args, **kwargs):  # 绑定权重方法
        # HF's PreTrainedModel.tie_weights uses ``_tied_weights_keys`` to bind  # HF的PreTrainedModel.tie_weights使用_tied_weights_keys来绑定
        # ``lm_head.weight`` to ``model.embed_tokens.weight``.  Under PP those  # lm_head.weight到model.embed_tokens.weight。在PP下这些
        # tensors live on different ranks (embed on first, head on last) and  # 张量位于不同rank上（嵌入在第一个，头在最后一个）
        # the missing side is a PPMissingLayer with no ``weight`` attribute,  # 缺失的一侧是PPMissingLayer，没有weight属性
        # which makes the default tie_weights crash.  load_weights routes the  # 这会使默认的tie_weights崩溃。load_weights将
        # checkpoint embedding into lm_head explicitly, so the tie is a no-op  # 检查点嵌入显式路由到lm_head，因此绑定是空操作
        # here when PP is active.  # 在PP激活时
        if self.pp_group.world_size > 1:  # 如果PP>1
            return  # 直接返回
        super().tie_weights(*args, **kwargs)  # 调用父类绑定权重

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.model.embed_tokens  # 返回模型嵌入层

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:  # 获取嵌入和头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和头权重

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小
        return get_attention_sliding_window_size(self.config)  # 调用模块级函数

    def dtype(self) -> torch.dtype:  # 获取数据类型
        return next(self.parameters()).dtype  # 返回第一个参数的数据类型

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        per_layer_inputs: Optional[torch.Tensor] = None,  # 每层输入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量（可选）
        **kwargs,  # 其他关键字参数
    ) -> Union[LogitsProcessor, PPProxyTensors]:  # 返回类型
        hidden_states = self.model(  # 通过文本模型
            input_ids,  # 输入ID
            positions,  # 位置编码
            forward_batch,  # 前向批次
            input_embeds,  # 输入嵌入
            per_layer_inputs,  # 每层输入
            pp_proxy_tensors=pp_proxy_tensors,  # PP代理张量
            **kwargs,  # 其他关键字参数
        )

        if not self.pp_group.is_last_rank:  # 如果不是最后一个PP rank
            # `hidden_states` here is actually a PPProxyTensors handed off to  # 这里的hidden_states实际上是传递给
            # the next stage; logits processing only happens on the last rank.  # 下一阶段的PPProxyTensors；logits处理仅在最后一个rank上进行
            return hidden_states  # 返回代理张量

        aux_hidden_states = None  # 辅助隐藏状态
        if self.capture_aux_hidden_states:  # 如果捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包

        return self.logits_processor(  # 通过logits处理器
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states  # 输入ID，隐藏状态，头，批次，辅助状态
        )

    def _get_k_eq_v_layers(self) -> set:  # 获取K等于V的层集合
        """Return set of layer indices where attention_k_eq_v applies (full-attention layers)."""  # 返回attention_k_eq_v适用的层索引集合（全注意力层）
        if not getattr(self.config, "attention_k_eq_v", False):  # 如果未启用attention_k_eq_v
            return set()  # 返回空集合
        return {  # 返回
            i for i, lt in enumerate(self.config.layer_types) if lt == "full_attention"  # 所有全注意力层的索引
        }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q投影映射
            ("qkv_proj", "k_proj", "k"),  # K投影映射
            ("qkv_proj", "v_proj", "v"),  # V投影映射
            ("gate_up_proj", "gate_proj", 0),  # 门投影映射
            ("gate_up_proj", "up_proj", 1),  # 上投影映射
        ]

        fused_expert_params_mapping = [  # 融合专家参数映射
            # (param_name, ckpt_weight_name, shard_ids)  # (参数名, 检查点权重名, 分片ID)
            # gate_up_proj is fused [E, 2*I, H] — chunk into w1 (gate) + w3 (up)  # gate_up_proj是融合的[E, 2*I, H]——分块为w1(门) + w3(上)
            ("experts.w13_weight", "experts.gate_up_proj", ("w1", "w3")),  # w13权重映射
            ("experts.w2_weight", "experts.down_proj", ("w2",)),  # w2权重映射
        ]
        # Dense subclasses (e.g. the Gemma4 MTP assistant) reuse this.  # 稠密子类（如Gemma4 MTP助手）复用此映射
        num_experts = getattr(self.config, "num_experts", None) or 0  # 专家数量

        # Per-expert checkpoint format used by compressed-tensors / FP8  # compressed-tensors / FP8使用的每专家检查点格式
        # (e.g. RedHatAI/*-FP8-Dynamic) and by ModelOpt NVFP4  # （例如RedHatAI/*-FP8-Dynamic）和ModelOpt NVFP4
        # (e.g. nvidia/Gemma-4-*-NVFP4). Each expert is stored as a  # （例如nvidia/Gemma-4-*-NVFP4）。每个专家存储为
        # separate key with shape (out, in):  # 形状为(out, in)的单独键：
        #   experts.<id>.{gate,up,down}_proj.{weight,weight_scale,  #   experts.<id>.{gate,up,down}_proj.{weight,weight_scale,
        #                                     weight_scale_2,input_scale}  #                                     weight_scale_2,input_scale}
        # `make_expert_params_mapping` emits tuples whose `weight_name` ends  # `make_expert_params_mapping`发出weight_name以
        # in a trailing dot, so the standard `name.replace(weight_name,  # 尾部点结尾的元组，因此标准的name.replace(weight_name,
        # param_name)` collapses every suffix uniformly to the fused  # param_name)将每个后缀统一折叠到融合的
        # FusedMoE params (experts.w13_*, experts.w2_*).  # FusedMoE参数（experts.w13_*, experts.w2_*）
        per_expert_params_mapping = (  # 每专家参数映射
            FusedMoE.make_expert_params_mapping(  # 创建每专家参数映射
                ckpt_gate_proj_name="gate_proj",  # 检查点门投影名
                ckpt_down_proj_name="down_proj",  # 检查点下投影名
                ckpt_up_proj_name="up_proj",  # 检查点上投影名
                num_experts=num_experts,  # 专家数量
            )
            if num_experts  # 如果有专家
            else []  # 否则为空列表
        )

        k_eq_v_layers = self._get_k_eq_v_layers()  # 获取K=V的层集合

        params_dict = dict(self.named_parameters())  # 获取参数字典
        params_dict.update(dict(self.named_buffers()))  # 更新缓冲区字典
        non_persistent_buffers: Set[str] = set()  # 非持久化缓冲区集合
        for mod_name, mod in self.named_modules():  # 遍历所有模块
            for buf_name in getattr(mod, "_non_persistent_buffers_set", set()):  # 遍历非持久化缓冲区
                full = f"{mod_name}.{buf_name}" if mod_name else buf_name  # 完整名称
                non_persistent_buffers.add(full)  # 添加到集合

        loaded_params: Set[str] = set()  # 已加载参数集合
        for name, loaded_weight in weights:  # 遍历权重
            name = name.replace("model.language_model.", "model.")  # 替换名称前缀

            # HF has router.per_expert_scale and experts.* on the decoder layer;  # HF在解码器层上有router.per_expert_scale和experts.*
            # remap into our moe.* subtree since Gemma4MoE owns both.  # 重映射到我们的moe.*子树，因为Gemma4MoE拥有两者
            name = name.replace(".router.per_expert_scale", ".moe.per_expert_scale")  # 重映射路由器缩放
            if ".experts." in name and ".moe.experts." not in name:  # 如果有experts但不在moe下
                name = name.replace(".experts.", ".moe.experts.")  # 重映射到moe子树

            if pp_filter_load_weight(  # PP过滤
                name,  # 参数名
                loaded_weight,  # 权重
                pp_group=self.pp_group,  # PP组
                start_layer=self.model.start_layer,  # 起始层
                end_layer=self.model.end_layer,  # 结束层
                params_dict=params_dict,  # 参数字典
                loaded_params=loaded_params,  # 已加载参数
                tie_word_embeddings=self.config.tie_word_embeddings,  # 是否绑定词嵌入
                embed_weight_name="model.embed_tokens.weight",  # 嵌入权重名称
                first_rank_only_patterns=(  # 仅第一个rank的模式
                    "embed_tokens",  # 嵌入层
                    "per_layer_model_projection",  # 每层模型投影
                    "per_layer_projection_norm",  # 每层投影归一化
                ),
                last_rank_only_prefixes=("model.norm.", "lm_head."),  # 仅最后一个rank的前缀
            ):
                continue  # 跳过

            # attention_k_eq_v: full-attention layers have no v_proj in the  # attention_k_eq_v：全注意力层在检查点中没有v_proj
            # checkpoint (K and V share weights).  When we see a k_proj weight  # （K和V共享权重）。当我们看到k_proj权重
            # for one of these layers, load it into both the "k" and "v" shards  # 对于这些层之一，将其加载到融合QKV的"k"和"v"分片中
            # of the fused QKV so the forward produces v_raw == k_raw.  # 以便前向传播产生v_raw == k_raw
            should_dup_k_to_v = (  # 是否应将K复制到V
                ".k_proj." in name  # 名称包含k_proj
                and k_eq_v_layers  # 有K=V层
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None  # 提取层索引
                and int(m.group(1)) in k_eq_v_layers  # 层索引在K=V集合中
            )

            # MoE expert weights checked first (gate_up_proj contains "up_proj"  # 首先检查MoE专家权重（gate_up_proj包含"up_proj"
            # which would false-match the stacked dense MLP mapping).  # 会误匹配堆叠稠密MLP映射）
            orig_name = name  # 保存原始名称

            # 1) Per-expert checkpoint layout (compressed-tensors FP8 like  # 1) 每专家检查点布局（如compressed-tensors FP8
            #    RedHatAI/*-FP8-Dynamic, ModelOpt NVFP4 like  #    RedHatAI/*-FP8-Dynamic，如ModelOpt NVFP4
            #    nvidia/Gemma-4-*-NVFP4): experts.<id>.{gate,up,down}_proj.*  #    nvidia/Gemma-4-*-NVFP4）：experts.<id>.{gate,up,down}_proj.*
            #    The trailing dot in `weight_name` lets a single mapping fold  # weight_name中的尾部点让单个映射可以折叠
            #    weight, weight_scale, weight_scale_2, and input_scale into  # weight, weight_scale, weight_scale_2和input_scale到
            #    their corresponding fused FusedMoE params (experts.w13_*,  # 对应的融合FusedMoE参数（experts.w13_*，
            #    experts.w2_*).  #    experts.w2_*）
            for (  # 遍历每专家参数映射
                param_name,  # 参数名
                weight_name,  # 权重名
                expert_id,  # 专家ID
                shard_id,  # 分片ID
            ) in per_expert_params_mapping:
                if weight_name not in orig_name:  # 如果权重名不在原始名称中
                    continue  # 跳过
                name = orig_name.replace(weight_name, param_name)  # 替换权重名
                if name not in params_dict:  # 如果名称不在参数字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 参数
                    loaded_weight,  # 权重
                    name,  # 参数名
                    shard_id=shard_id,  # 分片ID
                    expert_id=expert_id,  # 专家ID
                )
                loaded_params.add(name)  # 添加到已加载集合
                break  # 跳出内层循环
            else:  # 如果没有匹配每专家映射
                # 2) BF16 fused checkpoint layout: experts.gate_up_proj is a  # 2) BF16融合检查点布局：experts.gate_up_proj是一个
                #    [E, 2*I, H] tensor that needs per-expert chunking into  #    [E, 2*I, H]张量，需要按专家分块为
                #    w1 (gate) and w3 (up).  #    w1(门)和w3(上)
                for param_name, weight_name, shard_ids in fused_expert_params_mapping:  # 遍历融合专家映射
                    name = orig_name  # 恢复原始名称
                    if weight_name not in name:  # 如果权重名不在名称中
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    if name not in params_dict:  # 如果名称不在参数字典中
                        continue  # 跳过
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    for i in range(num_experts):  # 遍历每个专家
                        chunks = loaded_weight[i].chunk(len(shard_ids), dim=0)  # 分块权重
                        for chunk, sid in zip(chunks, shard_ids):  # 遍历分块和分片ID
                            weight_loader(param, chunk, name, sid, i)  # 加载权重
                    loaded_params.add(name)  # 添加到已加载集合
                    break  # 跳出内层循环
                else:  # 如果没有匹配融合专家映射
                    for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                        name = orig_name  # 恢复原始名称
                        if weight_name not in name:  # 如果权重名不在名称中
                            continue  # 跳过
                        name = name.replace(weight_name, param_name)  # 替换权重名
                        if name not in params_dict:  # 如果名称不在参数字典中
                            continue  # 跳过
                        param = params_dict[name]  # 获取参数
                        weight_loader = param.weight_loader  # 获取权重加载器
                        weight_loader(param, loaded_weight, shard_id)  # 加载权重
                        if should_dup_k_to_v:  # 如果应将K复制到V
                            weight_loader(param, loaded_weight, "v")  # 加载K权重到V分片
                        loaded_params.add(name)  # 添加到已加载集合
                        break  # 跳出内层循环
                    else:  # 如果没有匹配任何映射
                        name = orig_name  # 恢复原始名称
                        if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                            continue  # 跳过
                        name = maybe_remap_kv_scale_name(name, params_dict)  # 可能重映射KV缩放名称
                        if name is None:  # 如果名称为None
                            continue  # 跳过
                        if name not in params_dict:  # 如果名称不在参数字典中
                            continue  # 跳过
                        param = params_dict[name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader  # 默认使用default_weight_loader
                        )
                        weight_loader(param, loaded_weight)  # 加载权重
                        loaded_params.add(name)  # 添加到已加载集合
        unloaded_params = params_dict.keys() - loaded_params  # 未加载参数
        if unloaded_params:  # 如果有未加载参数
            param_names = set(dict(self.named_parameters()).keys())  # 参数名集合
            buckets = {  # 日志级别分桶
                logging.WARNING: (  # 警告级别
                    "Some weights are not initialized from checkpoints",  # 某些权重未从检查点初始化
                    lambda p: p in param_names,  # 过滤条件：是参数
                ),
                logging.INFO: (  # 信息级别
                    "Persistent buffers not in checkpoint (using default init)",  # 持久化缓冲区不在检查点中（使用默认初始化）
                    lambda p: p not in param_names and p not in non_persistent_buffers,  # 过滤条件：不是参数且不是非持久化缓冲区
                ),
                logging.DEBUG: (  # 调试级别
                    "Non-persistent buffers not in checkpoint (expected)",  # 非持久化缓冲区不在检查点中（预期行为）
                    lambda p: p in non_persistent_buffers,  # 过滤条件：是非持久化缓冲区
                ),
            }
            for level, (msg, pred) in buckets.items():  # 遍历日志级别
                names = sorted(p for p in unloaded_params if pred(p))  # 过滤并排序
                if names:  # 如果有名称
                    logger.log(level, "%s: %s", msg, names)  # 记录日志
        return loaded_params  # 返回已加载参数集合

    def _shard_weight(self, weight: torch.Tensor) -> torch.Tensor:  # 分片权重方法
        """Shard a full embedding/lm_head weight along vocab dim for the current TP rank.  # 沿词表维度为当前TP排名分片完整的嵌入/lm_head权重

        Gemma4 uses nn.Embedding (unsharded) but the Eagle3 draft model uses  # Gemma4使用nn.Embedding（不分片）但Eagle3草稿模型使用
        VocabParallelEmbedding (sharded). This method extracts the correct  # VocabParallelEmbedding（分片）。此方法提取正确的
        shard so the weights can be shared.  # 分片以便权重可以共享
        """
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP世界大小
        if tp_size <= 1:  # 如果TP大小<=1
            return weight  # 不分片
        tp_rank = get_tensor_model_parallel_rank()  # 获取TP排名
        shard_size = (weight.shape[0] + tp_size - 1) // tp_size  # 分片大小
        return weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]  # 返回当前排名的分片

    def get_embed(self):  # 获取嵌入权重
        return self._shard_weight(self.model.embed_tokens.weight)  # 返回分片后的嵌入权重

    def get_embed_and_head(self):  # 获取嵌入和头权重
        if self.pp_group.world_size > 1:  # 如果PP>1
            # Under PP, embed_tokens lives on the first rank and lm_head on  # 在PP下，embed_tokens在第一个rank上，lm_head在
            # the last; neither rank holds both tensors, so we can't return  # 最后一个上；两个rank都不持有两个张量，因此无法
            # the pair locally without a cross-stage gather.  Callers (RL  # 在本地返回该对，除非跨阶段收集。调用者（RL
            # weight sync, remote weight loader) currently assume a  # 权重同步，远程权重加载器）当前假设
            # single-rank view — fail loudly rather than dereference a  # 单rank视图——大声失败而不是解引用
            # PPMissingLayer.  # PPMissingLayer
            raise NotImplementedError(  # 抛出未实现错误
                "get_embed_and_head() is not implemented for Gemma4ForCausalLM "  # Gemma4ForCausalLM未实现get_embed_and_head()
                "under pipeline parallelism. embed_tokens lives on the first "  # 在流水线并行下。embed_tokens在第一个
                "PP rank and lm_head on the last; use --pp-size 1 if you "  # PP rank上，lm_head在最后一个上；如果需要此API
                "need this API."  # 使用--pp-size 1
            )
        embed = self._shard_weight(self.model.embed_tokens.weight)  # 分片嵌入权重
        head = self._shard_weight(self.lm_head.weight)  # 分片头权重
        return embed, head  # 返回嵌入和头权重

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):  # 设置Eagle3要捕获的层
        if layer_ids is None:  # 如果未指定层ID
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            num_layers = self.config.num_hidden_layers  # 总层数
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]  # 默认捕获第2、中间和倒数第3层
        else:  # 指定了层ID
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            # we plus 1 here because in sglang, for the ith layer, it takes the output  # 这里加1因为在sglang中，第i层取
            # of the (i-1)th layer as aux hidden state  # 第(i-1)层的输出作为辅助隐藏状态
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 加1偏移


EntryClass = Gemma4ForCausalLM  # 入口类为Gemma4ForCausalLM
