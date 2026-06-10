# HunYuan V3模型推理实现 - 基于稀疏专家混合架构的混元V3模型，支持共享专家和双流并行，仅用于推理
# coding=utf-8  # 编码格式
# Copyright 2026 The HunYuan team.  # 版权所有 2026 混元团队
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

from typing import Iterable, Optional, Tuple  # 类型提示导入

import torch  # PyTorch深度学习框架
from torch import nn  # 神经网络模块
from transformers import PretrainedConfig  # 预训练配置基类

from sglang.srt.distributed import (  # 分布式工具导入
    get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
    get_moe_tensor_parallel_world_size,  # 获取MoE张量并行世界大小
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    moe_expert_parallel_all_reduce,  # MoE专家并行全归约
    moe_tensor_model_parallel_all_reduce,  # MoE张量并行全归约
)
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活函数与乘法
from sglang.srt.layers.layernorm import RMSNorm  # RMS归一化层
from sglang.srt.layers.linear import (  # 线性层导入
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce  # 是否跳过专家后全归约
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 融合MoE Triton层
from sglang.srt.layers.moe.topk import TopK  # Top-K选择模块
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力机制
from sglang.srt.layers.rotary_embedding import get_rope  # 获取旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.managers.schedule_batch import ForwardBatch  # 前向批次信息
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 获取是否处于CUDA图捕获模式
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.utils import is_cuda  # 是否为CUDA
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 获取RoPE配置


class HYV3FeedForward(nn.Module):  # 混元V3前馈网络模块
    def __init__(  # 初始化前馈网络
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 创建gate和up合并投影层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小列表（gate和up各一个）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.gate_up_proj",  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # 创建down投影层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            reduce_results=reduce_results,  # 是否归约结果
            prefix=f"{prefix}.down_proj",  # 参数前缀
        )
        if hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出错误
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."  # 目前只支持silu
            )
        self.act_fn = SiluAndMul()  # 创建SiLU激活与乘法函数

    def forward(self, x):  # 前馈网络前向传播
        gate_up, _ = self.gate_up_proj(x)  # 计算gate和up投影
        out = self.act_fn(gate_up)  # 应用激活函数
        out, _ = self.down_proj(out)  # 计算down投影
        return out  # 返回结果


class HYV3MoEFused(nn.Module):  # 混元V3融合MoE模块
    def __init__(  # 初始化融合MoE
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_moe_tensor_parallel_world_size()  # 获取MoE张量并行大小
        self.ep_size = get_moe_expert_parallel_world_size()  # 获取MoE专家并行大小
        self.layer_id = layer_id  # 保存层ID
        self.alt_stream = alt_stream  # 保存备用CUDA流
        self.n_routed_experts = config.num_experts  # 路由专家数量
        top_k = config.num_experts_per_tok  # 每个token的top-k专家数
        intermediate_size = config.moe_intermediate_size  # MoE中间层大小

        self.expert_bias = nn.Parameter(  # 创建专家偏置参数
            torch.empty(config.num_experts, dtype=torch.float32)  # 浮点32位专家偏置
        )
        self.expert_bias.weight_loader = HYV3MoEFused.ebias_weight_loader  # 设置权重加载器
        scoring_func = "sigmoid"  # 评分函数使用sigmoid
        self.e_score_correction_bias = self.expert_bias  # 评分校正偏置
        self.router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)  # 路由器缩放因子
        self.gate = ReplicatedLinear(  # 创建门控线性层（路由器）
            config.hidden_size,  # 输入大小
            config.num_experts,  # 输出大小（专家数量）
            bias=False,  # 不使用偏置
            quant_config=None,  # 门控不使用量化
            params_dtype=torch.float32,  # 使用float32精度
            prefix=f"{prefix}.gate",  # 参数前缀
        )
        self.topk = TopK(  # 创建Top-K选择模块
            top_k=config.num_experts_per_tok,  # top-k值
            use_grouped_topk=True,  # 使用分组top-k
            num_expert_group=1,  # 专家分组数
            topk_group=1,  # 每组top-k
            renormalize=config.route_norm,  # 是否重新归一化
            scoring_func=scoring_func,  # 评分函数
            correction_bias=self.e_score_correction_bias,  # 校正偏置
            routed_scaling_factor=self.router_scaling_factor,  # 路由缩放因子
            apply_routed_scaling_factor_on_output=True,  # 对输出应用路由缩放因子
        )

        if getattr(config, "num_shared_experts", 0) > 0:  # 如果有共享专家
            self.shared_mlp = HYV3FeedForward(  # 创建共享MLP
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,  # 共享MLP中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.shared_mlp",  # 参数前缀
                reduce_results=False,  # 不归约结果
            )
        else:  # 否则没有共享专家
            self.shared_mlp = None  # 设为None

        self.experts = FusedMoE(  # 创建融合MoE专家层
            num_experts=self.n_routed_experts,  # 专家数量
            top_k=top_k,  # top-k值
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=intermediate_size,  # 中间层大小
            reduce_results=False,  # 不归约结果
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.experts",  # 参数前缀
        )

    @staticmethod  # 静态方法
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:  # 专家偏置权重加载器
        assert param.size() == loaded_weight.size()  # 断言大小匹配
        param.data.copy_(loaded_weight.to(torch.float32))  # 拷贝权重数据（转为float32）

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 融合MoE前向传播
        if (  # 如果满足双流并行条件
            self.alt_stream is not None  # 有备用CUDA流
            and self.shared_mlp is not None  # 有共享MLP
            and hidden_states.shape[0] > 0  # 有输入数据
            and get_is_capture_mode()  # 处于CUDA图捕获模式
        ):
            return self._forward_dual_stream(hidden_states)  # 使用双流并行前向传播
        return self._forward_single_stream(hidden_states)  # 否则使用单流前向传播

    def _forward_single_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 单流前向传播
        orig_shape = hidden_states.shape  # 保存原始形状
        hidden_dim = hidden_states.shape[-1]  # 获取隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑为2D

        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))  # 计算路由器logits
        topk_output = self.topk(hidden_states, router_logits)  # 获取top-k选择结果
        if self.shared_mlp is not None:  # 如果有共享MLP
            shared_output = self.shared_mlp(hidden_states)  # 计算共享MLP输出
            final_hidden_states = self.experts(  # 执行路由专家计算
                hidden_states=hidden_states, topk_output=topk_output
            )
            final_hidden_states = final_hidden_states + shared_output  # 路由专家+共享MLP
        else:  # 否则没有共享MLP
            final_hidden_states = self.experts(  # 仅执行路由专家计算
                hidden_states=hidden_states, topk_output=topk_output
            )

        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(  # 如果需要专家并行全归约
            is_tp_path=False,
        ):
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)  # 执行专家并行全归约

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 如果需要张量并行全归约
            is_tp_path=True,
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(  # 执行MoE张量并行全归约
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)  # 恢复原始形状并返回

    def _forward_dual_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 双流并行前向传播
        """Shared experts on main stream, routed experts on alt stream."""  # 共享专家在主流，路由专家在备用流
        orig_shape = hidden_states.shape  # 保存原始形状
        hidden_dim = hidden_states.shape[-1]  # 获取隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑为2D

        current_stream = torch.cuda.current_stream()  # 获取当前CUDA流
        self.alt_stream.wait_stream(current_stream)  # 备用流等待当前流完成

        shared_output = self.shared_mlp(hidden_states)  # 在主流上计算共享MLP

        with torch.cuda.stream(self.alt_stream):  # 在备用流上执行
            router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))  # 计算路由器logits
            topk_output = self.topk(hidden_states, router_logits)  # 获取top-k选择结果
            final_hidden_states = self.experts(  # 执行路由专家计算
                hidden_states=hidden_states, topk_output=topk_output
            )

        current_stream.wait_stream(self.alt_stream)  # 当前流等待备用流完成
        final_hidden_states = final_hidden_states + shared_output  # 合并路由专家和共享MLP结果

        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(  # 如果需要专家并行全归约
            is_tp_path=False,
        ):
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)  # 执行专家并行全归约

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 如果需要张量并行全归约
            is_tp_path=True,
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(  # 执行MoE张量并行全归约
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)  # 恢复原始形状并返回


class HYV3Attention(nn.Module):  # 混元V3注意力模块
    def __init__(  # 初始化注意力
        self,
        config: PretrainedConfig,  # 模型配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数量
        num_kv_heads: int,  # KV头数量
        layer_id: int = 0,  # 层ID
        rope_theta: float = 10000,  # RoPE基数
        rope_scaling: Optional[dict] = None,  # RoPE缩放配置
        max_position_embeddings: int = 8192,  # 最大位置嵌入数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个rank的注意力头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于TP大小
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则KV头数小于TP大小
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个rank的KV头数

        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.use_qk_norm = getattr(  # 是否使用QK归一化
            config, "use_qk_norm", getattr(config, "qk_norm", False)
        )

        self.qkv_proj = QKVParallelLinear(  # 创建QKV并行投影层
            hidden_size,  # 输入大小
            self.head_dim,  # 每个头的大小
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.qkv_proj",  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 创建输出投影层
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.o_proj",  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 创建旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # RoPE基数
            rope_scaling=rope_scaling,  # RoPE缩放配置
            is_neox_style=True,  # 使用Neox风格
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
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)  # 获取RMS归一化epsilon
            self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)  # Q归一化层
            self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)  # K归一化层

    def forward(  # 注意力前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV

        if self.use_qk_norm:  # 如果使用QK归一化
            q = self.q_norm(q.reshape(-1, self.head_dim))  # 对Q进行RMS归一化
            q = q.view(-1, self.q_size)  # 重塑Q形状
            k = self.k_norm(k.reshape(-1, self.head_dim))  # 对K进行RMS归一化
            k = k.view(-1, self.kv_size)  # 重塑K形状

        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出


class HYV3DecoderLayer(nn.Module):  # 混元V3解码器层
    def __init__(  # 初始化解码器层
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层ID
        self.hidden_size = config.hidden_size  # 隐藏层大小
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入数
        rope_theta, _ = get_rope_config(config)  # 获取RoPE配置
        self.self_attn = HYV3Attention(  # 创建自注意力层
            config=config,  # 配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE基数
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入数
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.self_attn",  # 参数前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)  # 注意力后归一化

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)  # 前k层使用Dense MLP
        if layer_id < first_k_dense_replace:  # 如果当前层在前k层内
            self.mlp = HYV3FeedForward(  # 创建Dense前馈网络
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=config.intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.mlp",  # 参数前缀
            )
            self.block_type = "feedforward"  # 块类型为前馈
        else:  # 否则使用MoE
            self.mlp = HYV3MoEFused(  # 创建融合MoE
                config=config,  # 配置
                layer_id=layer_id,  # 层ID
                quant_config=quant_config,  # 量化配置
                prefix=f"{prefix}.mlp",  # 参数前缀
                alt_stream=alt_stream,  # 备用CUDA流
            )
            self.block_type = "moe"  # 块类型为MoE

    def forward(  # 解码器层前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:  # 如果没有残差
            residual = hidden_states  # 当前隐藏状态作为残差
            hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        else:  # 否则有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 归一化（带残差）
        hidden_states = self.self_attn(  # 执行自注意力计算
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后归一化
        hidden_states = self.mlp(hidden_states)  # 执行MLP/MoE计算

        return hidden_states, residual  # 返回隐藏状态和残差


class HYV3Model(nn.Module):  # 混元V3模型主体
    def __init__(  # 初始化模型
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.embed_tokens = VocabParallelEmbedding(  # 创建词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            prefix=f"{prefix}.embed_tokens",  # 参数前缀
        )

        self.alt_stream = torch.cuda.Stream() if is_cuda() else None  # 如果是CUDA则创建备用流

        self.layers = nn.ModuleList(  # 创建解码器层列表
            [
                HYV3DecoderLayer(  # 每一层都是混元V3解码器层
                    config=config,  # 配置
                    layer_id=i,  # 层ID
                    quant_config=quant_config,  # 量化配置
                    prefix=f"{prefix}.layers.{i}",  # 参数前缀
                    alt_stream=self.alt_stream,  # 备用CUDA流
                )
                for i in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)  # 最终归一化层

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 模型前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 从token ID获取嵌入
        else:  # 否则
            hidden_states = input_embeds  # 直接使用输入嵌入
        residual = None  # 残差初始化为None
        for layer in self.layers:  # 遍历所有解码器层
            hidden_states, residual = layer(  # 执行当前层前向传播
                positions, hidden_states, forward_batch, residual
            )

        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class HYV3ForCausalLM(nn.Module):  # 混元V3因果语言模型
    def __init__(  # 初始化因果语言模型
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = HYV3Model(config, quant_config, prefix=f"{prefix}.model")  # 创建模型主体
        self.lm_head = ParallelLMHead(  # 创建语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.lm_head",  # 参数前缀
        )
        if getattr(self.config, "tie_word_embeddings", False):  # 如果绑定词嵌入
            self.lm_head.weight = self.model.embed_tokens.weight  # 共享权重
        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器

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

    def get_embed_and_head(self):  # 获取嵌入和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和lm_head权重

    def set_embed_and_head(self, embed, head):  # 设置嵌入和语言模型头权重
        del self.model.embed_tokens.weight  # 删除旧嵌入权重
        del self.lm_head.weight  # 删除旧lm_head权重
        self.model.embed_tokens.weight = embed  # 设置新嵌入权重
        self.lm_head.weight = head  # 设置新lm_head权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射
            ("qkv_proj", "q_proj", "q"),  # Q投影
            ("qkv_proj", "k_proj", "k"),  # K投影
            ("qkv_proj", "v_proj", "v"),  # V投影
            ("gate_up_proj", "gate_proj", 0),  # gate投影
            ("gate_up_proj", "up_proj", 1),  # up投影
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)  # 权重、fp8权重缩放和fp8激活缩放的参数映射
        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj",  # 检查点gate投影名称
            ckpt_down_proj_name="down_proj",  # 检查点down投影名称
            ckpt_up_proj_name="up_proj",  # 检查点up投影名称
            num_experts=self.config.num_experts,  # 专家数量
        )

        params_dict = dict(self.named_parameters())  # 获取参数字典
        num_nextn_layers = getattr(self.config, "num_nextn_predict_layers", 0)  # 获取next-n预测层数

        for name, loaded_weight in weights:  # 遍历所有权重
            if "lm_head.weight" in name and getattr(  # 如果是lm_head权重且绑定了词嵌入
                self.config, "tie_word_embeddings", False
            ):
                continue  # 跳过

            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码逆频率
                continue  # 跳过

            if num_nextn_layers > 0 and name.startswith("model.layers."):  # 如果有next-n层且是模型层权重
                parts = name.split(".")  # 拆分名称
                if len(parts) >= 3 and int(parts[2]) >= self.config.num_hidden_layers:  # 如果层号超出隐藏层数
                    continue  # 跳过（next-n层的权重不加载到主模型）

            is_found = False  # 是否找到匹配标志
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 跳过
                if "mlp.experts" in name:  # 如果是专家参数
                    continue  # 跳过（专家参数单独处理）
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                if name not in params_dict:  # 如果名称不在参数字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                is_found = True  # 设置找到标志
                break  # 跳出循环
            if is_found:  # 如果找到了匹配
                continue  # 跳到下一个权重

            # Handle expert weights (including fp8 weight_scale, input_scale)  # 处理专家权重（包括fp8权重缩放、输入缩放）
            is_expert_weight = False  # 专家权重标志
            for mapping in expert_params_mapping:  # 遍历专家参数映射
                param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 跳过
                is_expert_weight = True  # 设置专家权重标志
                name_mapped = name.replace(weight_name, param_name)  # 替换权重名
                if name_mapped not in params_dict:  # 如果映射后名称不在参数字典中
                    continue  # 跳过
                param = params_dict[name_mapped]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(  # 加载专家权重
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break  # 跳出循环
            if is_expert_weight:  # 如果是专家权重
                continue  # 跳到下一个权重

            if "router.gate." in name:  # 如果是路由器门控权重
                name = name.replace("router.", "")  # 移除router.前缀
            if name not in params_dict:  # 如果名称不在参数字典中
                continue  # 跳过
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重


EntryClass = [HYV3ForCausalLM]  # 入口类列表，用于模型注册
