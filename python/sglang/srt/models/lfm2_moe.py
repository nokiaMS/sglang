# LFM2-MoE模型推理实现文件
# 本文件实现了Liquid Foundation Model 2混合专家模型的推理专用版本
# 采用混合架构，包含注意力层、短卷积层和MoE层
# 注意力层使用标准KV缓存，卷积层使用MambaPool缓存状态
# 前num_dense_layers层使用密集MLP，其余使用sigmoid路由的MoE

"""
LFM2-MoE (Liquid Foundation Model 2 - Mixture of Experts) implementation for SGLang.

This is a hybrid architecture with attention, ShortConv, and MoE layers:
- Attention layers use standard KV cache (RadixAttention)
- Conv layers use MambaPool for state caching (via HybridReqToTokenPool)
- First `num_dense_layers` use dense MLP, rest use MoE with sigmoid routing

Key MoE characteristics:
- Sigmoid routing (not softmax) - auxiliary-loss-free style
- Expert bias (fp32) affects selection but not weighting
- Post-hoc normalization of top-k weights
"""

from typing import Iterable, Optional, Set, Tuple  # 类型提示

import torch  # PyTorch核心库
from torch import nn  # 神经网络模块

from sglang.srt.configs.lfm2_moe import Lfm2MoeConfig  # LFM2-MoE配置类
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 分布式通信
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活与乘法融合层
from sglang.srt.layers.attention.mamba.causal_conv1d import (  # 因果1D卷积
    causal_conv1d_fn,  # 因果1D卷积函数（预填充模式）
    causal_conv1d_update,  # 因果1D卷积更新（解码模式）
)
from sglang.srt.layers.layernorm import RMSNorm  # 均方根归一化层
from sglang.srt.layers.linear import (  # 并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 融合MoE Triton实现
from sglang.srt.layers.moe.topk import TopK  # Top-K选择器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 获取旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息
from sglang.srt.model_executor.forward_context import get_req_to_token_pool  # 获取请求到token池
from sglang.srt.model_loader.weight_utils import (  # 权重加载工具
    default_weight_loader,  # 默认权重加载器
    sharded_weight_loader,  # 分片权重加载器
)
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs  # 工具函数


class Lfm2MoeMLP(nn.Module):
    """Dense MLP for first N layers (before MoE kicks in)."""
    """前N层的密集MLP（MoE之前的层）"""

    def __init__(
        self,
        config: Lfm2MoeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        # Use MergedColumnParallelLinear for w1/w3 (gate/up projections)
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并
            config.hidden_size,
            [config.intermediate_size] * 2,  # 两个中间层大小
            bias=False,  # 无偏置
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(  # 下投影
            config.intermediate_size,
            config.hidden_size,
            bias=False,  # 无偏置
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()  # SiLU激活与乘法融合函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """密集MLP前向传播：门控上投影 -> SiLU激活 -> 下投影"""
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影
        x = self.act_fn(gate_up)  # SiLU激活和门控乘法
        out, _ = self.down_proj(x)  # 下投影
        return out  # 返回MLP输出


class Lfm2MoeSparseMoeBlock(nn.Module):
    """
    Sparse MoE block with sigmoid routing using optimized FusedMoE.
    使用sigmoid路由的稀疏MoE块

    Key features:
    - Sigmoid scoring (not softmax) - auxiliary-loss-free style
    - Expert bias (fp32) for load balancing
    - Bias affects selection only, not weighting
    - Uses FusedMoE for efficient batched expert computation
    关键特性：
    - Sigmoid评分（非softmax）——无辅助损失风格
    - 专家偏置（fp32）用于负载均衡
    - 偏置仅影响选择，不影响权重
    - 使用FusedMoE进行高效批量专家计算
    """

    def __init__(
        self,
        config: Lfm2MoeConfig,  # 模型配置
        layer_idx: int,  # 层索引
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.routed_scaling_factor = config.routed_scaling_factor  # 路由缩放因子

        if self.tp_size > config.num_experts:  # TP大小不能超过专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Gate (router) - outputs logits for each expert
        self.gate = ReplicatedLinear(  # 路由门控（复制线性层）
            config.hidden_size,
            config.num_experts,  # 输出维度等于专家数
            bias=False,  # 无偏置
            quant_config=None,  # 门控不量化
            prefix=add_prefix("gate", prefix),
        )

        # Expert bias (fp32) - affects selection but not weighting
        if config.use_expert_bias:  # 使用专家偏置
            self.expert_bias = nn.Parameter(
                torch.zeros(config.num_experts, dtype=torch.float32)  # 零初始化
            )
        else:  # 不使用专家偏置
            self.register_parameter("expert_bias", None)

        # TopK selector with sigmoid scoring
        self.topk = TopK(  # 使用sigmoid评分的Top-K选择器
            top_k=config.num_experts_per_tok,
            layer_id=layer_idx,
            renormalize=config.norm_topk_prob,  # 是否归一化Top-K概率
            scoring_func="sigmoid",  # sigmoid评分函数
            correction_bias=self.expert_bias if config.use_expert_bias else None,  # 修正偏置
        )

        # FusedMoE for efficient batched expert computation
        # Note: We intentionally do NOT pass routed_scaling_factor to FusedMoE.
        # While FusedMoE supports it, passing it there increases numerical
        # differences vs HuggingFace (likely due to different code paths in the
        # Triton runner when scaling_factor != None). We apply it manually below.
        self.experts = FusedMoE(  # 融合MoE专家
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
            layer_id=layer_idx,
            reduce_results=True,  # 在MoE内部归约
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Optimized expert forward pass using FusedMoE."""
        """使用FusedMoE的优化专家前向传播"""
        # Get router logits
        router_logits, _ = self.gate(hidden_states)  # 计算路由logits

        # Select top-k experts with sigmoid scoring
        topk_output = self.topk(hidden_states, router_logits)  # Top-K选择

        # Run fused expert computation
        final_hidden_states = self.experts(hidden_states, topk_output)  # 执行融合专家计算

        # Apply routed scaling factor (see __init__ comment for why not in FusedMoE)
        return final_hidden_states * self.routed_scaling_factor  # 手动应用路由缩放因子


class Lfm2MoeAttention(nn.Module):
    """Grouped-query attention with RoPE and Q/K layernorm."""
    """带RoPE和Q/K层归一化的分组查询注意力"""

    def __init__(
        self,
        config: Lfm2MoeConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        self.total_num_kv_heads = config.num_key_value_heads  # 总KV头数
        self.head_dim = self.hidden_size // self.total_num_heads  # 头维度
        self.scaling = self.head_dim**-0.5  # 缩放因子

        rope_parameters = getattr(config, "rope_parameters", None)  # 获取RoPE参数
        if rope_parameters is not None and "rope_theta" in rope_parameters:  # 从rope_parameters获取theta
            rope_theta = rope_parameters["rope_theta"]
        else:  # 从config直接获取
            rope_theta = getattr(config, "rope_theta", 1000000.0)

        self.rotary_emb = get_rope(  # 旋转位置编码
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 128000),
            rope_scaling=rope_parameters or getattr(config, "rope_scaling", None),
            base=rope_theta,
            is_neox_style=True,  # 使用Neox风格
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,  # 无偏置
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,  # 无偏置
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)  # Q层归一化
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)  # K层归一化

        self.num_local_q_heads = self.qkv_proj.num_heads  # 本地Q头数
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads  # 本地KV头数

        self.attn = RadixAttention(  # 基数注意力
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """注意力前向传播：QKV投影 -> QK归一化 -> RoPE -> 注意力计算 -> 输出投影"""
        T = hidden_states.shape[0]  # 序列长度
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影

        q_size = self.num_local_q_heads * self.head_dim  # Q的总维度
        kv_size = self.num_local_kv_heads * self.head_dim  # KV的总维度
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)  # 分离Q、K、V

        q = q.reshape(T, self.num_local_q_heads, self.head_dim)  # 重塑Q形状
        k = k.reshape(T, self.num_local_kv_heads, self.head_dim)  # 重塑K形状

        q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape(  # Q层归一化
            T, self.num_local_q_heads, self.head_dim
        )
        k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape(  # K层归一化
            T, self.num_local_kv_heads, self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码

        attn_out = self.attn(q.reshape(T, -1), k.reshape(T, -1), v, forward_batch)  # 计算注意力
        out, _ = self.out_proj(attn_out)  # 输出投影
        return out  # 返回注意力输出


class Lfm2MoeShortConv(nn.Module):
    """
    Gated short convolution layer using optimized causal_conv1d kernels.
    使用优化因果1D卷积核的门控短卷积层

    Architecture: in_proj -> split(B, C, x) -> Bx -> conv1d -> C*conv_out -> out_proj
    - Supports tensor parallelism: hidden dimension is sharded across TP ranks
    支持张量并行：隐藏维度在TP秩之间分片
    """

    def __init__(
        self,
        config: Lfm2MoeConfig,  # 模型配置
        layer_idx: int,  # 层索引
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.layer_idx = layer_idx  # 保存层索引
        self.conv_kernel = int(config.conv_L_cache)  # 卷积核大小
        self.use_bias = bool(config.conv_bias)  # 是否使用偏置
        self.hidden_size = config.hidden_size  # 隐藏层大小

        # Get tensor parallel size for sharding
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.hidden_size_per_partition = self.hidden_size // self.tp_size  # 每个分区的隐藏大小

        # Use MergedColumnParallelLinear so each output (B, C, x) is sharded separately
        self.in_proj = MergedColumnParallelLinear(  # 输入投影（B, C, x各分片）
            config.hidden_size,
            [config.hidden_size] * 3,  # B, C, x各有hidden_size维度
            bias=self.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(  # 输出投影
            config.hidden_size,
            config.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,  # 输入已经是并行的
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # Conv weights sharded along hidden dimension: (hidden_size/tp, kernel_size)
        self.conv_weight = nn.Parameter(  # 卷积权重（按隐藏维度分片）
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(self.conv_weight, {"weight_loader": sharded_weight_loader(0)})  # 设置分片加载器
        if self.use_bias:  # 卷积偏置
            self.conv_bias = nn.Parameter(torch.empty(self.hidden_size_per_partition))
            set_weight_attrs(
                self.conv_bias, {"weight_loader": sharded_weight_loader(0)}  # 设置分片加载器
            )
        else:  # 无偏置
            self.register_parameter("conv_bias", None)

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """短卷积前向传播：输入投影 -> 门控乘法 -> 因果卷积 -> 输出投影"""
        if forward_batch.forward_mode.is_idle():  # 空闲模式直接返回
            return hidden_states

        layer_cache = get_req_to_token_pool().mamba2_layer_cache(self.layer_idx)  # 获取层缓存
        conv_state = layer_cache.conv[0]  # 卷积状态
        req_pool_indices = forward_batch.req_pool_indices  # 请求池索引
        mamba_indices = get_req_to_token_pool().get_mamba_indices(req_pool_indices)  # Mamba索引

        proj, _ = self.in_proj(hidden_states)  # 输入投影
        B_gate, C_gate, x = proj.chunk(3, dim=-1)  # 分离为B门控、C门控和x
        Bx = B_gate * x  # B门控乘以x

        if forward_batch.forward_mode.is_decode():  # 解码模式
            conv_out = causal_conv1d_update(  # 使用增量卷积更新
                Bx,
                conv_state,
                self.conv_weight,
                self.conv_bias,
                activation=None,  # 无额外激活
                conv_state_indices=mamba_indices.to(torch.int32),  # 状态索引
            )
        else:  # 预填充模式
            T = hidden_states.shape[0]  # 序列长度
            Bx_t = Bx.transpose(0, 1).contiguous()  # 转置为(seq_len, batch)

            # Build query_start_loc for variable-length sequences
            # causal_conv1d_fn expects [start0, start1, ..., startN, T]
            extend_start_loc = forward_batch.extend_start_loc  # 扩展起始位置
            if extend_start_loc is not None and len(extend_start_loc) > 1:  # 多序列
                # Multiple sequences: append T to extend_start_loc
                # Allocate and fill to avoid torch.cat overhead
                query_start_loc = extend_start_loc.new_empty(len(extend_start_loc) + 1)  # 分配空间
                query_start_loc[:-1] = extend_start_loc  # 填充起始位置
                query_start_loc[-1] = T  # 最后一个元素为总长度
                cache_indices = mamba_indices.to(torch.int32)  # 缓存索引
                has_initial_state = forward_batch.extend_prefix_lens > 0  # 是否有初始状态
            else:  # 单序列
                # Single sequence: [0, T]
                query_start_loc = hidden_states.new_tensor([0, T], dtype=torch.int32)  # [0, T]
                cache_indices = mamba_indices[:1].to(torch.int32)  # 仅取第一个
                has_initial_state = forward_batch.extend_prefix_lens[:1] > 0  # 第一个请求的初始状态

            conv_out = causal_conv1d_fn(  # 使用完整卷积函数
                Bx_t,
                self.conv_weight,
                self.conv_bias,
                query_start_loc=query_start_loc,  # 查询起始位置
                cache_indices=cache_indices,  # 缓存索引
                has_initial_state=has_initial_state,  # 是否有初始状态
                conv_states=conv_state,  # 卷积状态
                activation=None,  # 无额外激活
            ).transpose(0, 1)  # 转置回(batch, seq_len)

        output, _ = self.out_proj(C_gate * conv_out)  # C门控乘以卷积输出后投影
        return output  # 返回短卷积输出


class Lfm2MoeDecoderLayer(nn.Module):
    """
    Decoder layer with attention/conv and dense MLP or MoE.
    带注意力/卷积和密集MLP或MoE的解码器层

    - Layers 0 to num_dense_layers-1: use Lfm2MoeMLP (dense)
    - Layers num_dense_layers+: use Lfm2MoeSparseMoeBlock (MoE)
    - 前0到num_dense_layers-1层使用密集MLP
    - num_dense_layers及之后的层使用稀疏MoE
    """

    def __init__(
        self,
        config: Lfm2MoeConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.layer_type = config.layer_types[layer_id]  # 当前层类型
        self.is_attention_layer = self.layer_type == "full_attention"  # 是否为注意力层

        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)  # 算子归一化
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)  # FFN归一化

        # Attention or Conv
        if self.is_attention_layer:  # 注意力层
            self.self_attn = Lfm2MoeAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:  # 卷积层
            self.conv = Lfm2MoeShortConv(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )

        # Dense MLP or MoE
        if layer_id < config.num_dense_layers:  # 密集层
            self.feed_forward = Lfm2MoeMLP(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        else:  # MoE层
            self.feed_forward = Lfm2MoeSparseMoeBlock(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )

    def forward(
        self,
        layer_id: int,  # 层ID
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        residual: Optional[torch.Tensor],  # 残差连接
        forward_batch: ForwardBatch,  # 前向批次
        **kwargs,  # 其他参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播：归一化 -> 注意力/卷积 -> 残差 -> FFN归一化 -> MLP/MoE"""
        if not forward_batch.forward_mode.is_idle():  # 非空闲模式
            residual = hidden_states  # 保存残差
            normed = self.operator_norm(hidden_states)  # 算子归一化

            if self.is_attention_layer:  # 注意力计算
                hidden_states = self.self_attn(positions, normed, forward_batch)
            else:  # 卷积计算
                hidden_states = self.conv(normed, forward_batch)

            hidden_states = hidden_states + residual  # 残差连接
            hidden_states = hidden_states + self.feed_forward(  # FFN计算并残差连接
                self.ffn_norm(hidden_states)  # FFN归一化
            )

        return hidden_states, residual  # 返回隐藏状态和残差


class Lfm2MoeModel(nn.Module):
    """LFM2-MoE模型主体，包含嵌入层、解码器层和最终归一化"""

    def __init__(
        self,
        config: Lfm2MoeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,  # 原始嵌入数
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Count attention layers for KV cache sizing
        self.num_attention_layers = sum(  # 统计注意力层数
            1 for lt in config.layer_types if lt == "full_attention"
        )

        def get_layer(idx: int, prefix: str, **kwargs):  # 获取解码器层
            return Lfm2MoeDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers(  # 创建解码器层
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)  # 嵌入归一化

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        """模型主体前向传播：嵌入 -> 解码器层 -> 归一化"""
        hidden_states = (  # 获取隐藏状态
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        )

        residual = None  # 初始无残差
        for i in range(len(self.layers)):  # 遍历所有解码器层
            hidden_states, residual = self.layers[i](
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        return self.embedding_norm(hidden_states)  # 返回归一化后的隐藏状态


class Lfm2MoeForCausalLM(nn.Module):
    """LFM2-MoE for causal language modeling."""
    """LFM2-MoE因果语言模型"""

    fall_back_to_pt_during_load = False  # 加载权重时不回退到PyTorch

    def __init__(
        self,
        config: Lfm2MoeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.pp_group = get_pp_group()  # 获取流水线并行组
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank  # 必须是单秩

        self.quant_config = quant_config  # 保存量化配置
        self.model = Lfm2MoeModel(  # 模型主体
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,  # 原始嵌入数
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.num_attention_layers = self.model.num_attention_layers  # 注意力层数

    def get_num_kv_cache_layers(self) -> int:
        """获取KV缓存层数"""
        return self.num_attention_layers  # 返回注意力层数

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
        **kwargs,  # 其他参数
    ):
        """因果语言模型前向传播：模型主体 -> logits处理"""
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)  # 模型前向
        return self.logits_processor(  # 返回logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False  # 权重列表，是否为MTP
    ) -> Set[str]:
        """Load weights with FusedMoE expert format."""
        """加载权重，支持FusedMoE专家格式"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),  # Q合并
            ("qkv_proj", "k_proj", "k"),  # K合并
            ("qkv_proj", "v_proj", "v"),  # V合并
            # Dense MLP w1/w3 -> gate_up_proj
            ("gate_up_proj", "w1", 0),  # w1合并到门控投影
            ("gate_up_proj", "w3", 1),  # w3合并到上投影
        ]

        # FusedMoE expert params mapping
        # HF format: experts.{expert_id}.w{1,2,3}.weight
        # FusedMoE format: experts.w13_weight, experts.w2_weight
        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())  # 参数字典
        loaded_params: Set[str] = set()  # 已加载参数集合
        embed_tokens_weight = None  # 嵌入权重缓存

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码频率
                continue

            if "embed_tokens.weight" in name:  # 缓存嵌入权重
                embed_tokens_weight = loaded_weight

            # Handle conv weight/bias naming: HF uses conv.conv, we use conv_weight/conv_bias
            if ".conv.conv.weight" in name:  # 处理卷积权重命名差异
                name = name.replace(".conv.conv.weight", ".conv.conv_weight")
                loaded_weight = loaded_weight.squeeze(1)  # (D, 1, K) -> (D, K) 移除多余维度
            if ".conv.conv.bias" in name:  # 处理卷积偏置命名差异
                name = name.replace(".conv.conv.bias", ".conv.conv_bias")

            # Handle dense MLP w2 -> down_proj
            if "feed_forward.w2" in name and "experts" not in name:  # 密集MLP的w2映射为down_proj
                name = name.replace("feed_forward.w2", "feed_forward.down_proj")

            # Handle stacked params (QKV, dense MLP gate_up)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:  # 名称不包含权重名则跳过
                    continue
                # Skip expert weights (handled below)
                if "experts" in name:  # 跳过专家权重（单独处理）
                    continue
                name = name.replace(weight_name, param_name)  # 替换为参数名
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置
                    break
                if name not in params_dict:  # 跳过不存在的参数
                    break
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader")  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重分片
                loaded_params.add(name)  # 记录已加载
                break
            else:
                # Handle MoE expert weights using FusedMoE format
                # HF format: model.layers.X.feed_forward.experts.Y.wZ.weight
                # FusedMoE format: model.layers.X.feed_forward.experts.w13_weight/w2_weight
                for (  # 遍历专家参数映射
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:  # 名称不包含权重名则跳过
                        continue
                    # Build our parameter name
                    name = name.replace(weight_name, param_name)  # 替换为参数名
                    if name not in params_dict:  # 跳过不存在的参数
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载专家权重
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(name)  # 记录已加载
                    break
                else:
                    # Handle regular weights
                    if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置
                        continue
                    if name not in params_dict:  # 跳过不存在的参数
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                    loaded_params.add(name)  # 记录已加载

        # Handle tied lm_head weight
        if "lm_head.weight" not in loaded_params and "lm_head.weight" in params_dict:  # 处理共享lm_head权重
            if embed_tokens_weight is not None:  # 使用嵌入权重
                param = params_dict["lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, embed_tokens_weight)  # 加载共享权重
                loaded_params.add("lm_head.weight")

        return loaded_params  # 返回已加载参数集合


EntryClass = [Lfm2MoeForCausalLM]  # 入口类列表
