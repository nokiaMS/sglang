# LFM2（Liquid Foundation Model 2）模型的SGLang推理实现
# 混合架构：包含注意力层和短卷积层，使用优化因果卷积内核
"""
LFM2 (Liquid Foundation Model 2) implementation for SGLang. # SGLang的LFM2（Liquid Foundation Model 2）实现

This is a hybrid architecture with both attention and short conv layers. # 这是一个包含注意力和短卷积层的混合架构
- Attention layers use standard KV cache (RadixAttention) # 注意力层使用标准KV缓存（RadixAttention）
- Conv layers use MambaPool for state caching (via HybridReqToTokenPool) # 卷积层使用MambaPool进行状态缓存（通过HybridReqToTokenPool）

The model uses a gated 1D causal convolution (kernel=3) instead of attention # 该模型在某些层中使用门控1D因果卷积（kernel=3）替代注意力
in some layers, providing linear memory complexity for those layers. # 为这些层提供线性内存复杂度

Uses optimized causal_conv1d kernels from the mamba package for fast inference. # 使用mamba包中优化的causal_conv1d内核进行快速推理
"""

import logging # 导入日志模块
from typing import Iterable, Optional, Set, Tuple # 导入类型提示

import torch # 导入PyTorch深度学习框架
import torch.nn.functional as F # 导入PyTorch函数式API
from torch import nn # 导入神经网络模块

from sglang.srt.configs.lfm2 import Lfm2Config # 导入LFM2配置
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size # 导入分布式工具
from sglang.srt.layers.attention.mamba.causal_conv1d import ( # 导入因果卷积1D内核
    causal_conv1d_fn, # 因果卷积前向函数（prefill用）
    causal_conv1d_update, # 因果卷积更新函数（decode用）
)
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.layers.linear import ( # 导入线性层
    ColumnParallelLinear, # 列并行线性层
    MergedColumnParallelLinear, # 合并列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入Radix注意力
from sglang.srt.layers.rotary_embedding import get_rope # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息
from sglang.srt.model_executor.forward_context import get_req_to_token_pool # 导入请求到令牌池
from sglang.srt.model_loader.weight_utils import ( # 导入权重加载工具
    default_weight_loader, # 默认权重加载器
    sharded_weight_loader, # 分片权重加载器
)
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs # 导入工具函数

logger = logging.getLogger(__name__) # 获取日志记录器


class Lfm2MLP(nn.Module): # LFM2 MLP模块
    """MLP with SwiGLU activation.""" # 带SwiGLU激活的MLP

    def __init__( # 初始化方法
        self,
        config: Lfm2Config, # LFM2配置
        quant_config: Optional[QuantizationConfig] = None, # 可选量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        intermediate_size = config.intermediate_size # 获取中间层大小

        if config.block_auto_adjust_ff_dim: # 如果启用自动调整前馈维度
            intermediate_size = int(2 * intermediate_size / 3) # 调整为2/3
            if config.block_ffn_dim_multiplier is not None: # 如果有前馈维度乘数
                intermediate_size = int(
                    config.block_ffn_dim_multiplier * intermediate_size
                ) # 应用乘数
                intermediate_size = config.block_multiple_of * ( # 对齐到block_multiple_of的倍数
                    (intermediate_size + config.block_multiple_of - 1)
                    // config.block_multiple_of
                )

        self.w1 = ColumnParallelLinear( # 门控投影（列并行）
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )
        self.w3 = ColumnParallelLinear( # 上投影（列并行）
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w3", prefix),
        )
        self.w2 = RowParallelLinear( # 下投影（行并行）
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 前向传播方法
        gate, _ = self.w1(x) # 计算门控值
        up, _ = self.w3(x) # 计算上投影值
        out, _ = self.w2(F.silu(gate) * up) # SwiGLU激活：silu(gate)*up，再通过下投影
        return out # 返回输出


class Lfm2Attention(nn.Module): # LFM2注意力模块
    """Grouped-query attention with RoPE and Q/K layernorm.""" # 带RoPE和Q/K层归一化的分组查询注意力

    def __init__( # 初始化方法
        self,
        config: Lfm2Config, # LFM2配置
        layer_id: int, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 可选量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.hidden_size = config.hidden_size # 隐藏维度
        self.total_num_heads = config.num_attention_heads # 总注意力头数
        self.total_num_kv_heads = config.num_key_value_heads # 总KV头数
        self.head_dim = getattr(config, "head_dim", None) or ( # 头维度
            self.hidden_size // self.total_num_heads
        )
        self.scaling = self.head_dim**-0.5 # 缩放因子

        rope_parameters = getattr(config, "rope_parameters", None) # 获取RoPE参数
        if rope_parameters is not None and "rope_theta" in rope_parameters: # 如果有RoPE theta
            rope_theta = rope_parameters["rope_theta"] # 使用配置中的theta
        else:
            rope_theta = getattr(config, "rope_theta", 1000000.0) # 否则使用默认值

        self.rotary_emb = get_rope( # 创建旋转位置编码
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_scaling=rope_parameters or getattr(config, "rope_scaling", None),
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear( # QKV并行线性投影
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear( # 输出投影（行并行）
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps) # Q层归一化
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps) # K层归一化

        self.num_local_q_heads = self.qkv_proj.num_heads # 本地Q头数
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads # 本地KV头数

        self.attn = RadixAttention( # Radix注意力模块
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward( # 前向传播方法
        self,
        positions: torch.Tensor, # 位置编码
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        T = hidden_states.shape[0] # 获取序列长度
        qkv, _ = self.qkv_proj(hidden_states) # 计算QKV

        q_size = self.num_local_q_heads * self.head_dim # Q的维度
        kv_size = self.num_local_kv_heads * self.head_dim # KV的维度
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1) # 分割QKV

        q = q.reshape(T, self.num_local_q_heads, self.head_dim) # 重塑Q的形状
        k = k.reshape(T, self.num_local_kv_heads, self.head_dim) # 重塑K的形状

        q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape( # 对Q应用层归一化
            T, self.num_local_q_heads, self.head_dim
        )
        k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape( # 对K应用层归一化
            T, self.num_local_kv_heads, self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码

        attn_out = self.attn(q.reshape(T, -1), k.reshape(T, -1), v, forward_batch) # 计算注意力输出
        out, _ = self.out_proj(attn_out) # 通过输出投影
        return out # 返回输出


class Lfm2ShortConv(nn.Module): # LFM2短卷积模块
    """
    Gated short convolution layer using optimized causal_conv1d kernels. # 使用优化causal_conv1d内核的门控短卷积层

    Architecture: in_proj -> split(B, C, x) -> Bx -> conv1d -> C*conv_out -> out_proj # 架构：输入投影->分割(B,C,x)->Bx->卷积1D->C*卷积输出->输出投影
    - Uses double gating: B (before conv) and C (after conv) # 使用双重门控：B（卷积前）和C（卷积后）
    - Fixed-size cache: stores last (kernel_size - 1) tokens # 固定大小缓存：存储最后(kernel_size-1)个令牌
    - Uses causal_conv1d_fn for prefill and causal_conv1d_update for decode # prefill使用causal_conv1d_fn，decode使用causal_conv1d_update
    - Supports tensor parallelism: hidden dimension is sharded across TP ranks # 支持张量并行：隐藏维度在TP秩之间分片
    """

    def __init__( # 初始化方法
        self,
        config: Lfm2Config, # LFM2配置
        layer_idx: int, # 层索引
        quant_config: Optional[QuantizationConfig] = None, # 可选量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.layer_idx = layer_idx # 保存层索引
        self.conv_kernel = int(config.conv_L_cache) # 卷积核大小
        self.use_bias = bool(config.conv_bias) # 是否使用偏置
        self.hidden_size = config.hidden_size # 隐藏维度

        tp_size = get_tensor_model_parallel_world_size() # 获取TP大小
        self.hidden_size_per_partition = self.hidden_size // tp_size # 每个分区的隐藏维度

        # Use MergedColumnParallelLinear so each output (B, C, x) is sharded separately # 使用合并列并行线性层使每个输出(B,C,x)分别分片
        self.in_proj = MergedColumnParallelLinear( # 输入投影（合并列并行）
            config.hidden_size,
            [config.hidden_size] * 3,  # B, C, x each get hidden_size # B、C、x各获得hidden_size维度
            bias=self.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear( # 输出投影（行并行）
            config.hidden_size,
            config.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True, # 输入已分片
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # Conv weights sharded along hidden dimension: (hidden_size/tp, kernel_size) # 卷积权重按隐藏维度分片：(hidden_size/tp, kernel_size)
        self.conv_weight = nn.Parameter( # 卷积权重参数
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(self.conv_weight, {"weight_loader": sharded_weight_loader(0)}) # 设置分片权重加载器
        if self.use_bias: # 如果使用偏置
            self.conv_bias = nn.Parameter(torch.empty(self.hidden_size_per_partition)) # 卷积偏置参数
            set_weight_attrs(
                self.conv_bias, {"weight_loader": sharded_weight_loader(0)}
            )
        else:
            self.register_parameter("conv_bias", None) # 注册为None

    def forward( # 前向传播方法
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle(): # 如果是空闲模式
            return hidden_states # 直接返回

        layer_cache = get_req_to_token_pool().mamba2_layer_cache(self.layer_idx) # 获取当前层的Mamba2层缓存
        conv_state = layer_cache.conv[0] # 获取卷积状态
        req_pool_indices = forward_batch.req_pool_indices # 获取请求池索引
        mamba_indices = get_req_to_token_pool().get_mamba_indices(req_pool_indices) # 获取Mamba索引

        # Project and split into gates: B (pre-conv), C (post-conv), x (input) # 投影并分割为门控：B（卷积前）、C（卷积后）、x（输入）
        proj, _ = self.in_proj(hidden_states) # 通过输入投影
        B_gate, C_gate, x = proj.chunk(3, dim=-1) # 分割为三个门控
        Bx = B_gate * x # 计算B与x的逐元素乘积

        if forward_batch.forward_mode.is_decode(): # 如果是解码模式
            # Decode: single token per request, use optimized update kernel # 解码：每个请求单个令牌，使用优化的更新内核
            conv_out = causal_conv1d_update( # 使用解码更新内核
                Bx,
                conv_state,
                self.conv_weight,
                self.conv_bias,
                activation=None, # 无额外激活
                conv_state_indices=mamba_indices.to(torch.int32), # 卷积状态索引
            )
        else: # 否则是prefill模式
            # Prefill: multiple tokens, use varlen kernel # Prefill：多个令牌，使用变长内核
            T = hidden_states.shape[0] # 获取序列长度
            Bx_t = Bx.transpose(0, 1).contiguous() # 转置为(隐藏维度, 序列长度)

            # Build query_start_loc: [0, cumsum(seq_lens)...] # 构建查询起始位置：[0, cumsum(seq_lens)...]
            extend_start_loc = forward_batch.extend_start_loc # 获取扩展起始位置
            if extend_start_loc is not None and len(extend_start_loc) > 1: # 如果有多个序列
                query_start_loc = torch.cat( # 拼接起始位置
                    [
                        extend_start_loc,
                        torch.tensor(
                            [T], dtype=torch.int32, device=hidden_states.device
                        ),
                    ]
                )
                cache_indices = mamba_indices.to(torch.int32) # 缓存索引
                has_initial_state = forward_batch.extend_prefix_lens > 0 # 是否有初始状态
            else: # 单序列情况
                query_start_loc = torch.tensor(
                    [0, T], dtype=torch.int32, device=hidden_states.device
                )
                cache_indices = mamba_indices[:1].to(torch.int32) # 仅使用第一个索引
                has_initial_state = forward_batch.extend_prefix_lens[:1] > 0 # 检查第一个序列

            conv_out = causal_conv1d_fn( # 使用prefill因果卷积函数
                Bx_t,
                self.conv_weight,
                self.conv_bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                conv_states=conv_state,
                activation=None,
            ).transpose(0, 1) # 转置回(序列长度, 隐藏维度)

        output, _ = self.out_proj(C_gate * conv_out) # 输出投影：C门控*卷积输出
        return output # 返回输出


class Lfm2DecoderLayer(nn.Module): # LFM2解码器层
    """Decoder layer - either attention or conv based on config.""" # 解码器层——根据配置选择注意力或卷积

    def __init__( # 初始化方法
        self,
        config: Lfm2Config, # LFM2配置
        layer_id: int, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 可选量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.layer_type = config.layer_types[layer_id] # 获取层类型
        self.is_attention_layer = self.layer_type == "full_attention" # 是否为注意力层

        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps) # 操作子归一化
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps) # FFN归一化

        if self.is_attention_layer: # 如果是注意力层
            self.self_attn = Lfm2Attention( # 创建注意力模块
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else: # 否则是卷积层
            self.conv = Lfm2ShortConv( # 创建短卷积模块
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )

        self.feed_forward = Lfm2MLP( # 创建前馈网络
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

    def forward( # 前向传播方法
        self,
        layer_id: int, # 层ID
        positions: torch.Tensor, # 位置编码
        hidden_states: torch.Tensor, # 隐藏状态
        residual: Optional[torch.Tensor], # 残差
        forward_batch: ForwardBatch, # 前向批次信息
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not forward_batch.forward_mode.is_idle(): # 如果不是空闲模式
            residual = hidden_states # 保存残差
            normed = self.operator_norm(hidden_states) # 归一化

            if self.is_attention_layer: # 如果是注意力层
                hidden_states = self.self_attn(positions, normed, forward_batch) # 通过注意力层
            else:
                hidden_states = self.conv(normed, forward_batch) # 通过卷积层

            hidden_states = hidden_states + residual # 添加残差连接
            hidden_states = hidden_states + self.feed_forward( # 通过前馈网络并添加残差
                self.ffn_norm(hidden_states)
            )

        return hidden_states, residual # 返回隐藏状态和残差


class Lfm2Model(nn.Module): # LFM2模型类
    def __init__( # 初始化方法
        self,
        config: Lfm2Config, # LFM2配置
        quant_config: Optional[QuantizationConfig] = None, # 可选量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置

        self.embed_tokens = VocabParallelEmbedding( # 词表并行嵌入层
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Count attention layers for KV cache sizing # 计算注意力层数用于KV缓存大小设定
        self.num_attention_layers = sum(
            1 for lt in config.layer_types if lt == "full_attention"
        )

        def get_layer(idx: int, prefix: str, **kwargs): # 获取解码器层的工厂函数
            return Lfm2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers( # 创建解码器层列表
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps) # 嵌入归一化层

    def forward( # 前向传播方法
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None, # 可选输入嵌入
    ) -> torch.Tensor:
        hidden_states = ( # 获取隐藏状态
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )

        residual = None # 初始化残差
        for i in range(len(self.layers)): # 遍历所有层
            hidden_states, residual = self.layers[i]( # 通过当前层
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        return self.embedding_norm(hidden_states) # 返回归一化后的隐藏状态


class Lfm2ForCausalLM(nn.Module): # LFM2因果语言模型
    """LFM2 for causal language modeling with hybrid attention/conv architecture.""" # 带混合注意力/卷积架构的LFM2因果语言建模

    fall_back_to_pt_during_load = False # 加载权重时不回退到PyTorch默认方式

    def __init__( # 初始化方法
        self,
        config: Lfm2Config, # LFM2配置
        quant_config: Optional[QuantizationConfig] = None, # 可选量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.pp_group = get_pp_group() # 获取流水线并行组
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank # 当前仅支持单阶段

        self.quant_config = quant_config # 保存量化配置
        self.model = Lfm2Model(config, quant_config, prefix=add_prefix("model", prefix)) # 创建LFM2模型
        self.lm_head = ParallelLMHead( # 并行语言模型头
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config) # logits处理器
        self.num_attention_layers = self.model.num_attention_layers # 注意力层数

    def get_num_kv_cache_layers(self) -> int: # 获取KV缓存层数
        return self.num_attention_layers # 返回注意力层数

    def get_input_embeddings(self) -> nn.Embedding: # 获取输入嵌入层
        return self.model.embed_tokens # 返回词嵌入层

    @torch.no_grad() # 禁用梯度计算
    def forward( # 前向推理方法
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None, # 可选输入嵌入
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds) # 通过模型获取隐藏状态
        return self.logits_processor( # 通过logits处理器
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights( # 加载权重方法
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        stacked_params_mapping = [ # 堆叠参数映射
            ("qkv_proj", "q_proj", "q"), # Q投影
            ("qkv_proj", "k_proj", "k"), # K投影
            ("qkv_proj", "v_proj", "v"), # V投影
        ]

        params_dict = dict(self.named_parameters()) # 获取参数字典
        loaded_params: Set[str] = set() # 已加载参数集合
        embed_tokens_weight = None # 词嵌入权重暂存

        for name, loaded_weight in weights: # 遍历所有权重
            if "rotary_emb.inv_freq" in name: # 跳过旋转嵌入的逆频率
                continue

            if "embed_tokens.weight" in name: # 暂存词嵌入权重（可能需要绑定）
                embed_tokens_weight = loaded_weight

            # Handle conv weight/bias naming: HF uses conv.conv, we use conv_weight/conv_bias # 处理卷积权重/偏置命名：HF使用conv.conv，我们使用conv_weight/conv_bias
            if ".conv.conv.weight" in name: # 如果是卷积权重
                name = name.replace(".conv.conv.weight", ".conv.conv_weight") # 替换名称
                loaded_weight = loaded_weight.squeeze(1)  # (D, 1, K) -> (D, K) # 压缩维度
            if ".conv.conv.bias" in name: # 如果是卷积偏置
                name = name.replace(".conv.conv.bias", ".conv.conv_bias") # 替换名称

            # Handle QKV stacking # 处理QKV堆叠
            for param_name, weight_name, shard_id in stacked_params_mapping: # 遍历堆叠映射
                if weight_name not in name: # 如果不匹配
                    continue
                name = name.replace(weight_name, param_name) # 替换名称
                if name.endswith(".bias") and name not in params_dict: # 跳过不存在的偏置
                    break
                if name not in params_dict: # 跳过不存在的参数
                    break
                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader") # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                loaded_params.add(name) # 记录已加载
                break
            else: # 非堆叠参数
                if name.endswith(".bias") and name not in params_dict: # 跳过不存在的偏置
                    continue
                if name not in params_dict: # 跳过不存在的参数
                    continue

                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重
                loaded_params.add(name) # 记录已加载

        # Handle tied lm_head weight # 处理绑定的语言模型头权重
        if "lm_head.weight" not in loaded_params and "lm_head.weight" in params_dict: # 如果lm_head未加载但存在
            if embed_tokens_weight is not None: # 如果有词嵌入权重
                param = params_dict["lm_head.weight"] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取加载器
                weight_loader(param, embed_tokens_weight) # 用词嵌入权重加载
                loaded_params.add("lm_head.weight") # 记录已加载

        return loaded_params # 返回已加载参数集合


EntryClass = [Lfm2ForCausalLM] # 入口类列表
