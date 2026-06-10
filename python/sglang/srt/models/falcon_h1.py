# Falcon-H1混合注意力模型推理实现文件
# 本文件实现了Falcon-H1模型的推理逻辑，该模型结合了Mamba2状态空间模型和标准注意力机制
# 主要包含：带乘数缩放的MLP、混合注意力解码器层、模型主体和因果语言模型

import logging  # 导入日志模块
from typing import Any, Iterable, List, Optional, Set, Tuple  # 导入类型注解

import torch  # 导入PyTorch
from torch import nn  # 从PyTorch导入神经网络模块

from sglang.srt.configs.falcon_h1 import FalconH1Config  # 导入FalconH1配置类
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # 导入混合线性注意力后端
    HybridLinearAttnBackend,  # 混合线性注意力后端
    Mamba2AttnBackend,  # Mamba2注意力后端
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2  # 导入Mamba2混合器
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes  # 导入层通信器和层散射模式
from sglang.srt.layers.dp_attention import (  # 从数据并行注意力模块导入
    get_attention_tp_rank,  # 获取注意力张量并行秩
    get_attention_tp_size,  # 获取注意力张量并行大小
    is_dp_attention_enabled,  # 判断是否启用数据并行注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 从线性层模块导入
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 从词表并行嵌入模块导入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_executor.forward_context import get_attn_backend  # 导入注意力后端获取
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取
from sglang.srt.utils import add_prefix, is_cuda, make_layers  # 导入前缀添加、CUDA判断和层创建工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
_is_cuda = is_cuda()  # 判断当前是否为CUDA环境


class FalconH1MLP(nn.Module):  # FalconH1的MLP模块（带乘数缩放）
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层维度大小
        intermediate_size: int,  # 中间层维度大小
        hidden_act: str,  # 隐藏层激活函数名称
        layer_id: int,  # 层ID
        mlp_multipliers: List[float],  # MLP乘数列表[gate_multiplier, down_multiplier]
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        reduce_results: bool = True,  # 是否归约结果，默认为True
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # gate和up的合并列并行线性层
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度（gate和up各一份）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # down行并行线性层
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
            reduce_results=reduce_results,  # 是否归约结果
        )
        if hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. "  # 不支持的激活函数
                "Only silu is supported for now."  # 目前仅支持silu
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数
        self.layer_id = layer_id  # 保存层ID

        self.intermediate_size = intermediate_size  # 保存中间层维度
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小

        self.gate_multiplier, self.down_multiplier = mlp_multipliers  # 解包gate和down乘数

    def forward(  # 前向传播函数
        self,
        x,  # 输入张量
        forward_batch=None,  # 前向批次信息
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ):
        gate_up, _ = self.gate_up_proj(x)  # 通过gate_up投影
        gate_up[:, : self.intermediate_size // self.tp_size] *= self.gate_multiplier  # 对gate部分应用乘数缩放

        x = self.act_fn(gate_up)  # 应用SiLU激活函数和门控
        x, _ = self.down_proj(  # 通过down投影
            x,
            skip_all_reduce=use_reduce_scatter,  # 是否跳过全归约
        )
        x = x * self.down_multiplier  # 对down输出应用乘数缩放
        return x  # 返回输出


class FalconH1HybridAttentionDecoderLayer(nn.Module):  # FalconH1混合注意力解码器层（注意力+Mamba2）

    def __init__(  # 初始化函数
        self,
        config: FalconH1Config,  # FalconH1配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.attn_tp_rank = get_attention_tp_rank()  # 获取注意力张量并行秩
        self.attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        assert self.total_num_heads % self.attn_tp_size == 0  # 确保头数能被并行大小整除
        self.num_heads = self.total_num_heads // self.attn_tp_size  # 每个并行秩的头数
        self.total_num_kv_heads = config.num_key_value_heads  # 总KV头数
        if self.total_num_kv_heads >= self.attn_tp_size:  # 如果KV头数大于等于并行大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此进行分区
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头分配到多个张量并行GPU上
            assert self.total_num_kv_heads % self.attn_tp_size == 0  # 确保KV头数能被并行大小整除
        else:  # 否则KV头数小于并行大小
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此进行复制
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头复制到多个张量并行GPU上
            assert self.attn_tp_size % self.total_num_kv_heads == 0  # 确保并行大小能被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)  # 每个并行秩的KV头数
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q维度大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV维度大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = config.rope_parameters["rope_theta"]  # 旋转位置编码基数
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置编码数
        self.rope_scaling = config.rope_parameters  # 旋转位置编码缩放配置
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)  # 部分旋转因子
        self.layer_id = layer_id  # 保存层ID

        self.rotary_emb = get_rope(  # 获取旋转位置编码
            head_size=self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=self.max_position_embeddings,  # 最大位置
            rope_scaling=self.rope_scaling,  # 缩放配置
            base=self.rope_theta,  # 基数
            partial_rotary_factor=self.partial_rotary_factor,  # 部分旋转因子
            is_neox_style=True,  # 使用Neox风格
            dtype=torch.get_default_dtype(),  # see impl of get_rope  # 参见get_rope的实现
        )

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            config.hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=self.attn_tp_rank,  # 注意力张量并行秩
            tp_size=self.attn_tp_size,  # 注意力张量并行大小
        )

        self.o_proj = RowParallelLinear(  # 输出行并行线性投影
            self.total_num_heads * self.head_dim,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            reduce_results=False,  # 不归约结果
            tp_rank=self.attn_tp_rank,  # 注意力张量并行秩
            tp_size=self.attn_tp_size,  # 注意力张量并行大小
        )

        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            prefix=f"{prefix}.attn",  # 参数前缀
        )

        self.d_ssm = (  # SSM维度
            int(config.mamba_expand * config.hidden_size)  # 根据扩展系数计算
            if config.mamba_d_ssm is None  # 如果未指定
            else config.mamba_d_ssm  # 使用指定值
        )

        self.mamba = MambaMixer2(  # Mamba2混合器
            cache_params=config.mamba2_cache_params,  # 缓存参数
            hidden_size=config.hidden_size,  # 隐藏层维度
            use_conv_bias=config.mamba_conv_bias,  # 是否使用卷积偏置
            use_bias=config.mamba_proj_bias,  # 是否使用投影偏置
            n_groups=config.mamba_n_groups,  # 分组数
            rms_norm_eps=config.rms_norm_eps,  # RMS归一化epsilon
            activation=config.hidden_act,  # 激活函数
            use_rms_norm=config.mamba_rms_norm,  # 是否使用RMS归一化
            prefix=f"{prefix}.mixer",  # 参数前缀
        )

        # FalconH1 all layers are dense and have no nextn now  # FalconH1所有层都是密集层，目前没有nextn
        self.is_layer_sparse = False  # 是否为稀疏层，FalconH1全为密集层
        is_previous_layer_sparse = False  # 前一层是否为稀疏层
        is_next_layer_sparse = False  # 后一层是否为稀疏层

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 层散射模式
            layer_id=layer_id,  # 层ID
            num_layers=config.num_hidden_layers,  # 总层数
            is_layer_sparse=self.is_layer_sparse,  # 当前层是否稀疏
            is_previous_layer_sparse=is_previous_layer_sparse,  # 前一层是否稀疏
            is_next_layer_sparse=is_next_layer_sparse,  # 后一层是否稀疏
        )

        self.feed_forward = FalconH1MLP(  # 前馈网络
            hidden_size=self.hidden_size,  # 隐藏层维度
            intermediate_size=config.intermediate_size,  # 中间层维度
            hidden_act=config.hidden_act,  # 激活函数
            layer_id=layer_id,  # 层ID
            mlp_multipliers=config.mlp_multipliers,  # MLP乘数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 前馈前层归一化

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q的RMS归一化
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K的RMS归一化

        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,  # 层散射模式
            input_layernorm=self.input_layernorm,  # 输入层归一化
            post_attention_layernorm=self.pre_ff_layernorm,  # 注意力后层归一化
            allow_reduce_scatter=True,  # 允许reduce-scatter
        )

        self.alt_stream = alt_stream  # 保存备用CUDA流
        self.key_multiplier = config.key_multiplier  # K乘数

        self.ssm_out_multiplier = config.ssm_out_multiplier  # SSM输出乘数
        self.ssm_in_multiplier = config.ssm_in_multiplier  # SSM输入乘数

        self.attention_in_multiplier = config.attention_in_multiplier  # 注意力输入乘数
        self.attn_out_multiplier = config.attention_out_multiplier  # 注意力输出乘数

        self.groups_time_state_size = self.mamba.n_groups * config.mamba_d_state  # 分组时间状态大小
        self.zxbcdt_multipliers = config.ssm_multipliers  # Z/X/B/C/dt乘数
        self._init_mup_vector()  # 初始化μP缩放向量

    def _init_mup_vector(self):  # 初始化μP（maximal update parameterization）缩放向量
        """
        Non learnable per-block scaling vector composed of element-wise  # 不可学习的逐块缩放向量，由逐元素
        multipliersapplied to each separate contiguous block of the output  # 乘数组成，应用于线性投影输出的
        of the linear projection (in_proj) before further processing  # 每个独立连续块，在进一步处理
        (gating, convolution, SSM):  # （门控、卷积、SSM）之前：

            - Z block:  [0 : d_ssm]                      → zxbcdt_multipliers[0]  # Z块：[0 : d_ssm] → zxbcdt乘数[0]
            - X block:  [d_ssm : 2 * d_ssm]              → zxbcdt_multipliers[1]  # X块：[d_ssm : 2*d_ssm] → zxbcdt乘数[1]
            - B block:  [2 * d_ssm : 2 * d_ssm + G * S]  → zxbcdt_multipliers[2]  # B块：[2*d_ssm : 2*d_ssm+G*S] → zxbcdt乘数[2]
            - C block:  [2 * d_ssm + G * S : 2 * d_ssm + 2 * G * S]  # C块：[2*d_ssm+G*S : 2*d_ssm+2*G*S]
                        → zxbcdt_multipliers[3]  # → zxbcdt乘数[3]
            - dt block: [2 * d_ssm + 2 * G * S : end]    → zxbcdt_multipliers[4]  # dt块：[2*d_ssm+2*G*S : 结束] → zxbcdt乘数[4]

        where:  # 其中：
            - d_ssm:     Dimension of state-space model latent  # d_ssm：状态空间模型潜变量维度
            - G:         Number of groups (n_groups)  # G：分组数
            - S:         SSM state size per group  # S：每组SSM状态大小
            - All indices are divided by tp_size to support tensor parallelism  # 所有索引除以tp_size以支持张量并行
        """
        vector_shape = (  # 向量形状
            2 * self.d_ssm + 2 * self.groups_time_state_size + self.config.mamba_n_heads
        ) // self.tp_size  # 除以张量并行大小
        mup_vector = torch.ones(1, vector_shape)  # 初始化为全1向量
        # Z vector 0 -> d_ssm  # Z向量 0 -> d_ssm
        mup_vector[:, : self.d_ssm // self.tp_size] *= self.zxbcdt_multipliers[0]  # Z块乘数
        # X vector d_ssm -> 2 * d_ssm  # X向量 d_ssm -> 2*d_ssm
        mup_vector[
            :, (self.d_ssm // self.tp_size) : (2 * self.d_ssm // self.tp_size)
        ] *= self.zxbcdt_multipliers[1]  # X块乘数
        # B vector 2 * d_ssm -> 2 * d_ssm + (n_group * d_state)  # B向量 2*d_ssm -> 2*d_ssm+(n_group*d_state)
        mup_vector[
            :,
            (2 * self.d_ssm)
            // self.tp_size : (2 * self.d_ssm + self.groups_time_state_size)
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[2]  # B块乘数
        # C vector 2 * d_ssm + (n_group * d_state)  # C向量 2*d_ssm+(n_group*d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state)  # -> 2*d_ssm+2*(n_group*d_state)
        mup_vector[
            :,
            (2 * self.d_ssm + self.groups_time_state_size)
            // self.tp_size : (2 * self.d_ssm + 2 * self.groups_time_state_size)
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[3]  # C块乘数
        # dt vector 2 * d_ssm + 2 * (n_group * d_state)  # dt向量 2*d_ssm+2*(n_group*d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state) + n_heads  # -> 2*d_ssm+2*(n_group*d_state)+n_heads
        mup_vector[
            :,
            (2 * self.d_ssm + 2 * self.groups_time_state_size) // self.tp_size :,
        ] *= self.zxbcdt_multipliers[4]  # dt块乘数

        self.register_buffer("mup_vector", mup_vector, persistent=False)  # 注册为非持久化缓冲区

    def self_attention(  # 自注意力计算
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割为Q、K、V
        k = k * self.key_multiplier  # 对K应用乘数缩放
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码

        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力

        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        residual: Optional[torch.Tensor],  # 残差张量
        forward_batch: ForwardBatch,  # 前向批次信息
        **kwargs: Any,  # 其他关键字参数
    ):
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力计算
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            # Attention block  # 注意力块
            attention_hidden_states = self.self_attention(  # 计算自注意力
                positions=positions,  # 位置编码
                hidden_states=hidden_states * self.attention_in_multiplier,  # 隐藏状态乘以注意力输入乘数
                forward_batch=forward_batch,  # 前向批次信息
            )
            attention_hidden_states = attention_hidden_states * self.attn_out_multiplier  # 注意力输出乘以乘数

            attn_backend = get_attn_backend()  # 获取当前注意力后端
            assert isinstance(attn_backend, HybridLinearAttnBackend)  # 确保是混合线性注意力后端
            assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)  # 确保线性注意力后端是Mamba2
            # Mamba block  # Mamba块
            mamba_hidden_states = torch.empty_like(hidden_states)  # 创建与隐藏状态相同形状的空张量
            attn_backend.linear_attn_backend.forward(  # 通过Mamba2注意力后端前向传播
                self.mamba,  # Mamba混合器
                hidden_states * self.ssm_in_multiplier,  # 隐藏状态乘以SSM输入乘数
                mamba_hidden_states,  # Mamba输出
                layer_id=self.layer_id,  # 层ID
                forward_batch=forward_batch,  # 前向批次信息
                mup_vector=self.mup_vector,  # μP缩放向量
            )
            mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier  # Mamba输出乘以SSM输出乘数

            hidden_states = attention_hidden_states + mamba_hidden_states  # 合并注意力和Mamba输出

        # Fully Connected  # 全连接层
        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP计算
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(  # 判断是否使用reduce-scatter
            forward_batch
        )
        hidden_states = self.feed_forward(  # 通过前馈网络
            hidden_states, forward_batch, use_reduce_scatter
        )

        hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理层
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual  # 返回隐藏状态和残差


ALL_DECODER_LAYER_TYPES = {  # 所有解码器层类型映射
    "falcon_h1": FalconH1HybridAttentionDecoderLayer,  # falcon_h1类型映射到混合注意力解码器层
}


class FalconH1Model(nn.Module):  # FalconH1模型主体
    def __init__(  # 初始化函数
        self,
        config: FalconH1Config,  # FalconH1配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        alt_stream = torch.cuda.Stream() if _is_cuda else None  # 如果是CUDA则创建备用流
        self.embedding_multiplier = config.embedding_multiplier  # 嵌入乘数

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            org_num_embeddings=config.vocab_size,  # 原始嵌入数量
            use_attn_tp_group=is_dp_attention_enabled(),  # 是否使用注意力张量并行组
        )

        def get_layer(idx: int, prefix: str):  # 获取解码器层的工厂函数
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]  # 根据层类型获取层类
            return layer_class(  # 返回层实例
                config,  # 配置
                idx,  # 层索引
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 参数前缀
                alt_stream=alt_stream,  # 备用CUDA流
            )

        self.layers = make_layers(  # 创建解码器层
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"  # 层数、工厂函数、前缀
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化
        self.infer_count = 0  # 推理计数器

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        # mamba_cache_params: MambaCacheParams,  # Mamba缓存参数（已注释）
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for  # 传递序列索引张量，这是以下操作所需的
        # proper continuous batching computation including  # 正确的连续批处理计算，包括
        # chunked prefill  # 分块预填充
        if inputs_embeds is not None:  # 如果提供了输入嵌入
            hidden_states = inputs_embeds * self.embedding_multiplier  # 乘以嵌入乘数
        else:  # 否则使用词嵌入
            hidden_states = self.embed_tokens(input_ids) * self.embedding_multiplier  # 通过词嵌入并乘以嵌入乘数

        residual = None  # 初始化残差为None
        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                layer_id=i,  # 层ID
                positions=positions,  # 位置编码
                hidden_states=hidden_states,  # 隐藏状态
                residual=residual,  # 残差
                forward_batch=forward_batch,  # 前向批次信息
            )

        if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
            if residual is None:  # 如果没有残差
                hidden_states = self.final_layernorm(hidden_states)  # 对隐藏状态做层归一化
            else:  # 如果有残差
                hidden_states, _ = self.final_layernorm(hidden_states, residual)  # 融合层归一化和残差

        return hidden_states  # 返回隐藏状态


class FalconH1ForCausalLM(nn.Module):  # FalconH1因果语言模型
    fall_back_to_pt_during_load = False  # 加载时是否回退到PyTorch

    def __init__(  # 初始化函数
        self,
        config: FalconH1Config,  # FalconH1配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.pp_group = get_pp_group()  # 获取流水线并行组
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank  # 确保FalconH1只使用单级流水线
        self.quant_config = quant_config  # 保存量化配置
        self.model = FalconH1Model(  # 模型主体
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if config.tie_word_embeddings:  # 如果绑定词嵌入权重
            self.lm_head = self.model.embed_tokens  # 语言模型头共享词嵌入
        else:  # 否则不绑定
            self.lm_head = ParallelLMHead(  # 并行语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层维度
                quant_config=quant_config,  # 量化配置
                org_num_embeddings=config.vocab_size,  # 原始嵌入数量
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力张量并行组
            )
        self.lm_head = self.lm_head.float()  # 将LM头转为float32精度
        self.lm_head_multiplier = config.lm_head_multiplier  # LM头乘数
        self.logits_processor = LogitsProcessor(  # logits处理器
            config, logit_scale=self.lm_head_multiplier  # 使用LM头乘数作为logit缩放
        )

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        **kwargs,  # 其他关键字参数
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)  # 通过模型主体获取隐藏状态

        return self.logits_processor(  # 通过logits处理器
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):  # 获取词嵌入和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和LM头权重

    def set_embed_and_head(self, embed, head):  # 设置词嵌入和语言模型头权重
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重
        del self.lm_head.weight  # 删除旧的LM头权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重
        self.lm_head.weight = head  # 设置新的LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA操作

    def load_weights(  # 加载权重函数
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False  # 权重迭代器，是否为MTP
    ) -> Set[str]:
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # QKV投影中Q的映射
            ("qkv_proj", "k_proj", "k"),  # QKV投影中K的映射
            ("qkv_proj", "v_proj", "v"),  # QKV投影中V的映射
            ("gate_up_proj", "gate_proj", 0),  # gate_up投影中gate的映射
            ("gate_up_proj", "up_proj", 1),  # gate_up投影中up的映射
        ]

        params_dict = dict(self.named_parameters())  # 参数名字典
        loaded_params: Set[str] = set()  # 已加载参数集合
        for name, loaded_weight in weights:  # 遍历所有权重

            if "rotary_emb.inv_freq" in name:  # 如果是旋转嵌入频率
                continue  # 跳过

            if ".self_attn." in name:  # 如果名称包含".self_attn."
                name = name.replace(".self_attn", "")  # 去掉".self_attn"前缀

            if "A_log" in name:  # 如果名称包含"A_log"（Mamba参数）
                name = name.replace("A_log", "A")  # 替换为"A"

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果分片名不在权重名中
                    continue  # 跳过

                name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                    continue  # 跳过
                # Skip layers on other devices.  # 跳过其他设备上的层
                # if is_pp_missing_parameter(name, self):  # 如果是流水线缺失参数（已注释）
                #     continue  # 跳过（已注释）
                if name not in params_dict:  # 如果参数名不在字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader")  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果堆叠参数映射中没有匹配
                # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                    continue  # 跳过
                # if is_pp_missing_parameter(name, self):  # 如果是流水线缺失参数（已注释）
                #     continue  # 跳过（已注释）

                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器

                weight_loader(param, loaded_weight)  # 加载权重

            loaded_params.add(name)  # 将参数名添加到已加载集合
        return loaded_params  # 返回已加载参数集合


EntryClass = FalconH1ForCausalLM  # 模型入口类
