# Jet Nemotron 模型实现
# 该文件实现了 Jet Nemotron 因果语言模型，包含 JetBlock（线性注意力块）、
# JetNemotronAttention（标准注意力）、动态短卷积等组件，
# 支持混合线性注意力和标准注意力的混合架构。

from collections.abc import Iterable  # 导入可迭代类型
from typing import cast  # 导入类型转换工具

import einops  # 导入张量重排库
import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块

from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig  # 导入Jet配置类
from sglang.srt.layers.attention.fla.fused_recurrent import (  # 导入融合递归门控delta规则更新
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated  # 导入门控RMS归一化
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # 导入混合线性注意力后端
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType  # 导入池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_executor.forward_context import get_attn_backend  # 导入获取注意力后端函数
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Model  # 导入Qwen2模型组件
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class DynamicShortConvolutionKernelGenerator(nn.Module):
    """动态短卷积核生成器，根据输入生成卷积核权重。"""
    def __init__(
        self,
        input_size: int,  # 输入维度大小
        hidden_size: int,  # 隐藏层维度大小
        output_size: int,  # 输出维度大小
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.w1 = ColumnParallelLinear(  # 第一层列并行线性层
            input_size,  # 输入维度
            hidden_size,  # 输出维度（隐藏层）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("w1", prefix),  # 参数前缀
        )

        self.act = nn.SiLU()  # SiLU激活函数

        self.w2 = ColumnParallelLinear(  # 第二层列并行线性层
            hidden_size,  # 输入维度（隐藏层）
            output_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("w2", prefix),  # 参数前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：生成动态卷积核权重。"""
        x, _ = self.w1(x)  # 通过第一层线性变换
        x = self.act(x)  # 应用SiLU激活
        x, _ = self.w2(x)  # 通过第二层线性变换
        return x  # 返回生成的卷积核权重


class DynamicShortConvolution(nn.Module):
    """动态短卷积模块，使用生成的卷积核进行1D卷积操作。"""
    def __init__(
        self,
        hidden_size: int,  # 隐藏层维度大小
        kernel_size: int,  # 卷积核大小
        generator_input_size: int,  # 核生成器输入维度
        generator_reduction: int,  # 核生成器缩减因子
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        generator_hidden_size = hidden_size // generator_reduction  # 计算生成器隐藏层维度

        self.kernel_generator = DynamicShortConvolutionKernelGenerator(  # 创建卷积核生成器
            input_size=generator_input_size,  # 生成器输入维度
            hidden_size=generator_hidden_size,  # 生成器隐藏层维度
            output_size=hidden_size * kernel_size,  # 生成器输出维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("kernel_generator", prefix),  # 参数前缀
        )

        self.hidden_size = hidden_size  # 保存隐藏层维度
        self.kernel_size = kernel_size  # 保存卷积核大小

    def forward(
        self,
        x: torch.Tensor,  # (cu_seq_len, hidden_size) 输入张量
        *,
        conv_state: torch.Tensor,  # (batch_size, hidden_size, kernel_size - 1) 卷积状态
        generator_input: torch.Tensor,  # (cu_seq_len, generator_input_size) 核生成器输入
        seq_lens: torch.Tensor,  # (batch_size,) 序列长度
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播：执行动态短卷积操作，返回输出和更新后的卷积状态。"""
        """
        Args:
            x: (cu_seq_len, hidden_size) 输入张量
            conv_state: (batch_size, hidden_size, kernel_size - 1) 卷积状态缓存
            generator_input: (cu_seq_len, generator_input_size) 卷积核生成器输入
            seq_lens: (batch_size,) 每个批次的序列长度

        Returns:
            out: (cu_seq_len, hidden_size) 卷积输出
            conv_state: (batch_size, hidden_size, kernel_size - 1) 更新后的卷积状态
        """

        x_seqs = self._continuous_to_seqs(x, seq_lens=seq_lens)  # 将连续张量拆分为序列列表
        conv_state = einops.rearrange(conv_state, "b d k -> b k d")  # 重排卷积状态维度
        x_seqs = [torch.cat([conv_state[i], x_seqs[i]]) for i in range(len(x_seqs))]  # 拼接卷积状态和输入序列
        x = self._seqs_to_batch(
            x_seqs
        )  # (batch_size, max_seq_len + kernel_size - 1, hidden_size) 将序列列表转为批处理张量

        x = einops.rearrange(x, "b l d -> b d l")  # 重排维度以适配卷积操作

        new_conv_state = x[
            :, :, -(self.kernel_size - 1) :
        ]  # (batch_size, hidden_size, kernel_size - 1) 提取新的卷积状态

        x = x.unfold(
            dimension=-1, size=self.kernel_size, step=1
        )  # (batch_size, hidden_size, max_seq_len, kernel_size) 展开为滑动窗口
        x = einops.rearrange(x, "b d l k -> b l d k")  # 重排维度顺序

        kernels = self.kernel_generator(
            generator_input
        )  # (cu_seq_len, hidden_size * kernel_size) 生成动态卷积核
        kernels = einops.rearrange(
            kernels,
            "l (d k) -> l d k",
            d=self.hidden_size,
            k=self.kernel_size,
        )  # 重排卷积核形状
        kernels = self._seqs_to_batch(
            self._continuous_to_seqs(kernels, seq_lens=seq_lens)
        )  # (batch_size, max_seq_len, hidden_size, kernel_size) 转为批处理格式

        out = (x * kernels).sum(dim=-1)  # (batch_size, max_seq_len, hidden_size) 执行逐元素乘法并求和（卷积）

        out = self._batch_to_continuous(
            out, seq_lens=seq_lens
        )  # (cu_seq_len, hidden_size) 将批处理转回连续格式

        out = nn.functional.silu(out)  # 应用SiLU激活函数

        return out, new_conv_state  # 返回卷积输出和新的卷积状态

    def _batch_to_continuous(
        self,
        x: torch.Tensor,  # 批处理张量
        *,
        seq_lens: torch.Tensor,  # 序列长度
    ) -> torch.Tensor:
        """将批处理格式张量转换为连续格式。"""
        return torch.cat([x[i, -seq_lens[i] :] for i in range(seq_lens.size(0))])  # 按序列长度截取并拼接

    def _continuous_to_seqs(
        self,
        x: torch.Tensor,  # 连续格式张量
        *,
        seq_lens: torch.Tensor,  # 序列长度
    ) -> list[torch.Tensor]:
        """将连续格式张量拆分为序列列表。"""
        return [
            x[(seq_lens[:i].sum()) : (seq_lens[: i + 1].sum())]  # 按序列长度切片
            for i in range(seq_lens.size(0))  # 遍历每个批次
        ]

    def _seqs_to_batch(
        self,
        seqs: list[torch.Tensor],  # 序列列表
    ) -> torch.Tensor:
        """将序列列表转换为填充后的批处理张量。"""
        return nn.utils.rnn.pad_sequence(  # 使用RNN填充函数
            seqs,  # 序列列表
            batch_first=True,  # 批次维度在前
            padding_side="left",  # 左侧填充
        )


class JetBlock(nn.Module):
    """JetBlock线性注意力块，使用门控delta规则和动态短卷积。"""
    def __init__(
        self,
        config: JetNemotronConfig,  # Jet Nemotron配置
        layer_id: int,  # 层ID
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        jet_block_config = JetBlockConfig(  # 创建JetBlock配置
            **self.config.efficient_attention_config[self.config.layer_types[layer_id]]  # 根据层类型获取配置
        )

        hidden_size = self.config.hidden_size  # 隐藏层大小
        num_heads = jet_block_config.num_heads  # 注意力头数
        head_k_dim = jet_block_config.head_dim  # 每个头的键维度
        total_k_dim = num_heads * head_k_dim  # 总键维度
        head_v_dim = int(head_k_dim * jet_block_config.expand_v)  # 每个头的值维度
        total_v_dim = num_heads * head_v_dim  # 总值维度
        conv_size = jet_block_config.conv_size  # 卷积核大小

        self.qkvabz_proj = MergedColumnParallelLinear(  # 合并列并行线性投影（q,k,v,a,beta,z）
            hidden_size,  # 输入维度
            [  # 各分片输出维度
                total_k_dim,  # q维度
                total_k_dim,  # k维度
                total_v_dim,  # v维度
                num_heads,  # a维度
                num_heads,  # beta维度
                total_v_dim,  # z维度
            ],
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkvabz_proj", prefix),  # 参数前缀
        )

        self.o_proj = RowParallelLinear(total_v_dim, hidden_size, bias=False)  # 输出投影层

        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))  # A的对数参数（衰减率）
        self.dt_bias = nn.Parameter(torch.empty(num_heads))  # dt偏置参数

        self.dynamic_conv1d = DynamicShortConvolution(  # 动态短卷积模块
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("dynamic_conv1d", prefix),  # 参数前缀
            hidden_size=total_v_dim,  # 隐藏维度
            kernel_size=conv_size,  # 卷积核大小
            generator_input_size=hidden_size,  # 生成器输入维度
            generator_reduction=jet_block_config.dconv_generator_reduction,  # 生成器缩减因子
        )

        self.o_norm = RMSNormGated(  # 门控RMS归一化
            head_v_dim,  # 归一化维度
            eps=float(jet_block_config.norm_eps),  # epsilon值
        )

        # Attributes.  # 属性
        self.conv_size = conv_size  # 卷积核大小
        self.head_k_dim = head_k_dim  # 每头键维度
        self.head_v_dim = head_v_dim  # 每头值维度
        self.layer_id = layer_id  # 层ID
        self.num_heads = num_heads  # 注意力头数
        self.total_k_dim = total_k_dim  # 总键维度
        self.total_v_dim = total_v_dim  # 总值维度

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        """前向传播：执行JetBlock线性注意力计算。"""
        assert isinstance(get_attn_backend(), HybridLinearAttnBackend)  # 确认使用混合线性注意力后端
        assert isinstance(get_attn_backend().linear_attn_backend, MambaAttnBackendBase)  # 确认线性注意力后端类型
        linear_attn_backend = get_attn_backend().linear_attn_backend  # 获取线性注意力后端
        forward_metadata = linear_attn_backend.forward_metadata  # 获取前向元数据
        layer_cache = linear_attn_backend.req_to_token_pool.mamba2_layer_cache(  # 获取层缓存
            self.layer_id  # 当前层ID
        )

        qkvabz, _ = self.qkvabz_proj(hidden_states)  # 投影得到q,k,v,a,beta,z
        q, k, v, a, beta, z = qkvabz.split(  # 拆分为各分量
            [  # 各分量维度
                self.total_k_dim,  # q维度
                self.total_k_dim,  # k维度
                self.total_v_dim,  # v维度
                self.num_heads,  # a维度
                self.num_heads,  # beta维度
                self.total_v_dim,  # z维度
            ],
            dim=-1,  # 在最后一维拆分
        )

        q = nn.functional.silu(q)  # 对q应用SiLU激活
        q = einops.rearrange(q, "l (h d) -> l h d", h=self.num_heads, d=self.head_k_dim)  # 重排q为多头格式

        k = nn.functional.silu(k)  # 对k应用SiLU激活
        k = einops.rearrange(k, "l (h d) -> l h d", h=self.num_heads, d=self.head_k_dim)  # 重排k为多头格式

        conv_cache = layer_cache.conv  # 获取卷积缓存
        assert isinstance(conv_cache, torch.Tensor)  # 确认卷积缓存是张量
        v, new_conv_state = self.dynamic_conv1d(  # 对v执行动态短卷积
            v,  # 值向量
            conv_state=conv_cache[  # 卷积状态
                forward_metadata.mamba_cache_indices, -self.total_v_dim :, :  # 根据缓存索引取对应状态
            ],
            generator_input=hidden_states,  # 卷积核生成器输入
            seq_lens=(  # 序列长度
                forward_batch.extend_seq_lens  # 扩展序列长度
                if forward_batch.extend_seq_lens is not None  # 如果存在
                else torch.ones(  # 否则使用全1
                    (forward_batch.batch_size,),  # 批次大小
                    dtype=torch.long,  # 长整型
                )
            ),
        )
        conv_cache[forward_metadata.mamba_cache_indices, -self.total_v_dim :, :] = (
            new_conv_state  # 更新卷积缓存
        )
        v = einops.rearrange(v, "l (h d) -> l h d", h=self.num_heads, d=self.head_v_dim)  # 重排v为多头格式

        g = -self.A_log.float().exp() * nn.functional.softplus(a.float() + self.dt_bias)  # 计算衰减因子g

        beta = nn.functional.sigmoid(beta)  # 对beta应用sigmoid

        o = fused_recurrent_gated_delta_rule_update(  # 执行融合递归门控delta规则更新
            q=q.unsqueeze(0),  # q增加batch维度
            k=k.unsqueeze(0),  # k增加batch维度
            v=v.unsqueeze(0),  # v增加batch维度
            g=g.unsqueeze(0),  # g增加batch维度
            beta=beta.unsqueeze(0),  # beta增加batch维度
            initial_state_source=layer_cache.temporal,  # 初始时序状态
            initial_state_indices=forward_metadata.mamba_cache_indices,  # 初始状态索引
            cu_seqlens=cast(torch.LongTensor, forward_metadata.query_start_loc),  # 累计序列长度
            use_qk_l2norm_in_kernel=True,  # 在内核中使用qk的L2归一化
        ).squeeze(0)  # 去除batch维度

        z = einops.rearrange(z, "l (h d) -> l h d", h=self.num_heads)  # 重排z为多头格式

        o = self.o_norm(o, z)  # 应用门控RMS归一化

        o = einops.rearrange(o, "l h d -> l (h d)")  # 将多头输出展平

        o, _ = self.o_proj(o)  # 通过输出投影层

        return o  # 返回输出


class JetNemotronAttention(nn.Module):
    """Jet Nemotron标准注意力模块，支持全注意力和滑动窗口注意力。"""
    def __init__(
        self,
        config: JetNemotronConfig,  # Jet Nemotron配置
        layer_id: int,  # 层ID
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        self.head_dim = self.config.hidden_size // self.config.num_attention_heads  # 计算每个头的维度

        self.q_size = self.config.num_attention_heads * self.head_dim  # Q的总维度
        self.kv_size = self.config.num_key_value_heads * self.head_dim  # KV的总维度

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            self.config.hidden_size,  # 输入维度
            self.head_dim,  # 每个头的维度
            self.config.num_attention_heads,  # Q头数
            self.config.num_key_value_heads,  # KV头数
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影层
            self.config.num_attention_heads * self.head_dim,  # 输入维度
            self.config.hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 创建旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=self.config.max_position_embeddings,  # 最大位置数
            base=int(self.config.rope_parameters["rope_theta"]),  # RoPE基频
            rope_scaling=self.config.rope_parameters,  # RoPE缩放参数
        )

        match self.config.layer_types[layer_id]:  # 根据层类型设置滑动窗口大小
            case "attn":  # 全注意力层
                sliding_window_size = -1  # 不使用滑动窗口

            case "swa":  # 滑动窗口注意力层
                sliding_window_size = self.config.efficient_attention_config["swa"][  # 获取滑动窗口大小
                    "window_size"
                ]

            case _:  # 其他类型
                raise NotImplementedError  # 不支持的层类型

        self.attn = RadixAttention(  # 创建基数注意力模块
            self.config.num_attention_heads,  # 注意力头数
            self.head_dim,  # 头维度
            self.head_dim**-0.5,  # 缩放因子
            num_kv_heads=self.config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            sliding_window_size=sliding_window_size,  # 滑动窗口大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        """前向传播：执行标准自注意力计算。"""
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出


class JetNemotronDecoderLayer(nn.Module):
    """Jet Nemotron解码器层，根据层类型选择使用标准注意力或JetBlock。"""
    def __init__(
        self,
        config: JetNemotronConfig,  # Jet Nemotron配置
        alt_stream: torch.cuda.Stream | None = None,  # 备用CUDA流
        layer_id: int = 0,  # 层ID
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        match config.layer_types[layer_id]:  # 根据层类型选择注意力模块
            case "attn" | "swa":  # 标准注意力或滑动窗口注意力
                self.self_attn = JetNemotronAttention(  # 使用标准注意力模块
                    config,  # 配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("self_attn", prefix),  # 参数前缀
                    layer_id=layer_id,  # 层ID
                )

            case "jet":  # Jet线性注意力块
                self.self_attn = JetBlock(  # 使用JetBlock
                    config,  # 配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("self_attn", prefix),  # 参数前缀
                    layer_id=layer_id,  # 层ID
                )

            case _:  # 其他类型
                raise NotImplementedError  # 不支持的层类型

        self.mlp = Qwen2MLP(  # MLP模块（复用Qwen2的MLP）
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 隐藏层激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏层大小和epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: torch.Tensor | None,  # 残差连接
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向传播：执行自注意力 + MLP的解码器层计算。"""
        # Self Attention  # 自注意力
        residual = hidden_states  # 保存残差

        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化

        hidden_states = self.self_attn(  # 执行自注意力
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 归一化后的隐藏状态
            forward_batch=forward_batch,  # 前向批次信息
        )

        hidden_states = residual + hidden_states  # 残差连接

        # Fully Connected  # 全连接层
        residual = hidden_states  # 保存残差

        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化

        hidden_states = self.mlp(hidden_states)  # 通过MLP

        hidden_states = residual + hidden_states  # 残差连接

        return hidden_states, None  # 返回隐藏状态和None（无额外残差）


class JetNemotronForCausalLM(nn.Module):
    """Jet Nemotron因果语言模型，结合线性注意力块和标准注意力的混合架构。"""
    def __init__(
        self,
        config: JetNemotronConfig,  # Jet Nemotron配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = Qwen2Model(  # 创建模型主体（复用Qwen2结构）
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 参数前缀
            decoder_layer_type=JetNemotronDecoderLayer,  # 使用JetNemotron解码器层
        )

        if config.tie_word_embeddings:  # 如果共享词嵌入
            self.lm_head = self.model.embed_tokens  # 语言模型头共享嵌入层
        else:  # 否则
            self.lm_head = ParallelLMHead(  # 创建独立的语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
            )

        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.pooler = Pooler(PoolingType.LAST, normalize=True)  # 池化层（取最后一个token并归一化）

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor | None = None,  # 输入嵌入（可选）
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> EmbeddingPoolerOutput | LogitsProcessorOutput:
        """前向传播：执行因果语言模型推理。"""
        hidden_states = self.model(  # 通过模型主体获取隐藏状态
            input_ids,  # 输入token ID
            positions,  # 位置编码
            forward_batch,  # 前向批次信息
            input_embeds,  # 输入嵌入
        )

        if not get_embedding:  # 如果不需要获取嵌入
            return self.logits_processor(  # 返回logits处理结果
                input_ids, hidden_states, self.lm_head, forward_batch  # 输入ID、隐藏状态、语言模型头、批次信息
            )
        else:  # 否则
            return self.pooler(hidden_states, forward_batch)  # 返回池化嵌入结果

    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层。"""
        return self.model.embed_tokens  # 返回嵌入层

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """加载模型权重，支持堆叠参数映射。"""
        stacked_params_mapping: list[tuple[str, str, str | int]] = [  # 堆叠参数映射表
            # (param_name, shard_weight_name, shard_id)  # (参数名, 分片权重名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # QKV投影中的Q
            ("qkv_proj", "k_proj", "k"),  # QKV投影中的K
            ("qkv_proj", "v_proj", "v"),  # QKV投影中的V
            ("gate_up_proj", "gate_proj", 0),  # gate_up投影中的gate
            ("gate_up_proj", "up_proj", 1),  # gate_up投影中的up
            ("qkvabz_proj", "q_proj", 0),  # qkvabz投影中的Q
            ("qkvabz_proj", "k_proj", 1),  # qkvabz投影中的K
            ("qkvabz_proj", "v_proj", 2),  # qkvabz投影中的V
            ("qkvabz_proj", "a_proj", 3),  # qkvabz投影中的A
            ("qkvabz_proj", "b_proj", 4),  # qkvabz投影中的B
            ("qkvabz_proj", "g_proj", 5),  # qkvabz投影中的G
        ]

        params_dict = dict(self.named_parameters())  # 获取参数字典
        for weight_name, loaded_weight in weights:  # 遍历权重
            # Handle stacked parameters first.  # 优先处理堆叠参数
            for (
                param_name_part,
                shard_weight_name_part,
                shard_id,
            ) in stacked_params_mapping:  # 遍历堆叠参数映射
                if shard_weight_name_part not in weight_name.split("."):  # 如果分片名不在权重名中
                    continue  # 跳过

                param_name = weight_name.replace(  # 替换权重名中的分片名
                    shard_weight_name_part, param_name_part  # 替换为目标参数名
                )

                if param_name not in params_dict:  # 如果参数名不在参数字典中
                    # Fall back to direct match if no such stacked parameter.  # 如果没有堆叠参数则回退到直接匹配
                    continue  # 跳过

                param = params_dict[param_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader")  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环

            else:  # 如果没有匹配到堆叠参数
                param_name = weight_name  # 直接使用权重名

                param = params_dict[param_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = JetNemotronForCausalLM  # 入口类：Jet Nemotron因果语言模型
