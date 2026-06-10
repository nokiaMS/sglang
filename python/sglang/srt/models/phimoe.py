# Phi-4多模态模型MoE（混合专家）推理实现文件
# 本文件实现了PhiMoE混合专家模型的推理架构
# 包含配置类、稀疏混合器、MoE模块、注意力层、解码器层及因果语言模型等组件

from typing import Iterable, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers.configuration_utils import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size  # 导入注意力TP工具
from sglang.srt.layers.linear import (  # 导入线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE
from sglang.srt.layers.moe.topk import TopK  # 导入TopK选择
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    DEFAULT_VOCAB_PADDING_SIZE,  # 默认词表填充大小
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称
)
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加和层创建工具


class PhiMoEConfig(PretrainedConfig):  # PhiMoE配置类，继承自预训练配置

    model_type = "phimoe"  # 模型类型

    def __init__(  # 初始化函数
        self,
        vocab_size=32000,  # 词表大小
        hidden_size=4096,  # 隐藏层大小
        intermediate_size=14336,  # 中间层大小
        num_hidden_layers=32,  # 隐藏层数量
        num_attention_heads=32,  # 注意力头数
        num_key_value_heads=8,  # KV头数
        head_dim=None,  # 每个头的维度
        hidden_act="silu",  # 隐藏层激活函数
        max_position_embeddings=4096 * 32,  # 最大位置嵌入数
        initializer_range=0.02,  # 初始化范围
        rms_norm_eps=1e-5,  # RMS归一化eps
        use_cache=True,  # 是否使用缓存
        pad_token_id=None,  # 填充token ID
        bos_token_id=1,  # 开始token ID
        eos_token_id=2,  # 结束token ID
        tie_word_embeddings=False,  # 是否绑定词嵌入
        rope_theta=1e6,  # RoPE theta
        sliding_window=None,  # 滑动窗口
        attention_dropout=0.0,  # 注意力dropout
        num_experts_per_tok=2,  # 每个token的专家数
        num_local_experts=16,  # 本地专家数
        output_router_logits=False,  # 是否输出路由logits
        router_aux_loss_coef=0.001,  # 路由辅助损失系数
        router_jitter_noise=0.0,  # 路由抖动噪声
        attention_bias=False,  # 注意力偏置
        lm_head_bias=False,  # 语言模型头偏置
        **kwargs,  # 其他关键字参数
    ):
        self.vocab_size = vocab_size  # 词表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.hidden_size = hidden_size  # 隐藏层大小
        self.intermediate_size = intermediate_size  # 中间层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.sliding_window = sliding_window  # 滑动窗口
        self.attention_bias = attention_bias  # 注意力偏置
        self.lm_head_bias = lm_head_bias  # 语言模型头偏置
        # for backward compatibility  # 向后兼容
        if num_key_value_heads is None:  # 如果KV头数为空
            num_key_value_heads = num_attention_heads  # 默认与注意力头数相同
        if head_dim is None:  # 如果头维度为空
            head_dim = hidden_size // num_attention_heads  # 计算头维度

        self.num_key_value_heads = num_key_value_heads  # KV头数
        self.head_dim = head_dim  # 头维度
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.initializer_range = initializer_range  # 初始化范围
        self.rms_norm_eps = rms_norm_eps  # RMS归一化eps
        self.use_cache = use_cache  # 是否使用缓存
        self.rope_theta = rope_theta  # RoPE theta
        self.attention_dropout = attention_dropout  # 注意力dropout

        self.num_experts_per_tok = num_experts_per_tok  # 每个token的专家数
        self.num_local_experts = num_local_experts  # 本地专家数
        self.output_router_logits = output_router_logits  # 是否输出路由logits
        self.router_aux_loss_coef = router_aux_loss_coef  # 路由辅助损失系数
        self.router_jitter_noise = router_jitter_noise  # 路由抖动噪声
        super().__init__(  # 调用父类初始化
            pad_token_id=pad_token_id,  # 填充token ID
            bos_token_id=bos_token_id,  # 开始token ID
            eos_token_id=eos_token_id,  # 结束token ID
            tie_word_embeddings=tie_word_embeddings,  # 是否绑定词嵌入
            **kwargs,  # 其他参数
        )


def sparsemixer(scores, jitter_eps=0.01):  # 稀疏混合器函数，选择top-2专家
    ################ Select first expert (topk=2) ################  # 选择第一个专家

    # compute mask for sparsity  # 计算稀疏掩码
    mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)  # 获取最大值和索引
    factor = scores.abs().clamp(min=mask_logits_threshold)  # 计算缩放因子
    mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (  # 计算掩码阈值
        2 * jitter_eps  # 抖动阈值
    )

    # apply mask  # 应用掩码
    masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))  # 掩码门控值
    selected_experts = max_ind  # 选择的专家

    # compute scores for gradients  # 计算梯度分数
    masked_gates = torch.softmax(masked_gates, dim=-1)  # softmax归一化
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)  # 收集选择的专家分数

    multiplier = multiplier_o  # 乘数

    # masked out first expert  # 屏蔽第一个专家
    masked_scores = torch.scatter(  # 散射操作屏蔽第一个专家
        scores,  # 分数
        -1,  # 最后一维
        selected_experts,  # 选择的专家索引
        float("-inf"),  # 屏蔽值
    )

    ################ Select second expert (topk=2) ################  # 选择第二个专家
    # compute mask for sparsity  # 计算稀疏掩码
    mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)  # 获取最大值和索引
    factor = scores.abs().clamp(min=mask_logits_threshold)  # 计算缩放因子
    mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (  # 计算掩码阈值
        2 * jitter_eps  # 抖动阈值
    )

    # apply mask  # 应用掩码
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))  # 掩码门控值
    selected_experts_top2 = max_ind  # 选择的第二个专家
    # compute scores for gradients  # 计算梯度分数
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)  # softmax归一化
    multiplier_top2 = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)  # 收集分数

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)  # 拼接两个专家的乘数
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)  # 拼接专家索引

    return (  # 返回乘数和专家索引
        multiplier,  # 乘数
        selected_experts,  # 专家索引
    )


def phimoe_routing_function(  # PhiMoE路由函数
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    topk: int,  # topk值
    renormalize: bool,  # 是否重归一化
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言token数匹配
    assert topk == 2, "Only top-2 routing is supported"  # 断言仅支持top-2路由
    assert renormalize is False, "Renormalization is not supported"  # 断言不支持重归一化

    topk_weights, topk_ids = sparsemixer(gating_output)  # 通过稀疏混合器获取权重和ID
    return topk_weights, topk_ids  # 返回topk权重和ID


class PhiMoE(nn.Module):  # PhiMoE混合专家模块
    """A tensor-parallel MoE implementation for PhiMoE that shards each expert  # 张量并行的MoE实现，每个专家分片
    across all ranks.  # 跨所有秩

    Each expert's weights are sharded across all ranks and a fused MoE  # 每个专家的权重在所有秩上分片
    kernel is used for the forward pass, and finally we reduce the outputs  # 使用融合MoE核进行前向传播
    across ranks.  # 最后跨秩归约输出
    """

    def __init__(  # 初始化函数
        self,
        num_experts: int,  # 专家数量
        top_k: int,  # topk值
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小

        # Gate always runs at half / full precision for now.  # 门控目前以半精度/全精度运行
        self.gate = ReplicatedLinear(  # 门控线性层
            hidden_size,  # 输入大小
            num_experts,  # 输出大小（专家数）
            bias=False,  # 不使用偏置
            quant_config=None,  # 不量化门控
        )

        self.topk = TopK(  # TopK选择模块
            top_k=top_k,  # topk值
            renormalize=False,  # 不重归一化
            custom_routing_function=phimoe_routing_function,  # 自定义路由函数
        )

        self.experts = FusedMoE(  # 融合MoE专家模块
            num_experts=num_experts,  # 专家数量
            top_k=top_k,  # topk值
            layer_id=layer_id,  # 层ID
            hidden_size=hidden_size,  # 隐藏层大小
            intermediate_size=intermediate_size,  # 中间层大小
            reduce_results=True,  # 归约结果
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("experts", prefix),  # 添加前缀
        )

    def forward(  # 前向传播函数，执行MoE计算
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None  # 隐藏状态和前向批次
    ) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.  # 注意：隐藏状态可以是1D或2D形状
        orig_shape = hidden_states.shape  # 保存原始形状
        hidden_states = hidden_states.view(-1, self.hidden_size)  # 重塑为2D
        router_logits, _ = self.gate(hidden_states)  # 通过门控
        topk_output = self.topk(hidden_states, router_logits)  # TopK选择
        final_hidden_states = self.experts(hidden_states, topk_output)  # 通过专家
        return final_hidden_states.view(orig_shape)  # 恢复原始形状并返回


class PhiMoEAttention(nn.Module):  # PhiMoE注意力模块

    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        head_dim: Optional[int] = None,  # 每个头的维度
        max_position: int = 4096 * 32,  # 最大位置
        rope_theta: float = 10000,  # RoPE theta
        layer_id: int = 0,  # 层ID
        attention_bias: bool = False,  # 注意力偏置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        rope_scaling: Optional[dict] = None,  # RoPE缩放
        prefix: str = "",  # 前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小

        attn_tp_rank = get_attention_tp_rank()  # 获取注意力TP秩
        attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小

        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % attn_tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 每个TP秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= attn_tp_size:  # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，进行分区
            # the KV heads across multiple tensor parallel GPUs.  # 跨多个TP GPU分配KV头
            assert self.total_num_kv_heads % attn_tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，进行复制
            # the KV heads across multiple tensor parallel GPUs.  # 跨多个TP GPU复制KV头
            assert attn_tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 每个TP秩的KV头数
        if head_dim is None:  # 如果头维度为空
            head_dim = hidden_size // num_heads  # 计算头维度
        self.head_dim = head_dim  # 保存头维度

        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 保存RoPE theta
        self.rope_scaling = rope_scaling  # 保存RoPE缩放

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影层
            hidden_size,  # 输入大小
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=attention_bias,  # 注意力偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # TP秩
            tp_size=attn_tp_size,  # TP大小
            prefix=add_prefix("qkv_proj", prefix),  # 添加前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影行并行线性层
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=attention_bias,  # 注意力偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # TP秩
            tp_size=attn_tp_size,  # TP大小
            prefix=add_prefix("o_proj", prefix),  # 添加前缀
        )
        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position,  # 最大位置
            base=int(self.rope_theta),  # 基础theta
            rope_scaling=self.rope_scaling,  # RoPE缩放
        )
        self.attn = RadixAttention(  # 基数注意力模块
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
        )

    def forward(  # 前向传播函数，执行注意力计算
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割QKV
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class PhiMoEDecoderLayer(nn.Module):  # PhiMoE解码器层

    def __init__(  # 初始化函数
        self,
        config: PhiMoEConfig,  # PhiMoE配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        rope_theta = config.rope_parameters["rope_theta"]  # RoPE theta
        self.self_attn = PhiMoEAttention(  # 自注意力模块
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            max_position=config.max_position_embeddings,  # 最大位置
            num_kv_heads=config.num_key_value_heads,  # KV头数
            head_dim=getattr(  # 头维度
                config, "head_dim", self.hidden_size // config.num_attention_heads  # 默认计算
            ),
            rope_theta=rope_theta,  # RoPE theta
            layer_id=layer_id,  # 层ID
            attention_bias=config.attention_bias,  # 注意力偏置
            quant_config=quant_config,  # 量化配置
            rope_scaling=config.rope_parameters,  # RoPE缩放
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
        )
        self.block_sparse_moe = PhiMoE(  # 稀疏MoE模块
            num_experts=config.num_local_experts,  # 专家数量
            top_k=config.num_experts_per_tok,  # topk值
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("block_sparse_moe", prefix),  # 添加前缀
        )
        self.input_layernorm = nn.LayerNorm(  # 输入层归一化
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True  # 参数
        )
        self.post_attention_layernorm = nn.LayerNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True  # 参数
        )

    def forward(  # 前向传播函数，执行解码器层计算
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: Optional[torch.Tensor],  # 残差
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states  # 保存残差

        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化

        hidden_states = self.self_attn(  # 通过自注意力层
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = hidden_states + residual  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化
        hidden_states = self.block_sparse_moe(  # 通过稀疏MoE
            hidden_states, forward_batch=forward_batch  # 传入参数
        )

        hidden_states = hidden_states + residual  # 残差连接
        return hidden_states, residual  # 返回隐藏状态和残差


class PhiMoEModel(nn.Module):  # PhiMoE模型主体

    def __init__(  # 初始化函数
        self,
        config: PhiMoEConfig,  # PhiMoE配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.vocab_size = config.vocab_size  # 词表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词表嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("embed_tokens", prefix),  # 添加前缀
        )

        self.layers = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 隐藏层数量
            lambda idx, prefix: PhiMoEDecoderLayer(  # 解码器层构造函数
                config, int(prefix.split(".")[-1]), quant_config, prefix=prefix  # 传入参数
            ),
            prefix=add_prefix("layers", prefix),  # 添加前缀
        )
        self.norm = nn.LayerNorm(  # 最终归一化
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True  # 参数
        )

    def forward(  # 前向传播函数，执行模型主体计算
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
    ) -> Union[torch.Tensor]:
        if input_embeds is None:  # 如果没有输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词表嵌入层
        else:  # 否则
            hidden_states = input_embeds  # 使用输入嵌入
        residual = None  # 残差初始化为空

        for layer in self.layers:  # 遍历解码器层
            hidden_states, residual = layer(  # 通过当前层
                positions, hidden_states, residual, forward_batch=forward_batch  # 传入参数
            )

        hidden_states = self.norm(hidden_states)  # 应用最终归一化
        return hidden_states  # 返回隐藏状态


class PhiMoEForCausalLM(nn.Module):  # PhiMoE因果语言模型

    def __init__(  # 初始化函数
        self,
        config: PhiMoEConfig,  # PhiMoE配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):

        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = PhiMoEModel(  # 模型主体
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传入参数
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            org_num_embeddings=config.vocab_size,  # 原始嵌入数
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,  # 填充大小
            quant_config=quant_config,  # 量化配置
            bias=True,  # 使用偏置
            prefix=add_prefix("lm_head", prefix),  # 添加前缀
        )
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head.weight = self.model.embed_tokens.weight  # 绑定权重
        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 池化器

    @torch.no_grad()  # 不计算梯度
    def forward(  # 前向传播函数，执行因果语言模型计算
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)  # 通过模型主体

        if not get_embedding:  # 如果不获取嵌入
            return self.logits_processor(  # 返回logits处理器结果
                input_ids, hidden_states, self.lm_head, forward_batch  # 传入参数
            )

        else:  # 否则
            return self.pooler(hidden_states, forward_batch)  # 返回池化结果

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q投影映射
            ("qkv_proj", "k_proj", "k"),  # K投影映射
            ("qkv_proj", "v_proj", "v"),  # V投影映射
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
            ckpt_gate_proj_name="w1",  # 检查点门控投影名
            ckpt_down_proj_name="w2",  # 检查点下投影名
            ckpt_up_proj_name="w3",  # 检查点上投影名
            num_experts=self.config.num_local_experts,  # 专家数量
        )

        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 继续
                name = name.replace(weight_name, param_name)  # 替换权重名
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出循环
            else:  # 如果没有匹配的堆叠参数
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不在参数名中
                        continue  # 继续
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载权重
                        param,  # 参数
                        loaded_weight,  # 加载的权重
                        name,  # 名称
                        shard_id=shard_id,  # 分片ID
                        expert_id=expert_id,  # 专家ID
                    )
                    break  # 跳出循环
                else:  # 如果没有匹配的专家参数
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在字典中
                        continue  # 跳过
                    # Remapping the name of FP8 kv-scale.  # 重映射FP8 KV缩放名称
                    name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射名称
                    if name is None:  # 如果重映射后为空
                        continue  # 跳过

                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重


EntryClass = PhiMoEForCausalLM  # 入口类为PhiMoEForCausalLM
