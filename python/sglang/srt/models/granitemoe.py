# GraniteMoe模型推理实现 - 基于稀疏专家混合(MoE)架构的Granite模型，仅用于推理
"""Inference-only GraniteMoe model."""  # 仅推理的GraniteMoe模型

from typing import Iterable, Optional  # 类型提示导入

import torch  # PyTorch深度学习框架
from torch import nn  # 神经网络模块
from transformers import GraniteConfig  # Granite模型配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 获取张量并行世界大小
from sglang.srt.layers.layernorm import RMSNorm  # RMS归一化层
from sglang.srt.layers.linear import (  # 线性层导入
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # logits处理器
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 融合MoE Triton内核
from sglang.srt.layers.moe.topk import TopK  # Top-K选择模块
from sglang.srt.layers.pooler import Pooler, PoolingType  # 池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力机制
from sglang.srt.layers.rotary_embedding import get_rope  # 获取旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息
from sglang.srt.models import mixtral  # Mixtral模型（用于权重加载）
from sglang.srt.utils import add_prefix  # 添加前缀工具


class GraniteMoeMoE(nn.Module):  # GraniteMoe稀疏专家混合模块
    """A tensor-parallel MoE implementation for GraniteMoe that shards each
    expert across all ranks.
    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """  # GraniteMoe的张量并行MoE实现，将每个专家分片到所有rank上。
       # 每个专家的权重在所有rank上分片，使用融合MoE内核进行前向传播，最后跨rank归约输出。

    def __init__(  # 初始化GraniteMoeMoE
        self,
        num_experts: int,  # 专家数量
        top_k: int,  # 每个token选择的top-k专家数
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        layer_id: int,  # 层ID
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        tp_size: Optional[int] = None,  # 张量并行大小
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小

        # Gate always runs at half / full precision for now.  # 门控始终以半精度/全精度运行
        self.gate = ReplicatedLinear(  # 创建门控线性层（路由器）
            hidden_size,  # 输入大小
            num_experts,  # 输出大小（专家数量）
            bias=False,  # 不使用偏置
            params_dtype=params_dtype,  # 参数数据类型
            quant_config=None,  # 门控不使用量化
            prefix=f"{prefix}.gate",  # 参数前缀
        )

        self.topk = TopK(  # 创建Top-K选择模块
            top_k=top_k,  # 选择top-k个专家
            renormalize=True,  # 对权重重新归一化
        )

        self.experts = FusedMoE(  # 创建融合MoE专家层
            num_experts=num_experts,  # 专家数量
            top_k=top_k,  # top-k值
            hidden_size=hidden_size,  # 隐藏层大小
            intermediate_size=intermediate_size,  # 中间层大小
            layer_id=layer_id,  # 层ID
            params_dtype=params_dtype,  # 参数数据类型
            reduce_results=True,  # 归约结果
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.experts",  # 参数前缀
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # MoE前向传播
        # NOTE: hidden_states can have either 1D or 2D shape.  # 注意：hidden_states可以是1D或2D形状
        orig_shape = hidden_states.shape  # 保存原始形状
        hidden_states = hidden_states.view(-1, self.hidden_size)  # 重塑为2D
        router_logits, _ = self.gate(hidden_states)  # 计算路由器logits
        topk_output = self.topk(hidden_states, router_logits)  # 获取top-k选择结果
        final_hidden_states = self.experts(hidden_states, topk_output)  # 执行专家计算
        return final_hidden_states.view(orig_shape)  # 恢复原始形状并返回


class GraniteMoeAttention(nn.Module):  # GraniteMoe注意力模块

    def __init__(  # 初始化GraniteMoeAttention
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数量
        num_kv_heads: int,  # KV头数量
        max_position: int = 4096 * 32,  # 最大位置编码长度
        layer_id: int = 0,  # 层ID
        rope_theta: float = 10000,  # RoPE基数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        attention_multiplier: Optional[float] = None,  # 注意力乘数
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
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.  # KV头数大于TP大小，因此在多个张量并行GPU上划分KV头
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.  # KV头数小于TP大小，因此在多个张量并行GPU上复制KV头
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个rank的KV头数
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = (  # 注意力缩放因子
            attention_multiplier  # 使用自定义注意力乘数
            if attention_multiplier is not None  # 如果提供了注意力乘数
            else self.head_dim**-1  # 否则使用头维度的倒数
        )
        self.rope_theta = rope_theta  # 保存RoPE基数

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
            max_position=max_position,  # 最大位置
            base=int(self.rope_theta),  # RoPE基数
            is_neox_style=True,  # 使用Neox风格
        )
        self.attn = RadixAttention(  # 创建基数注意力层
            self.num_heads,  # 注意力头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.attn",  # 参数前缀
        )

    def forward(  # 注意力前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出


class GraniteMoeDecoderLayer(nn.Module):  # GraniteMoe解码器层

    def __init__(  # 初始化解码器层
        self,
        config: GraniteConfig,  # 模型配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        rope_theta = config.rope_parameters["rope_theta"]  # 获取RoPE基数
        self.self_attn = GraniteMoeAttention(  # 创建自注意力层
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            max_position=config.max_position_embeddings,  # 最大位置编码
            num_kv_heads=config.num_key_value_heads,  # KV头数
            rope_theta=rope_theta,  # RoPE基数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.self_attn",  # 参数前缀
            attention_multiplier=config.attention_multiplier,  # 注意力乘数
        )
        self.block_sparse_moe = GraniteMoeMoE(  # 创建稀疏MoE层
            num_experts=config.num_local_experts,  # 专家数量
            top_k=config.num_experts_per_tok,  # 每个token的top-k专家数
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.block_sparse_moe",  # 参数前缀
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.residual_multiplier = config.residual_multiplier  # 残差乘数

    def forward(  # 解码器层前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        # Self Attention  # 自注意力
        hidden_states = self.self_attn(  # 执行自注意力计算
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 归一化后的隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = residual + hidden_states * self.residual_multiplier  # 残差连接（带乘数）
        residual = hidden_states  # 更新残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化
        hidden_states = self.block_sparse_moe(hidden_states)  # 执行MoE计算
        hidden_states = residual + hidden_states * self.residual_multiplier  # 残差连接（带乘数）

        return hidden_states  # 返回隐藏状态


class GraniteMoeModel(nn.Module):  # GraniteMoe模型主体

    def __init__(  # 初始化模型
        self,
        config: GraniteConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.embed_tokens = VocabParallelEmbedding(  # 创建词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            org_num_embeddings=config.vocab_size,  # 原始嵌入数量
        )
        self.embedding_multiplier = config.embedding_multiplier  # 嵌入乘数

        self.layers = nn.ModuleList(  # 创建解码器层列表
            [
                GraniteMoeDecoderLayer(  # 每一层都是GraniteMoe解码器层
                    config,  # 模型配置
                    i,  # 层ID
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 参数前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.embed_tokens(input_ids)  # 返回词嵌入结果

    def forward(  # 模型前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        if inputs_embeds is not None:  # 如果提供了输入嵌入
            hidden_states = inputs_embeds  # 直接使用输入嵌入
        else:  # 否则
            hidden_states = self.get_input_embeddings(input_ids)  # 从token ID获取嵌入
        hidden_states *= self.embedding_multiplier  # 应用嵌入乘数

        for i in range(len(self.layers)):  # 遍历所有解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states = layer(  # 执行当前层前向传播
                positions,  # 位置编码
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
            )
        hidden_states = self.norm(hidden_states)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class GraniteMoeForCausalLM(nn.Module):  # GraniteMoe因果语言模型

    def __init__(  # 初始化因果语言模型
        self,
        config: GraniteConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = GraniteMoeModel(  # 创建模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 创建语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 参数前缀
        )
        if config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head.weight = self.model.embed_tokens.weight  # 共享权重
        # Granite logit scaling factors are applied via division, but
        # LogitsProcessor expects a multiplicative factor.  # Granite logits缩放因子通过除法应用，但LogitsProcessor期望乘法因子
        if hasattr(config, "logits_scaling"):  # 如果配置有logits缩放
            logit_scale = 1.0 / config.logits_scaling  # 转换为乘法因子
        else:  # 否则
            logit_scale = None  # 不使用缩放
        self.logits_processor = LogitsProcessor(config, logit_scale=logit_scale)  # 创建logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 创建池化层

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 因果语言模型前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 模型前向传播
        if not get_embedding:  # 如果不获取嵌入
            logits_processor_output: LogitsProcessorOutput = self.logits_processor(  # 处理logits
                input_ids, hidden_states, self.lm_head, forward_batch
            )
            return logits_processor_output  # 返回logits处理结果
        else:  # 否则获取嵌入
            return self.pooler(hidden_states, forward_batch)  # 返回池化结果

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:  # 加载模型权重
        new_weights = {}  # 新权重字典
        for n, p in weights:  # 遍历权重
            if n.endswith(".block_sparse_moe.input_linear.weight"):  # 如果是MoE输入线性层权重
                for e in range(p.size(0)):  # 遍历每个专家
                    w1_name = n.replace(  # 构造w1权重名
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w1.weight",
                    )
                    w3_name = n.replace(  # 构造w3权重名
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w3.weight",
                    )
                    w1_param, w3_param = p[e].chunk(2, dim=0)  # 沿dim=0拆分为w1和w3
                    assert w1_name not in new_weights  # 断言w1名称不存在
                    assert w3_name not in new_weights  # 断言w3名称不存在
                    new_weights[w1_name] = w1_param  # 存入w1权重
                    new_weights[w3_name] = w3_param  # 存入w3权重
            elif n.endswith(".block_sparse_moe.output_linear.weight"):  # 如果是MoE输出线性层权重
                for e in range(p.size(0)):  # 遍历每个专家
                    w2_name = n.replace(  # 构造w2权重名
                        ".block_sparse_moe.output_linear.weight",
                        f".block_sparse_moe.experts.{e}.w2.weight",
                    )
                    w2_param = p[e]  # 获取当前专家的w2权重
                    assert w2_name not in new_weights  # 断言w2名称不存在
                    new_weights[w2_name] = w2_param  # 存入w2权重
            elif n.endswith(".block_sparse_moe.router.layer.weight"):  # 如果是路由器权重
                gate_name = n.replace(  # 构造门控权重名
                    ".block_sparse_moe.router.layer.weight",
                    ".block_sparse_moe.gate.weight",
                )
                assert gate_name not in new_weights  # 断言门控名称不存在
                new_weights[gate_name] = p  # 存入门控权重
            else:  # 其他权重
                new_weights[n] = p  # 直接存入
        mixtral.MixtralForCausalLM.load_weights(self, new_weights.items())  # 使用Mixtral的权重加载方法


EntryClass = [GraniteMoeForCausalLM]  # 入口类列表，用于模型注册
