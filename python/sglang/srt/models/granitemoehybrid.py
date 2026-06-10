# GraniteMoeHybrid模型推理实现 - 混合Mamba2+Attention+MoE架构的Granite模型，仅用于推理
from typing import Iterable, Optional  # 类型提示导入

import torch  # PyTorch深度学习框架
from torch import nn  # 神经网络模块
from transformers.models.granitemoeshared import GraniteMoeSharedConfig  # GraniteMoe共享配置类

from sglang.srt.configs.granitemoehybrid import GraniteMoeHybridConfig  # GraniteMoe混合配置类
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 分布式工具导入
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活函数与乘法
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # 混合线性注意力后端
    HybridLinearAttnBackend,  # 混合线性注意力后端
    Mamba2AttnBackend,  # Mamba2注意力后端
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2  # Mamba混合器
from sglang.srt.layers.layernorm import RMSNorm  # RMS归一化层
from sglang.srt.layers.linear import (  # 线性层导入
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力机制
from sglang.srt.layers.rotary_embedding import get_rope  # 获取旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer  # 流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 前向批次信息
from sglang.srt.model_executor.forward_context import get_attn_backend  # 获取注意力后端
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.models.transformers import maybe_prefix  # 可能添加前缀
from sglang.srt.utils import make_layers  # 创建层工具

from .granitemoe import GraniteMoeMoE  # 导入GraniteMoe的MoE模块


# in vLLM this is in a separate file, but keeping it here for decoupling  # 在vLLM中这在一个单独文件中，但为了解耦保留在这里
class GraniteMoeSharedMLP(nn.Module):  # GraniteMoe共享MLP模块
    def __init__(  # 初始化共享MLP
        self,
        config: GraniteMoeSharedConfig,  # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化

        self.input_size = config.hidden_size  # 输入大小
        self.hidden_size = config.shared_intermediate_size  # 共享中间层大小
        self.input_linear = MergedColumnParallelLinear(  # 创建输入线性层（gate+up合并）
            input_size=self.input_size,  # 输入大小
            output_sizes=[self.hidden_size] * 2,  # 输出大小（两个相同大小的分支）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.input_linear",  # 参数前缀
        )
        self.output_linear = RowParallelLinear(  # 创建输出线性层（down投影）
            self.hidden_size,  # 输入大小
            self.input_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.output_linear",  # 参数前缀
        )
        if config.hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出错误
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."  # 目前只支持silu
            )
        self.act_fn = SiluAndMul()  # 创建SiLU激活与乘法函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 共享MLP前向传播
        gate_up, _ = self.input_linear(hidden_states)  # 计算gate和up投影
        x = self.act_fn(gate_up)  # 应用激活函数
        x, _ = self.output_linear(x)  # 计算down投影
        return x  # 返回结果


class GraniteMoeHybridMambaDecoderLayer(nn.Module):  # GraniteMoe混合Mamba解码器层
    def __init__(  # 初始化Mamba解码器层
        self,
        config: GraniteMoeHybridConfig,  # 混合配置
        layer_idx: int,  # 层索引
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.layer_idx = layer_idx  # 保存层索引
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.residual_multiplier = config.residual_multiplier  # 残差乘数

        self.mamba = MambaMixer2(  # 创建Mamba2混合器
            cache_params=config.mamba2_cache_params,  # 缓存参数
            hidden_size=config.hidden_size,  # 隐藏层大小
            use_conv_bias=config.mamba_conv_bias,  # 是否使用卷积偏置
            use_bias=config.mamba_proj_bias,  # 是否使用投影偏置
            n_groups=config.mamba_n_groups,  # Mamba组数
            rms_norm_eps=config.rms_norm_eps,  # RMS归一化epsilon
            activation=config.hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.mixer",  # 参数前缀
        )

        self.block_sparse_moe = None  # 初始化MoE为None
        if getattr(config, "num_local_experts", 0) > 0:  # 如果配置有本地专家
            self.block_sparse_moe = GraniteMoeMoE(  # 创建稀疏MoE层
                num_experts=config.num_local_experts,  # 专家数量
                top_k=config.num_experts_per_tok,  # 每个token的top-k专家数
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=config.intermediate_size,  # 中间层大小
                layer_id=layer_idx,  # 层ID
                quant_config=quant_config,  # 量化配置
                tp_size=get_tensor_model_parallel_world_size(),  # 张量并行大小
                prefix=f"{prefix}.block_sparse_moe",  # 参数前缀
            )

        self.shared_mlp = (  # 创建共享MLP（如果配置了共享中间层大小）
            None  # 如果没有共享中间层则为None
            if getattr(config, "shared_intermediate_size", 0) == 0  # 检查共享中间层大小
            else GraniteMoeSharedMLP(  # 否则创建共享MLP
                config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # Mamba解码器层前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor | None,  # 残差
        forward_batch: ForwardBatch,  # 前向批次信息
    ):
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化

        output = torch.empty_like(hidden_states)  # 创建输出张量
        attn_backend = get_attn_backend()  # 获取当前注意力后端
        assert isinstance(attn_backend, HybridLinearAttnBackend)  # 断言是混合线性注意力后端
        assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)  # 断言线性注意力是Mamba2后端
        attn_backend.linear_attn_backend.forward(  # 执行Mamba2前向传播
            mixer=self.mamba,  # Mamba混合器
            layer_id=self.layer_idx,  # 层ID
            hidden_states=hidden_states,  # 隐藏状态
            output=output,  # 输出张量
            forward_batch=forward_batch,  # 前向批次
            use_triton_causal_conv=True,  # 使用Triton因果卷积
        )

        hidden_states = residual + output * self.residual_multiplier  # 残差连接（带乘数）

        residual = hidden_states  # 更新残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化
        if self.shared_mlp is None:  # 如果没有共享MLP
            if self.block_sparse_moe is not None:  # 如果有MoE
                hidden_states = self.block_sparse_moe(hidden_states)  # 执行MoE计算
            # else: skip  # 否则：跳过
        else:  # 如果有共享MLP
            # create a copy since block_sparse_moe modifies in-place  # 创建副本，因为block_sparse_moe会原地修改
            if self.block_sparse_moe is not None:  # 如果同时有MoE
                moe_hidden_states = hidden_states.clone()  # 克隆隐藏状态用于MoE
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)  # 执行MoE计算
                hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)  # MoE结果加上共享MLP结果
                del moe_hidden_states  # 删除临时变量
            else:  # 如果只有共享MLP
                hidden_states = self.shared_mlp(hidden_states)  # 执行共享MLP计算
        hidden_states = residual + hidden_states * self.residual_multiplier  # 残差连接（带乘数）

        return hidden_states, residual  # 返回隐藏状态和残差


class GraniteMoeHybridAttention(nn.Module):  # GraniteMoe混合注意力模块
    def __init__(  # 初始化混合注意力
        self,
        config: GraniteMoeHybridConfig,  # 混合配置
        layer_id: int,  # 层ID
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.causal = True  # 使用因果注意力
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.attention_bias = config.attention_bias  # 注意力偏置
        self.attention_multiplier = config.attention_multiplier  # 注意力乘数
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        self.head_dim = self.hidden_size // self.total_num_heads  # 头维度
        self.total_num_kv_heads = config.num_key_value_heads  # 总KV头数

        # TensorParallel logic  # 张量并行逻辑
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个rank的注意力头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.  # KV头数大于TP大小，因此在多个张量并行GPU上划分KV头
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.  # KV头数小于TP大小，因此在多个张量并行GPU上复制KV头
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_key_value_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个rank的KV头数

        self.qkv_proj = QKVParallelLinear(  # 创建QKV并行投影层
            self.hidden_size,  # 输入大小
            self.head_dim,  # 每个头的大小
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=self.attention_bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.qkv_proj",  # 参数前缀
        )

        self.o_proj = RowParallelLinear(  # 创建输出投影层
            self.hidden_size,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=self.attention_bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.o_proj",  # 参数前缀
        )

        if config.position_embedding_type == "rope":  # 如果使用RoPE位置编码

            self.rotary_emb = get_rope(  # 创建旋转位置编码
                head_size=self.head_dim,  # 头大小
                rotary_dim=self.head_dim,  # its not in the config  # 旋转维度（配置中未指定）
                max_position=config.max_position_embeddings,  # 最大位置编码
                base=config.rope_theta,  # RoPE基数
                rope_scaling=config.rope_scaling,  # RoPE缩放配置
            )
        else:  # 否则不使用旋转位置编码
            self.rotary_emb = None  # 设为None

        self.attn = RadixAttention(  # 创建基数注意力层
            num_heads=self.num_heads,  # 注意力头数
            head_dim=self.head_dim,  # 头维度
            scaling=self.attention_multiplier,  # 缩放因子
            num_kv_heads=self.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.attn",  # 参数前缀
        )

    def forward(  # 混合注意力前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch | None = None,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算QKV投影
        query, key, value = qkv.split(  # 拆分QKV
            [
                self.num_heads * self.head_dim,  # Q的大小
                self.num_key_value_heads * self.head_dim,  # K的大小
                self.num_key_value_heads * self.head_dim,  # V的大小
            ],
            dim=-1,  # 沿最后一维拆分
        )

        if self.rotary_emb is not None:  # 如果有旋转位置编码
            query, key = self.rotary_emb(positions, query, key)  # 应用旋转位置编码

        hidden_states = self.attn(query, key, value, forward_batch=forward_batch)  # 执行注意力计算
        del query, key, value  # 释放QKV内存

        hidden_states = self.o_proj(hidden_states)[0]  # 执行输出投影
        return hidden_states  # 返回结果


class GraniteMoeHybridAttentionDecoderLayer(nn.Module):  # GraniteMoe混合注意力解码器层
    def __init__(  # 初始化注意力解码器层
        self,
        config: GraniteMoeHybridConfig,  # 混合配置
        layer_idx: int,  # 层索引
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.residual_multiplier = config.residual_multiplier  # 残差乘数

        self.self_attn = GraniteMoeHybridAttention(  # 创建自注意力层
            config,  # 配置
            layer_id=layer_idx,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.self_attn",  # 参数前缀
        )

        self.block_sparse_moe = None  # 初始化MoE为None
        if getattr(config, "num_local_experts", 0) > 0:  # 如果配置有本地专家
            self.block_sparse_moe = GraniteMoeMoE(  # 创建稀疏MoE层
                num_experts=config.num_local_experts,  # 专家数量
                top_k=config.num_experts_per_tok,  # 每个token的top-k专家数
                hidden_size=config.hidden_size,  # 隐藏层大小
                intermediate_size=config.intermediate_size,  # 中间层大小
                layer_id=layer_idx,  # 层ID
                quant_config=quant_config,  # 量化配置
                tp_size=get_tensor_model_parallel_world_size(),  # 张量并行大小
                prefix=f"{prefix}.block_sparse_moe",  # 参数前缀
            )

        self.shared_mlp = (  # 创建共享MLP
            None  # 如果没有共享中间层则为None
            if getattr(config, "shared_intermediate_size", 0) == 0  # 检查共享中间层大小
            else GraniteMoeSharedMLP(  # 否则创建共享MLP
                config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 注意力解码器层前向传播
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor | None,  # 残差
        forward_batch: ForwardBatch | None = None,  # 前向批次信息
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化

        hidden_states = self.self_attn(  # 执行自注意力计算
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = residual + hidden_states * self.residual_multiplier  # 残差连接（带乘数）

        residual = hidden_states  # 更新残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化
        if self.shared_mlp is None:  # 如果没有共享MLP
            if self.block_sparse_moe is not None:  # 如果有MoE
                hidden_states = self.block_sparse_moe(hidden_states)  # 执行MoE计算
            # else: skip  # 否则：跳过
        else:  # 如果有共享MLP
            # create a copy since block_sparse_moe modifies in-place  # 创建副本，因为block_sparse_moe会原地修改
            if self.block_sparse_moe is not None:  # 如果同时有MoE
                moe_hidden_states = hidden_states.clone()  # 克隆隐藏状态用于MoE
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)  # 执行MoE计算
                hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)  # MoE结果加上共享MLP结果
                del moe_hidden_states  # 删除临时变量
            else:  # 如果只有共享MLP
                hidden_states = self.shared_mlp(hidden_states)  # 执行共享MLP计算
        hidden_states = residual + hidden_states * self.residual_multiplier  # 残差连接（带乘数）

        return hidden_states, residual  # 返回隐藏状态和残差


ALL_DECODER_LAYER_TYPES = {  # 所有解码器层类型映射
    "attention": GraniteMoeHybridAttentionDecoderLayer,  # 注意力类型
    "mamba": GraniteMoeHybridMambaDecoderLayer,  # Mamba类型
}


class GraniteMoeHybridModel(nn.Module):  # GraniteMoe混合模型主体
    def __init__(  # 初始化混合模型
        self,
        config: GraniteMoeHybridConfig,  # 混合配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.vocab_size = config.vocab_size  # 词表大小

        self.pp_group = get_pp_group()  # 获取流水线并行组

        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个rank
            self.embed_tokens = VocabParallelEmbedding(  # 创建词嵌入层
                self.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                org_num_embeddings=config.vocab_size,  # 原始嵌入数量
            )
        else:  # 否则不是第一个rank
            self.embed_tokens = PPMissingLayer()  # 使用流水线缺失层占位

        self.embedding_multiplier = config.embedding_multiplier  # 嵌入乘数

        def get_layer(idx: int, prefix: str):  # 获取层的工厂函数
            layer_idx = int(prefix.rsplit(".", 1)[1])  # 从前缀提取层索引
            layer_class = ALL_DECODER_LAYER_TYPES[config.layer_types[layer_idx]]  # 根据配置获取层类型
            return layer_class(  # 创建对应类型的层
                config,  # 配置
                layer_idx,  # 层索引
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 参数前缀
            )

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层列表
            config.num_hidden_layers,  # 隐藏层数量
            get_layer,  # 层工厂函数
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行rank
            pp_size=self.pp_group.world_size,  # 流水线并行世界大小
            prefix=f"{prefix}.layers",  # 参数前缀
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个rank
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 创建最终归一化层
        else:  # 否则不是最后一个rank
            self.norm = PPMissingLayer(return_tuple=True)  # 使用流水线缺失层占位
        self.layers_to_capture = []  # 需要捕获的层列表

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入
        """Get input embeddings from the model."""  # 从模型获取输入嵌入
        return self.embed_tokens  # 返回嵌入层

    def forward(  # 混合模型前向传播
        self,
        input_ids: torch.Tensor | None,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch | None = None,  # 前向批次信息
        inputs_embeds: torch.Tensor | None = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个rank
            if inputs_embeds is not None:  # 如果提供了输入嵌入
                hidden_states = inputs_embeds  # 直接使用输入嵌入
            else:  # 否则
                hidden_states = self.embed_tokens(input_ids)  # 从token ID获取嵌入
                hidden_states = hidden_states * self.embedding_multiplier  # 应用嵌入乘数
            residual = None  # 残差初始化为None
        else:  # 否则不是第一个rank
            assert pp_proxy_tensors is not None  # 断言代理张量不为None
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理张量获取残差

        aux_hidden_states = []  # 辅助隐藏状态列表
        for i in range(self.start_layer, self.end_layer):  # 遍历当前rank负责的层
            if i in self.layers_to_capture:  # 如果当前层需要捕获
                aux_hidden_states.append(hidden_states + residual)  # 捕获隐藏状态加残差
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 执行当前层前向传播
                positions,  # 位置编码
                hidden_states,  # 隐藏状态
                residual,  # 残差
                forward_batch,  # 前向批次
            )

        if not self.pp_group.is_last_rank:  # 如果不是最后一个rank
            return PPProxyTensors(  # 返回流水线代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:  # 否则是最后一个rank
            hidden_states, _ = self.norm(hidden_states, residual)  # 执行最终归一化

        if len(aux_hidden_states) == 0:  # 如果没有辅助隐藏状态
            return hidden_states  # 直接返回隐藏状态

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态


class GraniteMoeHybridForCausalLM(  # GraniteMoe混合因果语言模型
    nn.Module,
):
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [  # QKV投影打包
            "q_proj",  # Q投影
            "k_proj",  # K投影
            "v_proj",  # V投影
        ],
        "conv1d": ["conv1d"],  # 1D卷积
        "in_proj": ["in_proj"],  # 输入投影
        "input_linear": ["input_linear"],  # 输入线性层
    }
    embedding_modules = {  # 嵌入模块映射
        "embed_tokens": "input_embeddings",  # 输入嵌入
        "lm_head": "output_embeddings",  # 输出嵌入
    }

    def __init__(  # 初始化因果语言模型
        self,
        config: GraniteMoeHybridConfig,  # 混合配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化

        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态
        self.pp_group = get_pp_group()  # 获取流水线并行组

        self.quant_config = quant_config  # 保存量化配置
        self.config = config  # 保存配置
        self.model = GraniteMoeHybridModel(  # 创建模型主体
            config=config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=maybe_prefix(prefix, "model"),  # 参数前缀
        )

        self.lm_head = ParallelLMHead(  # 创建语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=self.quant_config,  # 量化配置
            prefix=maybe_prefix(prefix, "lm_head"),  # 参数前缀
        )

        if config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head.weight = self.model.embed_tokens.weight  # 共享权重

        self.logits_processor = LogitsProcessor(  # 创建logits处理器
            config,  # 配置
            logit_scale=1 / self.config.logits_scaling,  # logits缩放因子
        )

        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 创建池化层

    @property  # 属性装饰器
    def start_layer(self):  # 获取起始层
        return self.model.start_layer  # 返回模型的起始层

    @property  # 属性装饰器
    def end_layer(self):  # 获取结束层
        return self.model.end_layer  # 返回模型的结束层

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入
        return self.model.embed_tokens  # 返回模型的嵌入层

    def forward(  # 因果语言模型前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        get_embedding: bool = False,  # 是否获取嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ):
        hidden_states = self.model(  # 模型前向传播
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        aux_hidden_states = None  # 辅助隐藏状态初始化
        if self.capture_aux_hidden_states:  # 如果需要捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 拆分隐藏状态

        if self.pp_group.is_last_rank:  # 如果是最后一个rank
            if not get_embedding:  # 如果不获取嵌入
                return self.logits_processor(  # 返回logits处理结果
                    input_ids,  # 输入ID
                    hidden_states,  # 隐藏状态
                    self.lm_head,  # 语言模型头
                    forward_batch,  # 前向批次
                    aux_hidden_states,  # 辅助隐藏状态
                )
            else:  # 否则获取嵌入
                return self.pooler(hidden_states, forward_batch)  # 返回池化结果
        else:  # 否则不是最后一个rank
            return hidden_states  # 返回隐藏状态

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:  # 获取专家映射
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # layers.0.block_sparse_moe.expert_0.input_linear.input_scale  # 权重、fp8权重缩放和fp8激活缩放的参数映射
        ckpt_gate_proj_name = "gate_proj"  # 检查点gate投影名称
        ckpt_down_proj_name = "down_proj"  # 检查点down投影名称
        ckpt_up_proj_name = "up_proj"  # 检查点up投影名称
        num_experts = self.config.num_local_experts  # 专家数量

        return [  # 返回映射列表
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "block_sparse_moe.experts.w13_"  # w13合并参数名
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]  # gate或up投影
                    else "block_sparse_moe.experts.w2_"  # w2参数名
                ),
                f"block_sparse_moe.experts.{expert_id}.{weight_name}.",  # 权重名称
                expert_id,  # 专家ID
                shard_id,  # 分片ID
            )
            for expert_id in range(num_experts)  # 遍历每个专家
            for shard_id, weight_name in [  # 遍历每个分片
                ("w1", ckpt_gate_proj_name),  # w1对应gate投影
                ("w2", ckpt_down_proj_name),  # w2对应down投影
                ("w3", ckpt_up_proj_name),  # w3对应up投影
            ]
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:  # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典
        loaded_params: set[str] = set()  # 已加载参数集合
        expert_params_mapping = self.get_expert_mapping()  # 获取专家参数映射

        def _load(n, p):  # 加载权重内部函数
            param = params_dict[n]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, p)  # 加载权重
            loaded_params.add(n)  # 添加到已加载集合

        def _load_shard(n, p, shard_id):  # 加载分片权重内部函数
            # Skip layers on other devices.  # 跳过其他设备上的层
            param = params_dict[n]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, p, shard_id)  # 加载权重（带分片ID）
            loaded_params.add(n)  # 添加到已加载集合

        def _load_expert(n, p, name, shard_id, expert_id):  # 加载专家权重内部函数
            param = params_dict[n]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, p, name, shard_id=shard_id, expert_id=expert_id)  # 加载权重（带分片和专家ID）
            loaded_params.add(n)  # 添加到已加载集合

        def _load_quant_expert(name, loaded_weight):  # 加载量化专家权重内部函数
            for mapping in expert_params_mapping:  # 遍历专家参数映射
                param_name, weight_name, expert_id, shard_id = mapping  # 解包映射

                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 跳过

                name_mapped = name.replace(weight_name, param_name)  # 替换权重名为参数名

                # Skip layers on other devices.  # 跳过其他设备上的层
                # if is_pp_missing_parameter(name_mapped, self):
                #     continue

                param = params_dict[name_mapped]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                success = False  # 加载成功标志

                if weight_loader is not None:  # 如果有权重加载器
                    success = weight_loader(  # 执行权重加载
                        param,  # 参数
                        loaded_weight,  # 加载的权重
                        name_mapped,  # 映射后的名称
                        shard_id=shard_id,  # 分片ID
                        expert_id=expert_id,  # 专家ID
                        return_success=True,  # 返回成功标志
                    )

                if success:  # 如果加载成功
                    return name_mapped  # 返回映射后的名称
            return None  # 返回None

        for n, p in weights:  # 遍历所有权重
            if "A_log" in n:  # 如果是A_log参数
                n = n.replace("A_log", "A")  # 替换为A

            if self.quant_config is not None and (  # 如果有量化配置
                scale_name := self.quant_config.get_cache_scale(n)  # 获取缓存缩放名称
            ):
                # Loading kv cache quantization scales  # 加载KV缓存量化缩放因子
                loaded_weight = p  # 获取权重
                loaded_weight = (  # 处理维度
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                _load(scale_name, loaded_weight)  # 加载缩放权重
                loaded_params.add(scale_name)  # 添加到已加载集合
                continue  # 跳到下一个权重

            if _load_quant_expert(n, p):  # 如果成功加载量化专家权重
                continue  # 跳到下一个权重

            # Logic analogous to: https://github.com/vllm-project/vllm/blob/f49e5aff11c986ed4d45202b1716c5d74786efa9/vllm/model_executor/models/granitemoeshared.py#L215
            # Mapping different experts' layout:
            #  from HF (input_linear, output_linear, router)
            #  to vLLM (experts_w13({e}.w1, {e}.w2), experts_w3({e}.w3), gate)
            # The renaming and parameter loading logic is the same for weight
            # and weight_scale tensors so we can reuse them without issues.
            # 类似于vLLM的逻辑：将HF格式的专家布局(input_linear, output_linear, router)映射为vLLM格式(experts_w13, experts_w3, gate)。
            # 重命名和参数加载逻辑对权重和权重缩放张量相同，可以复用。
            if n.endswith(".block_sparse_moe.input_linear.weight") or n.endswith(  # 如果是MoE输入线性层权重
                ".block_sparse_moe.input_linear.weight_scale"
            ):
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
                    _load_expert(  # 加载w1专家权重
                        n.replace(".input_linear.", ".experts.w13_"),
                        w1_param,
                        w1_name,
                        shard_id="w1",
                        expert_id=e,
                    )
                    _load_expert(  # 加载w3专家权重
                        n.replace(".input_linear.", ".experts.w13_"),
                        w3_param,
                        w3_name,
                        shard_id="w3",
                        expert_id=e,
                    )
            elif n.endswith(".block_sparse_moe.output_linear.weight") or n.endswith(  # 如果是MoE输出线性层权重
                ".block_sparse_moe.output_linear.weight_scale"
            ):
                for e in range(p.size(0)):  # 遍历每个专家
                    w2_name = n.replace(  # 构造w2权重名
                        ".block_sparse_moe.output_linear.weight",
                        f".block_sparse_moe.experts.{e}.w2.weight",
                    )
                    w2_param = p[e]  # 获取当前专家的w2权重
                    _load_expert(  # 加载w2专家权重
                        n.replace(".output_linear.", ".experts.w2_"),
                        w2_param,
                        w2_name,
                        shard_id="w2",
                        expert_id=e,
                    )
            elif n.endswith(".block_sparse_moe.router.layer.weight"):  # 如果是路由器权重
                gate_name = n.replace(  # 构造门控权重名
                    ".block_sparse_moe.router.layer.weight",
                    ".block_sparse_moe.gate.weight",
                )
                _load(gate_name, p)  # 加载门控权重
            else:  # 其他权重
                loaded = False  # 加载标志
                for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                    if weight_name in n:  # 如果权重名在参数名中
                        _load_shard(  # 加载分片权重
                            n.replace(weight_name, param_name), p, shard_id=shard_id
                        )
                        loaded = True  # 设置加载标志
                if not loaded:  # 如果没有通过堆叠映射加载
                    _load(n, p)  # 直接加载权重

        return loaded_params  # 返回已加载参数集合


EntryClass = [GraniteMoeHybridForCausalLM]  # 入口类列表，用于模型注册
