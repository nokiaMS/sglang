# Phi-3 Small模型推理实现文件
# 本文件实现了Phi-3 Small大语言模型的推理架构
# 包含GEGELU激活函数、MLP、自注意力、解码器层、模型主体及因果语言模型等组件

import math  # 导入数学模块
from typing import Iterable, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import Phi3Config  # 导入Phi3配置
from transformers.configuration_utils import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.layers.linear import (  # 导入并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    DEFAULT_VOCAB_PADDING_SIZE,  # 默认词表填充大小
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加和层创建工具


@torch.jit.script
def quick_gelu(x):  # 快速GELU激活函数，使用JIT编译优化
    return x * torch.sigmoid(1.702 * x)  # x乘以sigmoid(1.702*x)


@torch.jit.script
def gegelu(input, limit: Optional[float] = None):  # 门控扩展GELU激活函数，使用JIT编译优化
    a_gelu, a_linear = input[..., ::2], input[..., 1::2]  # 分割为GELU部分和线性部分
    if limit is not None:  # 如果有限制值
        a_gelu = torch.where(  # 限制GELU部分
            torch.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)  # 限制上界
        )
        a_linear = torch.where(  # 限制线性部分
            torch.isinf(a_linear),  # 如果是无穷
            a_linear,  # 保持不变
            a_linear.clamp(min=-limit, max=limit),  # 限制上下界
        )
    out_gelu = quick_gelu(a_gelu)  # 对GELU部分应用快速GELU
    return out_gelu * (a_linear + 1)  # 门控输出


class Phi3SmallMLP(nn.Module):  # Phi-3 Small模型的MLP模块

    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        assert (  # 断言
            self.config.hidden_act == "gegelu"  # 激活函数必须是gegelu
        ), "Only `gegelu` is supported for the 4.7 series of models .."  # 仅4.7系列支持gegelu
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.gegelu_limit = config.gegelu_limit  # GEGELU限制值
        self.intermediate_size = config.intermediate_size  # 中间层大小

        self.up_proj = MergedColumnParallelLinear(  # 上投影合并列并行线性层
            self.hidden_size,  # 输入大小
            2 * [self.intermediate_size],  # 输出大小列表（门控和上投影）
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("up_proj", prefix),  # 添加前缀
        )
        self.down_proj = RowParallelLinear(  # 下投影行并行线性层
            self.intermediate_size,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 添加前缀
        )

    def forward(self, x):  # 前向传播函数，执行MLP计算
        gate_up, _ = self.up_proj(x)  # 通过上投影层获取门控和上投影结果
        x = gegelu(gate_up)  # 应用GEGELU激活
        x, _ = self.down_proj(x)  # 通过下投影层
        return x  # 返回输出


class Phi3SmallSelfAttention(nn.Module):  # Phi-3 Small模型的自注意力模块

    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层ID
        self.config = config  # 保存配置
        self.sparse_block_size = config.blocksparse_block_size  # 稀疏块大小
        self.homo_heads = config.blocksparse_homo_head_pattern  # 同质头模式
        self.local_blocks = config.blocksparse_num_local_blocks  # 局部块数
        self.vert_stride = config.blocksparse_vert_stride  # 垂直步幅

        assert (  # 断言稀疏块大小与Triton核块大小一致
            config.blocksparse_block_size == config.blocksparse_triton_kernel_block_size
        )

        self.hidden_size = config.hidden_size  # 隐藏层大小
        # Number of Query Heads  # 查询头数
        self.num_heads = config.num_attention_heads  # 注意力头数

        self.head_dim = self.hidden_size // self.num_heads  # 每个头的维度
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        # Number of total Key Value Heads before tensor parallel  # 张量并行前的总KV头数
        self.num_key_value_heads = config.num_key_value_heads  # KV头数
        self.num_q_per_kv = self.num_heads // self.num_key_value_heads  # 每个KV对应的Q头数
        if self.tp_size > 1:  # 如果TP大小大于1
            assert self.num_key_value_heads % self.tp_size == 0  # 断言KV头数可被TP大小整除
        self.num_kv_heads_per_partion = max(1, self.num_key_value_heads // self.tp_size)  # 每个分区的KV头数
        self.num_heads_per_partition = self.num_heads // self.tp_size  # 每个分区的Q头数

        self.max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入数
        self.rope_embedding_base = config.rope_embedding_base  # RoPE基础值
        self.rope_position_scale = config.rope_position_scale  # RoPE位置缩放
        self.is_causal = True  # 是否因果注意力

        norm_factor = None  # 归一化因子
        if config.mup_use_scaling:  # 如果使用muP缩放
            norm_factor = self.head_dim / config.mup_attn_multiplier  # muP归一化因子
        else:  # 否则
            norm_factor = math.sqrt(self.head_dim)  # 标准归一化因子
        self.scale = 1 / norm_factor  # 缩放因子

        self.query_key_value = QKVParallelLinear(  # QKV并行线性层
            self.hidden_size,  # 输入大小
            self.head_dim,  # 每个头的维度
            self.num_heads,  # Q头数
            self.num_key_value_heads,  # KV头数
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 添加前缀
        )

        self.dense = RowParallelLinear(  # 输出投影行并行线性层
            self.hidden_size,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 添加前缀
        )

        rope_scaling = self.config.rope_parameters  # RoPE参数
        if rope_scaling is not None:  # 如果RoPE参数存在
            for key in rope_scaling:  # 遍历RoPE参数
                if isinstance(rope_scaling[key], list):  # 如果是列表
                    rope_scaling[key] = tuple(rope_scaling[key])  # 转换为元组

            if "factor" not in rope_scaling:  # 如果没有factor
                rope_scaling["factor"] = self.rope_position_scale  # 设置factor为位置缩放
        else:  # 否则
            rope_scaling = {  # 创建默认RoPE配置
                "rope_type": "linear",  # 线性RoPE
                "factor": self.rope_position_scale,  # 位置缩放因子
            }

        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=self.max_position_embeddings,  # 最大位置
            base=self.rope_embedding_base,  # 基础值
            rope_scaling=rope_scaling,  # RoPE缩放
        )

        # blocksparse params  # 稀疏块参数
        self.blocksparse_block_size = config.blocksparse_block_size  # 稀疏块大小
        self.blocksparse_num_local_blocks = config.blocksparse_num_local_blocks  # 稀疏局部块数
        self.blocksparse_vert_stride = config.blocksparse_vert_stride  # 稀疏垂直步幅

        use_dense_attn = (  # 是否使用稠密注意力
            getattr(self.config, "dense_attention_every_n_layers", None)  # 获取每隔N层使用稠密注意力
            and (self.layer_id + 1) % self.config.dense_attention_every_n_layers == 0  # 当前层是否使用稠密注意力
        )

        bs_params = None  # 稀疏块参数
        if not use_dense_attn:  # 如果不使用稠密注意力
            bs_params = {  # 稀疏块参数
                "max_seqlen": self.max_position_embeddings,  # 最大序列长度
                "num_heads": self.num_heads_per_partition,  # Q头数
                "num_kv_heads": self.num_kv_heads_per_partion,  # KV头数
                "block_size": self.sparse_block_size,  # 块大小
                "local_blocks": self.local_blocks,  # 局部块数
                "vert_stride": self.vert_stride,  # 垂直步幅
                "homo_head": self.homo_heads,  # 同质头标志
            }

        self.attn = RadixAttention(  # 基数注意力模块
            self.num_heads_per_partition,  # Q头数
            self.head_dim,  # 头维度
            self.scale,  # 缩放因子
            num_kv_heads=self.num_kv_heads_per_partion,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
        )

    def forward(  # 前向传播函数，执行自注意力计算
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        qkv, _ = self.query_key_value(hidden_states)  # 通过QKV投影

        qkv = qkv.view(qkv.shape[:-1] + (-1, (self.num_q_per_kv + 2), self.head_dim))  # 重塑QKV
        q, k, v = qkv.split([self.num_q_per_kv, 1, 1], dim=-2)  # 分割Q、K、V

        # NOTE: this is required by RotaryEmbed, which indeed does not have to  # 注意：这是RotaryEmbed所需的，但实际上不必
        # TODO: allow 3D QK for rotary forward  # TODO：允许3D QK进行旋转前向
        q = q.reshape(-1, self.head_dim * self.num_heads_per_partition)  # 重塑Q
        k = k.reshape(-1, self.head_dim * self.num_kv_heads_per_partion)  # 重塑K
        v = v.reshape(-1, self.head_dim * self.num_kv_heads_per_partion)  # 重塑V

        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)  # 执行注意力计算
        output, _ = self.dense(attn_output)  # 通过输出投影

        return output  # 返回输出


class Phi3SmallDecoderLayer(nn.Module):  # Phi-3 Small模型的解码器层

    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.self_attn = Phi3SmallSelfAttention(  # 自注意力模块
            config,  # 配置
            layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
        )
        self.mlp = Phi3SmallMLP(  # MLP模块
            config,  # 配置
            quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 添加前缀
        )

        self.input_layernorm = nn.LayerNorm(  # 输入层归一化
            config.hidden_size, eps=config.layer_norm_epsilon  # 隐藏层大小和eps
        )
        self.post_attention_layernorm = nn.LayerNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.layer_norm_epsilon  # 隐藏层大小和eps
        )

    def forward(  # 前向传播函数，执行解码器层计算
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化

        hidden_states = self.self_attn(  # 通过自注意力层
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP
        hidden_states = residual + hidden_states  # 残差连接
        return hidden_states  # 返回隐藏状态


class Phi3SmallModel(nn.Module):  # Phi-3 Small模型主体

    def __init__(  # 初始化函数
        self,
        config: Phi3Config,  # Phi3配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        self.pp_group = get_pp_group()  # 获取流水线并行组
        if self.pp_group.is_first_rank:  # 如果是第一个秩
            self.embed_tokens = VocabParallelEmbedding(  # 词表嵌入层
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                prefix=add_prefix("embed_tokens", prefix),  # 添加前缀
            )
        else:  # 否则
            self.embed_tokens = PPMissingLayer()  # 使用缺失层占位

        self.embed_tokens = VocabParallelEmbedding(  # 词表嵌入层（覆盖上面的赋值）
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            prefix=add_prefix("embed_tokens", prefix),  # 添加前缀
        )
        self.mup_embedding_multiplier = config.mup_embedding_multiplier  # muP嵌入乘数
        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 隐藏层数量
            lambda idx, prefix: Phi3SmallDecoderLayer(  # 解码器层构造函数
                config,  # 配置
                int(prefix.split(".")[-1]),  # 从前缀提取层ID
                quant_config,  # 量化配置
                prefix=prefix,  # 前缀
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行大小
            prefix=add_prefix("layers", prefix),  # 添加前缀
        )

        self.final_layernorm = nn.LayerNorm(  # 最终层归一化
            config.hidden_size, eps=config.layer_norm_epsilon  # 隐藏层大小和eps
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.embed_tokens(input_ids)  # 通过词表嵌入层

    def forward(  # 前向传播函数，执行模型主体计算
        self,
        input_ids: torch.LongTensor,  # 输入ID
        positions: Optional[torch.LongTensor],  # 位置
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor],  # 输入嵌入
    ) -> Union[torch.Tensor]:

        if inputs_embeds is not None:  # 如果提供了输入嵌入
            hidden_states = inputs_embeds  # 使用输入嵌入
        else:  # 否则
            hidden_states = self.get_input_embeddings(input_ids)  # 通过词表嵌入层
        if (  # 如果
            self.mup_embedding_multiplier is not None  # muP嵌入乘数不为空
            and self.mup_embedding_multiplier > 0.0  # 且大于0
        ):
            hidden_states = hidden_states * self.mup_embedding_multiplier  # 应用muP缩放

        for i in range(self.start_layer, self.end_layer):  # 遍历解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states = layer(positions, hidden_states, forward_batch=forward_batch)  # 通过当前层

        hidden_states = self.final_layernorm(hidden_states)  # 应用最终层归一化
        return hidden_states  # 返回隐藏状态


class Phi3SmallForCausalLM(nn.Module):  # Phi-3 Small因果语言模型
    _tied_weights_keys = ["lm_head.weight"]  # 绑定权重的键

    def __init__(  # 初始化函数
        self,
        config: Phi3Config,  # Phi3配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):

        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = Phi3SmallModel(  # 模型主体
            config=config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 添加前缀
        )
        self.vocab_size = config.vocab_size  # 词表大小
        self.mup_width_multiplier = config.mup_width_multiplier  # muP宽度乘数
        self.lm_head = ParallelLMHead(  # 语言模型头
            self.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            org_num_embeddings=config.vocab_size,  # 原始嵌入数
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,  # 填充大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 添加前缀
        )
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head.weight = self.model.embed_tokens.weight  # 绑定权重
        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 池化器

        # tokens in tiktoken but not used  # tiktoken中未使用的token
        if hasattr(config, "dummy_token_indices"):  # 如果有虚拟token索引配置
            device = self.lm_head.weight.device  # 获取设备
            self.register_buffer(  # 注册缓冲区
                "dummy_token_indices",  # 虚拟token索引
                torch.LongTensor(config.dummy_token_indices).to(device),  # 创建张量
                persistent=False,  # 非持久化
            )
        else:  # 否则
            self.dummy_token_indices = None  # 虚拟token索引为空

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.model.get_input_embeddings(input_ids)  # 通过模型获取嵌入

    def set_input_embeddings(self, value):  # 设置输入嵌入
        self.model.embed_tokens = value  # 设置词表嵌入层

    def get_output_embeddings(self):  # 获取输出嵌入
        return self.lm_head  # 返回语言模型头

    def set_output_embeddings(self, value):  # 设置输出嵌入
        self.lm_head = value  # 设置语言模型头

    def set_decoder(self, decoder):  # 设置解码器
        self.model = decoder  # 设置模型主体

    def get_decoder(self):  # 获取解码器
        return self.model  # 返回模型主体

    def compute_logits(  # 计算logits函数
        self,
        input_ids: torch.LongTensor,  # 输入ID
        hidden_states: torch.Tensor,  # 隐藏状态
        sampling_metadata,  # 采样元数据
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(  # 通过logits处理器
            input_ids, self.lm_head, hidden_states, sampling_metadata  # 传入参数
        )
        if self.dummy_token_indices is not None and logits is not None:  # 如果有虚拟token索引
            logits.index_fill_(-1, self.dummy_token_indices, -torch.inf)  # 将虚拟token的logits设为负无穷
        return logits  # 返回logits

    def forward(  # 前向传播函数，执行因果语言模型计算
        self,
        input_ids: torch.LongTensor,  # 输入ID
        positions: Optional[torch.LongTensor],  # 位置
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(  # 通过模型主体
            input_ids=input_ids,  # 输入ID
            positions=positions,  # 位置
            forward_batch=forward_batch,  # 前向批次
            inputs_embeds=inputs_embeds,  # 输入嵌入
        )

        if not get_embedding:  # 如果不获取嵌入
            return self.logits_processor(  # 返回logits处理器结果
                input_ids, hidden_states, self.lm_head, forward_batch  # 传入参数
            )

        else:  # 否则
            return self.pooler(hidden_states, forward_batch)  # 返回池化结果

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数

        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码逆频率
                continue  # 跳过
            if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                continue  # 跳过
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 如果绑定词嵌入
                continue  # 跳过

            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重


EntryClass = Phi3SmallForCausalLM  # 入口类为Phi3SmallForCausalLM
