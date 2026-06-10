# EXAONE-4模型推理实现文件
# 本文件实现了EXAONE-4（LG AI Research第四代模型）的推理逻辑
# 主要包含：门控MLP、注意力层、解码器层、模型主体和因果语言模型
# 支持滑动窗口注意力、QK归一化、post-LN架构等特性

from collections.abc import Iterable  # 从collections.abc导入Iterable类型
from typing import Any, List, Optional, Tuple, Union  # 导入类型注解

import torch  # 导入PyTorch
from torch import nn  # 从PyTorch导入神经网络模块
from transformers import Exaone4Config  # 从transformers导入Exaone4配置类

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式工具
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.dp_attention import (  # 从数据并行注意力模块导入
    get_attention_tp_rank,  # 获取注意力张量并行秩
    get_attention_tp_size,  # 获取注意力张量并行大小
    get_local_attention_dp_size,  # 获取本地注意力数据并行大小
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 从线性层模块导入
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器及其输出
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层和池化类型
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入流水线缺失层和层ID获取
from sglang.srt.layers.vocab_parallel_embedding import (  # 从词表并行嵌入模块导入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和代理张量
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # KV缩放名称重映射
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加和层创建工具
from sglang.utils import get_exception_traceback, logger  # 导入异常回溯和日志记录器


# Aligned with HF's implementation, using sliding window inclusive with the last token  # 与HuggingFace实现对齐，滑动窗口包含最后一个token
# SGLang assumes exclusive  # SGLang假设滑动窗口不包含最后一个token
def get_attention_sliding_window_size(config):  # 获取注意力滑动窗口大小
    if getattr(config, "sliding_window", None) is not None:  # 如果配置中有滑动窗口参数
        return config.sliding_window - 1  # 减1以从包含式转为排除式
    else:  # 否则没有滑动窗口
        return None  # 返回None


class Exaone4GatedMLP(nn.Module):  # Exaone4门控MLP模块
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层维度大小
        intermediate_size: int,  # 中间层维度大小
        hidden_act: str,  # 隐藏层激活函数名称
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        bias: bool = False,  # 是否使用偏置，默认为False
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # gate和up的合并列并行线性层
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度（gate和up各一份）
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # down行并行线性层
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
        )
        if hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. "  # 不支持的激活函数
                "Only silu is supported for now."  # 目前仅支持silu
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数

    def forward(self, x):  # 前向传播函数
        gate_up, _ = self.gate_up_proj(x)  # 通过gate_up投影
        x = self.act_fn(gate_up)  # 应用SiLU激活函数和门控
        x, _ = self.down_proj(x)  # 通过down投影
        return x  # 返回输出


class Exaone4Attention(nn.Module):  # Exaone4注意力模块
    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        hidden_size: int,  # 隐藏层维度
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID，默认为0
        head_dim: Optional[int] = None,  # 头维度，默认为None
        rms_norm_eps: float = 1e-06,  # RMS归一化epsilon，默认1e-06
        rope_theta: float = 10000,  # 旋转位置编码基数，默认10000
        rope_scaling: Optional[dict[str, Any]] = None,  # 旋转位置编码缩放配置
        max_position_embeddings: int = 8192,  # 最大位置编码数，默认8192
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        bias: bool = False,  # 是否使用偏置，默认为False
        bias_o_proj: bool = False,  # 输出投影是否使用偏置，默认为False
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层维度
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小

        attn_tp_rank = get_attention_tp_rank()  # 获取注意力张量并行秩
        attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小

        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 确保头数能被并行大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个并行秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于并行大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此进行分区
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头分配到多个张量并行GPU上
            assert self.total_num_kv_heads % tp_size == 0  # 确保KV头数能被并行大小整除
        else:  # 否则KV头数小于并行大小
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此进行复制
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头复制到多个张量并行GPU上
            assert tp_size % self.total_num_kv_heads == 0  # 确保并行大小能被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个并行秩的KV头数

        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q维度大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV维度大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 旋转位置编码基数
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码数

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            hidden_size=hidden_size,  # 输入维度
            head_size=self.head_dim,  # 头维度
            total_num_heads=self.total_num_heads,  # 总Q头数
            total_num_kv_heads=self.total_num_kv_heads,  # 总KV头数
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
            tp_rank=attn_tp_rank,  # 注意力张量并行秩
            tp_size=attn_tp_size,  # 注意力张量并行大小
        )

        self.o_proj = RowParallelLinear(  # 输出行并行线性投影
            input_size=self.total_num_heads * self.head_dim,  # 输入维度
            output_size=hidden_size,  # 输出维度
            bias=bias_o_proj,  # 输出投影是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
            tp_rank=attn_tp_rank,  # 注意力张量并行秩
            tp_size=attn_tp_size,  # 注意力张量并行大小
        )

        is_neox_style = True  # 默认使用Neox风格RoPE
        if quant_config is not None and quant_config.get_name() == "gguf":  # 如果使用GGUF量化
            is_neox_style = False  # 不使用Neox风格

        interleaved_sliding_window = get_attention_sliding_window_size(config)  # 获取交错滑动窗口大小
        self.sliding_window_pattern = getattr(config, "sliding_window_pattern", None)  # 获取滑动窗口模式

        self.is_sliding = False  # 默认不使用滑动窗口
        if self.sliding_window_pattern:  # 如果有滑动窗口模式
            if (layer_id + 1) % len(self.sliding_window_pattern) != 0:  # 如果不是最后一个模式组
                self.is_sliding = True  # 使用滑动窗口

        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基数
            rope_scaling=rope_scaling,  # 缩放配置
            is_neox_style=is_neox_style,  # 是否Neox风格
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            sliding_window_size=(  # 滑动窗口大小
                interleaved_sliding_window if self.is_sliding else None  # 如果使用滑动窗口则设置大小
            ),
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)  # Q的RMS归一化
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)  # K的RMS归一化

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割为Q、K、V

        # Add qk-norm  # 添加QK归一化
        q_shape = q.shape  # 保存Q的原始形状
        q = q.reshape(-1, self.head_dim)  # 重塑Q为二维
        q = self.q_norm(q)  # 对Q进行RMS归一化
        q = q.reshape(q_shape)  # 恢复Q的原始形状

        k_shape = k.shape  # 保存K的原始形状
        k = k.reshape(-1, self.head_dim)  # 重塑K为二维
        k = self.k_norm(k)  # 对K进行RMS归一化
        k = k.reshape(k_shape)  # 恢复K的原始形状

        if not self.sliding_window_pattern or self.is_sliding:  # 如果没有滑动窗口模式或当前层使用滑动窗口
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力
        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class Exaone4DecoderLayer(nn.Module):  # Exaone4解码器层
    def __init__(  # 初始化函数
        self,
        config: Exaone4Config,  # Exaone4配置
        layer_id: int = 0,  # 层ID，默认为0
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层ID
        self.hidden_size = config.hidden_size  # 隐藏层维度

        rope_theta = getattr(config, "rope_theta", 1000000)  # 获取RoPE基数，默认1000000
        rope_scaling = getattr(config, "rope_scaling", None)  # 获取RoPE缩放配置
        if rope_scaling is not None and getattr(  # 如果RoPE缩放配置存在且配置中有原始最大位置编码
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 设置原始最大位置编码
                config.original_max_position_embeddings  # 从配置获取
            )

        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 获取最大位置编码数

        self.local_dp_size = get_local_attention_dp_size()  # 获取本地注意力数据并行大小
        self.attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小
        self.attn_tp_rank = get_attention_tp_rank()  # 获取注意力张量并行秩

        self.self_attn = Exaone4Attention(  # 自注意力模块
            config=config,  # 模型配置
            hidden_size=self.hidden_size,  # 隐藏层维度
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=getattr(  # KV头数
                config, "num_key_value_heads", config.num_key_value_heads  # 从配置获取或使用默认值
            ),
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE基数
            rope_scaling=rope_scaling,  # RoPE缩放配置
            max_position_embeddings=max_position_embeddings,  # 最大位置编码数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )
        self.mlp = Exaone4GatedMLP(  # 门控MLP模块
            hidden_size=self.hidden_size,  # 隐藏层维度
            intermediate_size=config.intermediate_size,  # 中间层维度
            hidden_act=config.hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )
        self.post_attention_layernorm = RMSNorm(  # 注意力后的层归一化
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(  # 前馈网络后的层归一化
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差张量
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if residual is None:  # 如果没有残差（第一层）
            residual = hidden_states  # 初始化残差为隐藏状态

        # Self Attention  # 自注意力计算
        hidden_states = self.self_attn(  # 通过自注意力模块
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次信息
        )

        # Use post-LN  # 使用后归一化（post-LN）
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化
        hidden_states = hidden_states + residual  # 残差连接
        residual = hidden_states  # 更新残差

        # Fully Connected  # 全连接层
        hidden_states = self.mlp(hidden_states)  # 通过MLP

        # Use post-LN  # 使用后归一化（post-LN）
        hidden_states = self.post_feedforward_layernorm(hidden_states)  # 前馈后层归一化
        hidden_states = hidden_states + residual  # 残差连接
        residual = hidden_states  # 更新残差

        return hidden_states, residual  # 返回隐藏状态和残差


class Exaone4Model(nn.Module):  # Exaone4模型主体
    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.vocab_size = config.vocab_size  # 词表大小
        self.pp_group = get_pp_group()  # 获取流水线并行组
        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层维度
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("embed_tokens", prefix),  # 参数前缀
            )
        else:  # 否则不是第一个秩
            self.embed_tokens = PPMissingLayer()  # 使用缺失层占位

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 层数
            lambda idx, prefix: Exaone4DecoderLayer(  # 层创建函数
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                layer_id=idx,  # 层ID
                prefix=prefix,  # 参数前缀
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行大小
            prefix=add_prefix("layers", prefix),  # 参数前缀
        )
        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化
        else:  # 否则不是最后一个秩
            self.norm = PPMissingLayer(return_tuple=True)  # 使用缺失层占位

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            if input_embeds is None:  # 如果没有提供输入嵌入
                hidden_states = self.get_input_embeddings(input_ids)  # 通过词嵌入层获取隐藏状态
            else:  # 否则使用提供的嵌入
                hidden_states = input_embeds  # 使用输入嵌入
            residual = None  # 初始化残差为None
        else:  # 否则不是第一个秩
            assert pp_proxy_tensors is not None  # 确保代理张量不为空
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理张量获取残差

        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                positions,  # 位置编码
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次信息
                residual,  # 残差
            )
        if not self.pp_group.is_last_rank:  # 如果不是最后一个秩
            return PPProxyTensors(  # 返回代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:  # 否则是最后一个秩
            hidden_states = self.norm(hidden_states)  # 通过最终层归一化
        return hidden_states  # 返回隐藏状态


class Exaone4ForCausalLM(nn.Module):  # Exaone4因果语言模型
    _tied_weights_keys = ["lm_head.weight"]  # 绑定权重的键名
    _tp_plan = {"lm_head": "colwise_rep"}  # 张量并行计划
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}  # 流水线并行计划
    base_model_prefix = "language_model"  # 基础模型前缀

    # BitandBytes specific attributes  # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认的BitandBytes量化目标模块
        ".gate_proj.",  # gate投影
        ".down_proj.",  # down投影
        ".up_proj.",  # up投影
        ".q_proj.",  # Q投影
        ".k_proj.",  # K投影
        ".v_proj.",  # V投影
        ".o_proj.",  # 输出投影
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射
        ".q_proj": (".qkv_proj", 0),  # Q到QKV的映射，分片ID为0
        ".k_proj": (".qkv_proj", 1),  # K到QKV的映射，分片ID为1
        ".v_proj": (".qkv_proj", 2),  # V到QKV的映射，分片ID为2
        ".gate_proj": (".gate_up_proj", 0),  # gate到gate_up的映射，分片ID为0
        ".up_proj": (".gate_up_proj", 1),  # up到gate_up的映射，分片ID为1
    }

    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [  # QKV投影
            "q_proj",  # Q投影
            "k_proj",  # K投影
            "v_proj",  # V投影
        ],
        "gate_up_proj": [  # gate_up投影
            "gate_proj",  # gate投影
            "up_proj",  # up投影
        ],
    }

    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))  # 初始化模型主体
        # Exaone-4.0 32B set tie_word_embeddins to False  # Exaone-4.0 32B设置tie_word_embeddings为False
        # Exaone-4.0 1.2B set tie_word_embeddins to True  # Exaone-4.0 1.2B设置tie_word_embeddings为True
        if config.tie_word_embeddings:  # 如果绑定词嵌入权重
            self.lm_head = self.model.embed_tokens  # 语言模型头共享词嵌入
        else:  # 否则不绑定
            self.lm_head = ParallelLMHead(  # 并行语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层维度
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力张量并行组
            )

        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 池化层（取最后一个token并归一化）

    def _init_model(  # 初始化模型主体的工厂方法
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        return Exaone4Model(config, quant_config=quant_config, prefix=prefix)  # 返回Exaone4Model实例

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.model.embed_tokens  # 返回词嵌入层

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        get_embedding: bool = False,  # 是否获取嵌入（用于嵌入任务），默认为False
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(  # 通过模型主体获取隐藏状态
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,  # 代理张量
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            if not get_embedding:  # 如果不获取嵌入
                return self.logits_processor(  # 通过logits处理器
                    input_ids,  # 输入ID
                    hidden_states,  # 隐藏状态
                    self.lm_head,  # 语言模型头
                    forward_batch,  # 前向批次信息
                )
            else:  # 否则获取嵌入
                return self.pooler(hidden_states, forward_batch)  # 通过池化层
        else:  # 否则不是最后一个秩
            return hidden_states  # 直接返回隐藏状态

    @torch.no_grad()  # 禁用梯度计算
    def forward_split_prefill(  # 分割预填充前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        split_interval: Tuple[int, int],  # [start, end) 0-based  # 分割区间，左闭右开，从0开始
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ):
        start, end = split_interval  # 获取分割区间的起始和结束
        # embed  # 嵌入
        if start == 0:  # 如果从第0层开始
            if input_embeds is None:  # 如果没有提供输入嵌入
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)  # 通过词嵌入层
            else:  # 否则使用提供的嵌入
                forward_batch.hidden_states = input_embeds  # 使用输入嵌入
        # decoder layer  # 解码器层
        for i in range(start, end):  # 遍历分割区间内的层
            layer = self.model.layers[i]  # 获取当前层
            forward_batch.hidden_states, forward_batch.residual = layer(  # 通过当前层
                positions,  # 位置编码
                forward_batch.hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次信息
                forward_batch.residual,  # 残差
            )

        if end == self.model.config.num_hidden_layers:  # 如果到达最后一层
            # norm  # 层归一化
            hidden_states, _ = self.model.norm(  # 通过最终层归一化
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states  # 更新隐藏状态
            # logits process  # logits处理
            result = self.logits_processor(  # 通过logits处理器
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:  # 否则未到达最后一层
            result = None  # 结果为None

        return result  # 返回结果

    @property  # 属性装饰器
    def start_layer(self):  # 获取起始层索引
        return self.model.start_layer  # 返回模型的起始层

    @property  # 属性装饰器
    def end_layer(self):  # 获取结束层索引
        return self.model.end_layer  # 返回模型的结束层

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小
        return get_attention_sliding_window_size(self.config)  # 调用模块级函数

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # QKV投影中Q的映射
            (".qkv_proj", ".k_proj", "k"),  # QKV投影中K的映射
            (".qkv_proj", ".v_proj", "v"),  # QKV投影中V的映射
            (".gate_up_proj", ".gate_proj", 0),  # gate_up投影中gate的映射
            (".gate_up_proj", ".up_proj", 1),  # gate_up投影中up的映射
        ]

        params_dict = dict(self.named_parameters())  # 参数名字典

        for name, loaded_weight in weights:  # 遍历所有权重
            layer_id = get_layer_id(name)  # 获取权重所属的层ID
            if (  # 如果层ID不在当前流水线范围内
                layer_id is not None  # 层ID不为None
                and hasattr(self.model, "start_layer")  # 模型有start_layer属性
                and (  # 且
                    layer_id < self.model.start_layer  # 层ID小于起始层
                    or layer_id >= self.model.end_layer  # 或层ID大于等于结束层
                )
            ):
                continue  # 跳过
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 如果是旋转嵌入频率或投影器
                continue  # 跳过
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 如果是缓存的余弦/正弦值
                # Models trained using ColossalAI may include these tensors in  # 使用ColossalAI训练的模型可能在检查点中包含这些张量
                # the checkpoint. Skip them.  # 跳过它们
                continue  # 跳过
            if name.startswith("model.vision_tower") and name not in params_dict:  # 如果是视觉塔但不在参数字典中
                continue  # 跳过
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 如果绑定词嵌入且是lm_head权重
                continue  # 跳过
            # Handle FP8 kv-scale remapping  # 处理FP8 KV缩放名称重映射
            if "scale" in name:  # 如果名称包含"scale"
                name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射KV缩放名称
                if name is None:  # 如果重映射后为None
                    continue  # 跳过

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果分片名不在权重名中
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                    continue  # 跳过
                if name not in params_dict:  # 如果参数名不在字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果堆叠参数映射中没有匹配
                # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                    continue  # 跳过
                # Skip loading kv_scale from ckpts towards new design.  # 跳过从检查点加载旧设计的kv_scale
                if name.endswith(".kv_scale") and name not in params_dict:  # 如果是kv_scale但不在参数字典中
                    continue  # 跳过
                if name in params_dict.keys():  # 如果参数名在字典中
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认使用default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                else:  # 否则参数未找到
                    logger.warning(f"Parameter {name} not found in params_dict")  # 记录警告

    def get_weights_by_name(  # 根据名称获取权重
        self, name: str, truncate_size: int = 100, tp_size: int = 1  # 参数名，截断大小，张量并行大小
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.  # 根据名称获取参数权重，类似于HuggingFace的get_parameter

        Only used for unit test with an unoptimized performance.  # 仅用于单元测试，性能未优化
        For optimized performance, please use torch.save and torch.load.  # 优化性能请使用torch.save和torch.load
        """
        try:  # 尝试获取权重
            if name == "lm_head.weight" and self.config.tie_word_embeddings:  # 如果是lm_head权重且绑定词嵌入
                logger.info(  # 记录信息
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."  # 词嵌入已绑定，返回embed_tokens.weight作为lm_head.weight
                )
                return (  # 返回嵌入权重
                    self.model.embed_tokens.weight.cpu()  # 移到CPU
                    .to(torch.float32)  # 转为float32
                    .numpy()  # 转为numpy
                    .tolist()[:truncate_size]  # 截断
                )

            mapped_name = name  # 映射后的名称
            mapped_shard_id = None  # 映射后的分片ID
            for param_name, weight_name, shard_id in self.stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name in name:  # 如果分片名在名称中
                    mapped_name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                    mapped_shard_id = shard_id  # 设置分片ID
                    break  # 跳出循环
            params_dict = dict(self.named_parameters())  # 参数名字典
            param = params_dict[mapped_name]  # 获取参数
            if mapped_shard_id is not None:  # 如果有分片ID
                if mapped_shard_id in ["q", "k", "v"]:  # 如果是QKV分片
                    num_heads = self.config.num_attention_heads // tp_size  # 每个并行秩的注意力头数
                    num_kv_heads = self.config.num_key_value_heads // tp_size  # 每个并行秩的KV头数
                    head_dim = (  # 头维度
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    if mapped_shard_id == "q":  # 如果是Q分片
                        offset = 0  # 偏移为0
                        size = num_heads * head_dim  # 大小为头数乘头维度
                    elif mapped_shard_id == "k":  # 如果是K分片
                        offset = num_heads * head_dim  # 偏移为Q的大小
                        size = num_kv_heads * head_dim  # 大小为KV头数乘头维度
                    elif mapped_shard_id == "v":  # 如果是V分片
                        offset = (num_heads + num_kv_heads) * head_dim  # 偏移为Q和K的大小之和
                        size = num_kv_heads * head_dim  # 大小为KV头数乘头维度
                    weight = param.data.narrow(0, offset, size)  # 窄化获取指定范围的权重
                elif mapped_shard_id in [0, 1]:  # 如果是gate/up分片
                    intermediate_size = self.config.intermediate_size  # 中间层维度
                    slice_size = intermediate_size // tp_size  # 每个并行秩的切片大小
                    if mapped_shard_id == 0:  # gate_proj  # 如果是gate分片
                        offset = 0  # 偏移为0
                        size = slice_size  # 大小为切片大小
                    elif mapped_shard_id == 1:  # up_proj  # 如果是up分片
                        offset = slice_size  # 偏移为切片大小
                        size = slice_size  # 大小为切片大小

                    weight = param.data.narrow(0, offset, size)  # 窄化获取指定范围的权重
                else:  # 其他情况
                    weight = param.data  # 使用完整参数数据
            else:  # 如果没有分片ID
                weight = param.data  # 使用完整参数数据
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):  # 如果张量并行且是行并行层
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]  # 创建收集缓冲区
                torch.distributed.all_gather(gathered_weights, weight)  # 全收集权重
                weight = torch.cat(gathered_weights, dim=1)  # 拼接收集的权重
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]  # 返回截断后的权重

        except Exception:  # 捕获异常
            logger.error(  # 记录错误
                f"Error getting weights by name {name} in Exaone4ForCausalLM: {get_exception_traceback()}"  # 获取权重时出错
            )
            return None  # 返回None

    def get_embed_and_head(self):  # 获取词嵌入和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和LM头权重

    def set_embed_and_head(self, embed, head):  # 设置词嵌入和语言模型头权重
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重
        del self.lm_head.weight  # 删除旧的LM头权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重
        self.lm_head.weight = head  # 设置新的LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA操作

    def get_embed(self):  # 获取词嵌入权重
        return self.model.embed_tokens.weight  # 返回嵌入权重

    def set_embed(self, embed):  # 设置词嵌入权重
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3  # 注意：如果草稿模型隐藏维度与目标模型不同，则嵌入权重不能在EAGLE3中共享
        if (  # 如果
            hasattr(self.config, "target_hidden_size")  # 配置有target_hidden_size属性
            and self.config.target_hidden_size != self.config.hidden_size  # 且目标隐藏维度与当前不同
        ):
            return  # 直接返回，不设置
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA操作

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:  # 加载KV缓存缩放因子
        self.model.load_kv_cache_scales(quantization_param_path)  # 委托给模型主体


EntryClass = Exaone4ForCausalLM  # 模型入口类
