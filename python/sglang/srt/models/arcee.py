# Arcee基础模型（AFM）推理实现，兼容HuggingFace权重格式
# 该文件实现了Arcee Foundational Model的推理专用版本，主要特点包括：
# - 使用ReLU平方（relu2）激活函数替代Llama的SwiGLU
# - MLP使用单一上投影而非合并的gate/up投影
# - 支持张量并行和流水线并行
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License"); # 许可证：Apache 2.0
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at # 您可以在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不附带任何明示或暗示的保证
# See the License for the specific language governing permissions and # 请参阅许可证以了解管理权限和
# limitations under the License. # 限制的具体条款
# ==============================================================================
"""Inference-only Arcee Foundational Model (AFM) compatible with HuggingFace weights.""" # 仅推理的Arcee基础模型，兼容HuggingFace权重

import logging # 导入日志模块
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union # 导入类型提示

import torch # 导入PyTorch
from torch import nn # 导入神经网络模块
from transformers import LlamaConfig # 导入Llama配置类

from sglang.srt.distributed import ( # 导入分布式相关模块
    get_pp_group, # 获取流水线并行组
    get_tensor_model_parallel_rank, # 获取张量并行排名
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
)
from sglang.srt.layers.activation import get_act_fn # 导入激活函数获取工具
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS层归一化
from sglang.srt.layers.linear import ( # 导入线性层
    ColumnParallelLinear, # 列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput # 导入logits处理器
from sglang.srt.layers.pooler import Pooler, PoolingType # 导入池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope # 导入旋转位置编码获取工具
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id # 导入流水线并行缺失层和层ID获取工具
from sglang.srt.layers.vocab_parallel_embedding import ( # 导入词表并行嵌入
    ParallelLMHead, # 并行语言模型头
    VocabParallelEmbedding, # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors # 导入前向批次信息和流水线代理张量
from sglang.srt.model_loader.weight_utils import ( # 导入权重加载工具
    default_weight_loader, # 默认权重加载器
    kv_cache_scales_loader, # KV缓存缩放加载器
    maybe_remap_kv_scale_name, # 可能重映射KV缩放名称
)
from sglang.srt.server_args import get_global_server_args # 导入全局服务器参数
from sglang.srt.utils import add_prefix, make_layers # 导入前缀添加和层创建工具

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


class ArceeMLP(nn.Module):
    """
    MLP block for the Arcee model, using a ReLU-squared activation function.
    This differs from the Llama SwiGLU activation.
    """
    # Arcee模型的MLP块，使用ReLU平方激活函数，与Llama的SwiGLU激活不同

    def __init__( # MLP初始化方法
        self,
        hidden_size: int, # 隐藏层大小
        intermediate_size: int, # 中间层大小
        hidden_act: str, # 隐藏层激活函数名称
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
        reduce_results: bool = True, # 是否归约结果
    ) -> None:
        super().__init__() # 调用父类初始化
        # Arcee uses a single up-projection, not a merged gate/up projection. # Arcee使用单一上投影，而非合并的gate/up投影
        self.up_proj = ColumnParallelLinear( # 上投影线性层
            hidden_size, # 输入维度
            intermediate_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("up_proj", prefix), # 参数前缀
        )
        self.down_proj = RowParallelLinear( # 下投影线性层
            intermediate_size, # 输入维度
            hidden_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("down_proj", prefix), # 参数前缀
            reduce_results=reduce_results, # 是否归约结果
        )
        if hidden_act != "relu2": # 如果激活函数不是relu2
            raise ValueError( # 抛出值错误
                f"Unsupported activation: {hidden_act}. " # 不支持的激活函数
                "Arcee model in SGLang only supports 'relu2'." # SGLang中的Arcee模型仅支持relu2
            )
        # The activation function is relu(x)^2 # 激活函数为relu(x)的平方
        self.act_fn = get_act_fn("relu2") # 获取relu2激活函数

    def forward(self, x, forward_batch=None): # MLP前向传播
        x, _ = self.up_proj(x) # 上投影
        x = self.act_fn(x) # 应用激活函数
        x, _ = self.down_proj(x) # 下投影
        return x # 返回输出


class ArceeAttention(nn.Module): # Arcee注意力模块
    def __init__( # 注意力初始化方法
        self,
        config: LlamaConfig, # 模型配置
        hidden_size: int, # 隐藏层大小
        num_heads: int, # 注意力头数
        num_kv_heads: int, # KV头数
        layer_id: int = 0, # 层ID
        rope_theta: float = 10000, # 旋转位置编码theta
        rope_scaling: Optional[Dict[str, Any]] = None, # 旋转位置编码缩放
        rope_is_neox_style: bool = True, # 旋转位置编码是否为Neox风格
        max_position_embeddings: int = 8192, # 最大位置嵌入数
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
        bias: bool = False, # 是否使用偏置
    ) -> None:
        super().__init__() # 调用父类初始化
        self.hidden_size = hidden_size # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size() # 获取张量并行世界大小
        self.total_num_heads = num_heads # 总注意力头数
        assert self.total_num_heads % tp_size == 0 # 断言总头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size # 每个TP rank的头数
        self.total_num_kv_heads = num_kv_heads # 总KV头数
        if self.total_num_kv_heads >= tp_size: # 如果KV头数大于等于TP大小
            assert self.total_num_kv_heads % tp_size == 0 # 断言KV头数可被TP大小整除
        else: # 否则
            assert tp_size % self.total_num_kv_heads == 0 # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size) # 每个TP rank的KV头数
        self.head_dim = getattr(config, "head_dim", None) # 获取头维度
        if self.head_dim is None: # 如果头维度未指定
            self.head_dim = self.hidden_size // self.total_num_heads # 由隐藏大小和头数计算
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1) # 获取部分旋转因子
        self.rotary_dim = int(self.partial_rotary_factor * self.head_dim) # 旋转维度
        self.q_size = self.num_heads * self.head_dim # Q的大小
        self.kv_size = self.num_kv_heads * self.head_dim # KV的大小
        self.scaling = self.head_dim**-0.5 # 缩放因子
        self.rope_theta = rope_theta # 旋转位置编码theta
        self.max_position_embeddings = max_position_embeddings # 最大位置嵌入数

        self.qkv_proj = QKVParallelLinear( # QKV并行线性投影
            hidden_size, # 输入维度
            self.head_dim, # 头维度
            self.total_num_heads, # 总Q头数
            self.total_num_kv_heads, # 总KV头数
            bias=bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("qkv_proj", prefix), # 参数前缀
        )
        self.o_proj = RowParallelLinear( # 输出投影
            self.total_num_heads * self.head_dim, # 输入维度
            hidden_size, # 输出维度
            bias=bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("o_proj", prefix), # 参数前缀
        )

        self.rotary_emb = get_rope( # 获取旋转位置编码
            self.head_dim, # 头维度
            rotary_dim=self.rotary_dim, # 旋转维度
            max_position=max_position_embeddings, # 最大位置
            base=rope_theta, # 基础频率
            rope_scaling=rope_scaling, # 旋转缩放
            is_neox_style=rope_is_neox_style, # 是否为Neox风格
        )
        self.attn = RadixAttention( # 基数注意力
            self.num_heads, # 注意力头数
            self.head_dim, # 头维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数前缀
        )

    def forward( # 注意力前向传播
        self,
        positions: torch.Tensor, # 位置张量
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states) # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch) # 计算注意力
        output, _ = self.o_proj(attn_output) # 输出投影
        return output # 返回输出


class ArceeDecoderLayer(nn.Module): # Arcee解码器层
    def __init__( # 解码器层初始化方法
        self,
        config: LlamaConfig, # 模型配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.hidden_size = config.hidden_size # 保存隐藏层大小
        rope_theta = config.rope_parameters["rope_theta"] # 从配置获取旋转theta
        rope_scaling = config.rope_parameters # 旋转位置编码参数
        if rope_scaling is not None and getattr( # 如果有旋转缩放且有原始最大位置嵌入
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = ( # 设置原始最大位置嵌入
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True) # 获取旋转编码风格
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192) # 获取最大位置嵌入
        attention_bias = getattr(config, "attention_bias", False) or getattr( # 获取注意力偏置
            config, "bias", False
        )
        self.self_attn = ArceeAttention( # 自注意力层
            config=config, # 模型配置
            hidden_size=self.hidden_size, # 隐藏层大小
            num_heads=config.num_attention_heads, # 注意力头数
            num_kv_heads=config.num_key_value_heads, # KV头数
            layer_id=layer_id, # 层ID
            rope_theta=rope_theta, # 旋转theta
            rope_scaling=rope_scaling, # 旋转缩放
            rope_is_neox_style=rope_is_neox_style, # 旋转风格
            max_position_embeddings=max_position_embeddings, # 最大位置嵌入
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("self_attn", prefix), # 参数前缀
            bias=attention_bias, # 注意力偏置
        )
        self.mlp = ArceeMLP( # MLP层
            hidden_size=self.hidden_size, # 隐藏层大小
            intermediate_size=config.intermediate_size, # 中间层大小
            hidden_act=config.hidden_act, # 激活函数
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("mlp", prefix), # 参数前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 输入层归一化
        self.post_attention_layernorm = RMSNorm( # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward( # 解码器层前向传播
        self,
        positions: torch.Tensor, # 位置张量
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次
        residual: Optional[torch.Tensor], # 残差连接
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention # 自注意力
        if residual is None: # 如果没有残差
            residual = hidden_states # 残差等于隐藏状态
            hidden_states = self.input_layernorm(hidden_states) # 对隐藏状态做层归一化
        else: # 否则
            hidden_states, residual = self.input_layernorm(hidden_states, residual) # 带残差的层归一化
        hidden_states = self.self_attn( # 自注意力计算
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected # 全连接层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual) # 注意力后层归一化
        hidden_states = self.mlp(hidden_states) # MLP前向传播
        return hidden_states, residual # 返回隐藏状态和残差


class ArceeModel(nn.Module): # Arcee模型主体
    def __init__( # 模型初始化方法
        self,
        config: LlamaConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.padding_idx = config.pad_token_id # 填充token ID
        self.vocab_size = config.vocab_size # 词表大小
        self.pp_group = get_pp_group() # 获取流水线并行组
        if self.pp_group.is_first_rank: # 如果是第一个rank
            self.embed_tokens = VocabParallelEmbedding( # 词嵌入层
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏层大小
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("embed_tokens", prefix), # 参数前缀
            )
        else: # 否则
            self.embed_tokens = PPMissingLayer() # 使用缺失层占位

        self.layers, self.start_layer, self.end_layer = make_layers( # 创建解码器层
            config.num_hidden_layers, # 隐藏层数量
            lambda idx, prefix: ArceeDecoderLayer( # 解码器层工厂函数
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group, # 流水线并行rank
            pp_size=self.pp_group.world_size, # 流水线并行大小
            prefix="model.layers", # 参数前缀
        )

        if self.pp_group.is_last_rank: # 如果是最后一个rank
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 最终层归一化
        else: # 否则
            self.norm = PPMissingLayer(return_tuple=True) # 使用缺失层占位
        self.layers_to_capture = [] # 需要捕获隐藏状态的层列表

    def forward( # 模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        input_embeds: torch.Tensor = None, # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None, # 流水线代理张量
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank: # 如果是第一个rank
            if input_embeds is None: # 如果没有输入嵌入
                hidden_states = self.embed_tokens(input_ids) # 通过词嵌入获取隐藏状态
            else: # 否则
                hidden_states = input_embeds # 直接使用输入嵌入
            residual = None # 初始化残差为空
        else: # 否则
            assert pp_proxy_tensors is not None # 断言代理张量不为空
            hidden_states = pp_proxy_tensors["hidden_states"] # 从代理获取隐藏状态
            residual = pp_proxy_tensors["residual"] # 从代理获取残差

        aux_hidden_states = [] # 辅助隐藏状态列表
        for i in range(self.start_layer, self.end_layer): # 遍历每一层
            if i in self.layers_to_capture: # 如果需要捕获该层
                aux_hidden_states.append(hidden_states + residual) # 添加隐藏状态和残差之和
            layer = self.layers[i] # 获取当前层
            hidden_states, residual = layer( # 前向传播当前层
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank: # 如果不是最后一个rank
            return PPProxyTensors( # 返回代理张量
                {
                    "hidden_states": hidden_states, # 隐藏状态
                    "residual": residual, # 残差
                }
            )
        else: # 否则
            hidden_states, _ = self.norm(hidden_states, residual) # 最终层归一化

        if len(aux_hidden_states) == 0: # 如果没有辅助隐藏状态
            return hidden_states # 只返回隐藏状态

        return hidden_states, aux_hidden_states # 返回隐藏状态和辅助隐藏状态

    def load_kv_cache_scales(self, quantization_param_path: str) -> None: # 加载KV缓存缩放因子
        tp_size = get_tensor_model_parallel_world_size() # 获取TP大小
        tp_rank = get_tensor_model_parallel_rank() # 获取TP排名
        for layer_idx, scaling_factor in kv_cache_scales_loader( # 遍历KV缓存缩放
            quantization_param_path, # 量化参数路径
            tp_rank, # TP排名
            tp_size, # TP大小
            self.config.num_hidden_layers, # 隐藏层数量
            self.config.__class__.model_type, # 模型类型
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity): # 如果层不是恒等层
                layer_self_attn = self.layers[layer_idx].self_attn # 获取自注意力层

            if hasattr(layer_self_attn.attn, "k_scale"): # 如果注意力层有k_scale属性
                layer_self_attn.attn.k_scale = scaling_factor # 设置k缩放因子
                layer_self_attn.attn.v_scale = scaling_factor # 设置v缩放因子
            else: # 否则
                raise RuntimeError( # 抛出运行时错误
                    "Self attention has no KV cache scaling factor attribute!" # 自注意力没有KV缓存缩放因子属性
                )


class ArceeForCausalLM(nn.Module): # Arcee因果语言模型
    # BitandBytes specific attributes # BitandBytes特定属性
    default_bitsandbytes_target_modules = [ # 默认BitandBytes目标模块
        # Note: gate_proj is removed compared to Llama # 注意：与Llama相比，gate_proj被移除
        ".down_proj.", # 下投影
        ".up_proj.", # 上投影
        ".q_proj.", # Q投影
        ".k_proj.", # K投影
        ".v_proj.", # V投影
        ".o_proj.", # O投影
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1) # 在TP中，这些权重沿列维度分区
    column_parallel_weights_modules = [".down_proj.", ".o_proj."] # 列并行权重模块
    bitsandbytes_stacked_params_mapping = { # BitandBytes堆叠参数映射
        # shard_name, weight_name, index # 分片名称，权重名称，索引
        # Note: gate_proj and up_proj are removed as they are not stacked in ArceeMLP # 注意：gate_proj和up_proj被移除，因为在ArceeMLP中它们不是堆叠的
        ".q_proj": (".qkv_proj", 0), # Q投影映射
        ".k_proj": (".qkv_proj", 1), # K投影映射
        ".v_proj": (".qkv_proj", 2), # V投影映射
    }

    def __init__( # 因果语言模型初始化方法
        self,
        config: LlamaConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.pp_group = get_pp_group() # 获取流水线并行组
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix)) # 初始化模型
        # Arcee does not tie word embeddings # Arcee不绑定词嵌入
        self.lm_head = ParallelLMHead( # 语言模型头
            config.vocab_size, # 词表大小
            config.hidden_size, # 隐藏层大小
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("lm_head", prefix), # 参数前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head, # 是否使用注意力TP组
        )
        self.logits_processor = LogitsProcessor(config) # logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True) # 池化层
        # Parameters that are stacked in a single tensor in this model # 在此模型中堆叠为单个张量的参数
        self.stacked_params_mapping = [ # 堆叠参数映射
            # (param_name, shard_name, shard_id) # （参数名，分片名，分片ID）
            (".qkv_proj", ".q_proj", "q"), # QKV投影中Q的映射
            (".qkv_proj", ".k_proj", "k"), # QKV投影中K的映射
            (".qkv_proj", ".v_proj", "v"), # QKV投影中V的映射
        ]
        self.capture_aux_hidden_states = False # 是否捕获辅助隐藏状态

    def _init_model( # 初始化模型内部方法
        self,
        config: LlamaConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        return ArceeModel(config, quant_config=quant_config, prefix=prefix) # 返回ArceeModel实例

    @torch.no_grad() # 禁用梯度计算
    def forward( # 因果语言模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置张量
        forward_batch: ForwardBatch, # 前向批次
        input_embeds: torch.Tensor = None, # 输入嵌入
        get_embedding: bool = False, # 是否获取嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None, # 流水线代理张量
    ) -> LogitsProcessorOutput:
        hidden_states = self.model( # 通过模型获取隐藏状态
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None # 辅助隐藏状态初始化为空
        if self.capture_aux_hidden_states: # 如果需要捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states # 拆分隐藏状态

        if self.pp_group.is_last_rank: # 如果是最后一个rank
            if not get_embedding: # 如果不获取嵌入
                return self.logits_processor( # 通过logits处理器返回
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else: # 否则
                return self.pooler(hidden_states, forward_batch) # 通过池化层返回
        else: # 否则
            return hidden_states # 返回隐藏状态

    @property
    def start_layer(self): # 起始层属性
        return self.model.start_layer # 返回模型的起始层

    @property
    def end_layer(self): # 结束层属性
        return self.model.end_layer # 返回模型的结束层

    def get_input_embeddings(self) -> nn.Embedding: # 获取输入嵌入
        return self.model.embed_tokens # 返回词嵌入层

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载权重
        params_dict = dict(self.named_parameters()) # 获取参数字典

        for name, loaded_weight in weights: # 遍历所有权重
            layer_id = get_layer_id(name) # 获取层ID
            if ( # 如果层不在当前PP范围内则跳过
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue # 跳过
            if "rotary_emb.inv_freq" in name or "projector" in name: # 跳过旋转频率和投影器
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name: # 跳过旋转缓存
                continue

            # Handle FP8 kv-scale remapping # 处理FP8 KV缩放重映射
            if "scale" in name: # 如果名称包含scale
                name = maybe_remap_kv_scale_name(name, params_dict) # 可能重映射KV缩放名称
                if name is None: # 如果重映射后为空
                    continue # 跳过

            is_stacked = False # 是否为堆叠参数
            for param_name, weight_name, shard_id in self.stacked_params_mapping: # 遍历堆叠参数映射
                if weight_name not in name: # 如果权重名不在参数名中
                    continue # 跳过

                name = name.replace(weight_name, param_name) # 替换权重名为参数名
                if name not in params_dict: # 如果参数名不在参数字典中
                    continue # 跳过

                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重
                is_stacked = True # 标记为堆叠参数
                break # 跳出循环

            if not is_stacked: # 如果不是堆叠参数
                if name in params_dict: # 如果参数名在参数字典中
                    param = params_dict[name] # 获取参数
                    weight_loader = getattr( # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight) # 加载权重
                else: # 否则
                    logger.warning(f"Parameter {name} not found in model.") # 记录参数未找到警告

    def load_kv_cache_scales(self, quantization_param_path: str) -> None: # 加载KV缓存缩放因子
        self.model.load_kv_cache_scales(quantization_param_path) # 委托给模型加载


EntryClass = [ArceeForCausalLM] # 入口类列表
