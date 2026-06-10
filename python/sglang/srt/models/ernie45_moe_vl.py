# 文件说明：Ernie4.5 VL MoE多模态模型实现，兼容baidu/ERNIE-4.5-VL-*-PT权重
# 包含视觉语言MoE注意力、MoE层（支持文本/视觉双路由）、解码器层及模型主体

# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. # 详见License限制条件
# ==============================================================================

"""Inference-only Ernie4.5 VL model compatible with baidu/ERNIE-4.5-VL-*-PT weights.""" # 仅推理的Ernie4.5 VL模型，兼容百度ERNIE-4.5-VL-*-PT权重

import logging # 导入日志模块
from itertools import islice # 导入迭代器切片工具
from typing import Any, Dict, Optional, Tuple, Union # 导入类型提示工具

import torch # 导入PyTorch库
from torch import nn # 导入神经网络模块
from transformers import PretrainedConfig # 导入预训练配置基类

from sglang.srt.distributed import (
    get_pp_group, # 获取流水线并行组
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
    tensor_model_parallel_all_reduce, # 张量并行全归约操作
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled # 导入DP注意力使能判断
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.layers.linear import (
    QKVParallelLinear, # 导入QKV并行线性层
    ReplicatedLinear, # 导入复制线性层
    RowParallelLinear, # 导入行并行线性层
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class # 导入MoE实现类获取函数
from sglang.srt.layers.moe.topk import TopK # 导入TopK选择层
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import Ernie4_5_VLRotaryEmbedding # 导入Ernie4.5 VL旋转嵌入
from sglang.srt.layers.utils import PPMissingLayer # 导入流水线并行缺失层工具
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors # 导入前向批信息和PP代理张量
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP as Ernie4_5_VLMoeMLP # 导入DeepseekV2 MLP作为Ernie4.5 VL MoE MLP
from sglang.srt.utils import add_prefix, make_layers # 导入工具函数

logger = logging.getLogger(__name__) # 获取当前模块日志记录器


class Ernie4_5_VLMoeAttention(nn.Module): # Ernie4.5 VL MoE注意力模块，支持3D旋转位置编码
    def __init__( # Ernie4.5 VL MoE注意力初始化
        self,
        config: PretrainedConfig, # 预训练配置
        hidden_size: int, # 隐藏维度
        num_heads: int, # 注意力头数
        num_kv_heads: int, # KV头数
        layer_id: int = 0, # 层ID
        rope_theta: float = 10000, # RoPE基频
        rope_scaling: Optional[Dict[str, Any]] = None, # RoPE缩放配置
        rope_is_neox_style: bool = True, # RoPE是否为Neox风格
        freq_allocation: int = 20, # 频率分配（时间维度）
        max_position_embeddings: int = 8192, # 最大位置编码数
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
        bias: bool = False, # 是否使用偏置
    ) -> None:
        super().__init__() # 调用父类初始化
        self.hidden_size = hidden_size # 保存隐藏维度
        tp_size = get_tensor_model_parallel_world_size() # 获取张量并行世界大小
        self.total_num_heads = num_heads # 保存总注意力头数
        assert self.total_num_heads % tp_size == 0 # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size # 计算每个TP的头数
        self.total_num_kv_heads = num_kv_heads # 保存总KV头数
        if self.total_num_kv_heads >= tp_size: # KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs. # KV头数大于TP大小，跨多个TP GPU分区KV头
            assert self.total_num_kv_heads % tp_size == 0 # 断言KV头数可被TP大小整除
        else: # KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs. # KV头数小于TP大小，跨多个TP GPU复制KV头
            assert tp_size % self.total_num_kv_heads == 0 # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size) # 计算每个TP的KV头数
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo # MistralConfig有一个可选的head_dim由Mistral-Nemo引入
        self.head_dim = getattr( # 获取头维度
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1) # 获取部分旋转因子
        self.rotary_dim = int(partial_rotary_factor * self.head_dim) # 计算旋转维度
        self.q_size = self.num_heads * self.head_dim # Q大小
        self.kv_size = self.num_kv_heads * self.head_dim # KV大小
        self.scaling = self.head_dim**-0.5 # 缩放因子
        self.rope_theta = rope_theta # 保存RoPE基频
        self.max_position_embeddings = max_position_embeddings # 保存最大位置编码数

        self.qkv_proj = QKVParallelLinear( # 创建QKV并行投影
            hidden_size, # 输入维度
            self.head_dim, # 头维度
            self.total_num_heads, # 总Q头数
            self.total_num_kv_heads, # 总KV头数
            bias=bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("qkv_proj", prefix), # 参数前缀
        )
        self.o_proj = RowParallelLinear( # 创建输出投影
            self.total_num_heads * self.head_dim, # 输入维度
            hidden_size, # 输出维度
            bias=bias, # 是否使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("o_proj", prefix), # 参数前缀
        )

        # 3D rope # 3D旋转位置编码
        t_rope = freq_allocation # 时间维度频率分配
        h_rope = (self.head_dim // 2 - freq_allocation) // 2 # 高度维度频率分配
        w_rope = (self.head_dim // 2 - freq_allocation) // 2 # 宽度维度频率分配

        self.rotary_emb = Ernie4_5_VLRotaryEmbedding( # 创建Ernie4.5 VL旋转位置嵌入
            head_size=self.head_dim, # 头维度
            rotary_dim=self.head_dim, # 旋转维度
            max_position_embeddings=max_position_embeddings, # 最大位置编码数
            base=rope_theta, # 基频
            is_neox_style=False, # 非Neox风格
            dtype=torch.get_default_dtype(), # 默认数据类型
            mrope_section=[h_rope, w_rope, t_rope], # 多模态RoPE分段：高度、宽度、时间
        )
        self.attn = RadixAttention( # 创建基数注意力
            self.num_heads, # 注意力头数
            self.head_dim, # 头维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数前缀
        )

    def forward( # Ernie4.5 VL MoE注意力前向传播
        self,
        positions: torch.Tensor, # 位置编码
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批信息
    ) -> torch.Tensor: # 返回注意力输出
        qkv, _ = self.qkv_proj(hidden_states) # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch) # 计算注意力
        output, _ = self.o_proj(attn_output) # 输出投影
        return output # 返回输出


class Ernie4_5_VLMoeMoE(nn.Module): # Ernie4.5 VL MoE模块，支持文本和视觉双路由专家
    def __init__( # Ernie4.5 VL MoE初始化
        self,
        config: PretrainedConfig, # 预训练配置
        layer_id: int, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.layer_id = layer_id # 保存层ID
        self.tp_size = get_tensor_model_parallel_world_size() # 获取张量并行世界大小
        self.moe_num_shared_experts = getattr(config, "moe_num_shared_experts", 0) # 获取共享专家数
        self.hidden_size = config.hidden_size # 保存隐藏维度

        moe_num_experts = config.moe_num_experts # 获取各模态专家数列表
        max_moe_num_experts = max(moe_num_experts) # 获取最大专家数

        if self.tp_size > max_moe_num_experts: # TP大小大于最大专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {moe_num_experts}." # TP大小不能大于专家数
            )

        moe_layer_start_index = config.moe_layer_start_index # 获取MoE层起始索引列表
        text_moe_layer_start_index = moe_layer_start_index[0] # 文本MoE层起始索引
        vision_moe_layer_start_index = moe_layer_start_index[1] # 视觉MoE层起始索引
        moe_layer_end_index = config.moe_layer_end_index # 获取MoE层结束索引
        moe_layer_end_index = getattr( # 获取MoE层结束索引（带默认值）
            config,
            "moe_layer_end_index",
            [config.num_hidden_layers - 1, config.num_hidden_layers - 1],
        )
        text_moe_layer_end_index = moe_layer_end_index[0] # 文本MoE层结束索引
        vision_moe_layer_end_index = moe_layer_end_index[1] # 视觉MoE层结束索引

        assert config.moe_num_experts[0] == config.moe_num_experts[1] # 断言文本和视觉专家数相等
        self.e_score_correction_bias = nn.Parameter( # 专家分数校正偏置
            torch.empty(2, config.moe_num_experts[0], dtype=torch.float32) # 形状：(2, 专家数)，分别对应文本和视觉
        )

        assert text_moe_layer_start_index <= text_moe_layer_end_index # 断言文本MoE层索引有效

        if ( # 当前层在文本MoE层范围内
            layer_id >= text_moe_layer_start_index
            and layer_id <= text_moe_layer_end_index
        ):
            self.text_experts_gate = ReplicatedLinear( # 文本专家门控
                config.hidden_size, # 输入维度
                config.moe_num_experts[0], # 输出维度（专家数）
                bias=False, # 无偏置
                params_dtype=torch.float32, # 参数数据类型为float32
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("text_experts_gate", prefix), # 参数前缀
            )

            self.text_experts_topk = TopK( # 文本专家TopK选择
                top_k=config.moe_k, # Top-K值
                renormalize=True, # 启用重归一化
                use_grouped_topk=False, # 不使用分组TopK
                correction_bias=self.e_score_correction_bias[0], # 文本校正偏置
            )

            self.text_experts = get_moe_impl_class(quant_config)( # 文本专家网络
                num_experts=config.moe_num_experts[0], # 文本专家数
                top_k=config.moe_k, # Top-K值
                hidden_size=config.hidden_size, # 隐藏维度
                intermediate_size=config.moe_intermediate_size[0], # 文本中间维度
                layer_id=self.layer_id, # 层ID
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("text_experts", prefix), # 参数前缀
            )

        assert vision_moe_layer_start_index <= vision_moe_layer_end_index # 断言视觉MoE层索引有效
        if ( # 当前层在视觉MoE层范围内
            layer_id >= vision_moe_layer_start_index
            and layer_id <= vision_moe_layer_end_index
        ):

            self.vision_experts_gate = ReplicatedLinear( # 视觉专家门控
                config.hidden_size, # 输入维度
                config.moe_num_experts[1], # 输出维度（专家数）
                bias=False, # 无偏置
                params_dtype=torch.float32, # 参数数据类型为float32
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("vision_experts_gate", prefix), # 参数前缀
            )

            self.vision_experts_topk = TopK( # 视觉专家TopK选择
                top_k=config.moe_k, # Top-K值
                renormalize=True, # 启用重归一化
                use_grouped_topk=False, # 不使用分组TopK
                correction_bias=self.e_score_correction_bias[1], # 视觉校正偏置
            )

            self.vision_experts = get_moe_impl_class(quant_config)( # 视觉专家网络
                num_experts=config.moe_num_experts[1], # 视觉专家数
                top_k=config.moe_k, # Top-K值
                hidden_size=config.hidden_size, # 隐藏维度
                intermediate_size=config.moe_intermediate_size[1], # 视觉中间维度
                layer_id=self.layer_id, # 层ID
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("vision_experts", prefix), # 参数前缀
            )

        if self.moe_num_shared_experts > 0: # 如果存在共享专家
            intermediate_size = ( # 计算共享专家中间维度
                config.moe_intermediate_size[0] * config.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_VLMoeMLP( # 创建共享专家MLP
                hidden_size=config.hidden_size, # 隐藏维度
                intermediate_size=intermediate_size, # 中间维度
                hidden_act=config.hidden_act, # 激活函数
                quant_config=quant_config, # 量化配置
                reduce_results=False, # 不归约结果
                prefix=add_prefix("shared_experts", prefix), # 参数前缀
            )

    def forward( # Ernie4.5 VL MoE前向传播，支持文本/视觉双路由
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        visual_token_mask: torch.Tensor, # 视觉token掩码
        **kwargs: object, # 其他参数
    ) -> torch.Tensor: # 返回最终隐藏状态
        shared_output = ( # 计算共享专家输出
            self.shared_experts(hidden_states)
            if self.moe_num_shared_experts > 0
            else None # 无共享专家时为None
        )

        orig_shape = hidden_states.shape # 保存原始形状
        hidden_dim = hidden_states.shape[-1] # 获取隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim) # 展平为2D

        capturing = torch.cuda.is_current_stream_capturing() # 判断是否在CUDA Graph捕获中

        if visual_token_mask is not None and not capturing: # 有视觉掩码且非CUDA Graph捕获
            all_visual = visual_token_mask.all() # 是否全部为视觉token
            any_visual = visual_token_mask.any() # 是否存在视觉token
        else:
            # During CUDA Graph capture, all set false # CUDA Graph捕获期间，全部设为False
            all_visual = False # 不认为全部为视觉
            any_visual = False # 不认为存在视觉

        if all_visual: # 全部为视觉token
            # vision modal input processing directly # 视觉模态输入直接处理
            vision_router_logits, _ = self.vision_experts_gate( # 视觉专家门控
                hidden_states.to(dtype=torch.float32) # 转为float32计算
            )
            vision_topk_output = self.vision_experts_topk( # 视觉专家TopK选择
                hidden_states, vision_router_logits
            )
            final_hidden_states = self.vision_experts( # 视觉专家计算
                hidden_states=hidden_states, topk_output=vision_topk_output
            )
        elif any_visual: # 混合文本和视觉token
            visual_token_mask = visual_token_mask.repeat(1, self.hidden_size).bool() # 扩展视觉掩码到隐藏维度
            text_token_mask = ~visual_token_mask # 文本掩码为视觉掩码取反
            final_hidden_states = torch.zeros_like(hidden_states) # 初始化最终隐藏状态

            text_hidden_states = hidden_states[text_token_mask].reshape( # 提取文本隐藏状态
                -1, self.hidden_size
            )
            vision_hidden_states = hidden_states[visual_token_mask].reshape( # 提取视觉隐藏状态
                -1, self.hidden_size
            )

            text_router_logits, _ = self.text_experts_gate( # 文本专家门控
                text_hidden_states.to(dtype=torch.float32) # 转为float32计算
            )
            text_topk_output = self.text_experts_topk( # 文本专家TopK选择
                text_hidden_states, text_router_logits
            )
            final_hidden_states[text_token_mask] = self.text_experts( # 文本专家计算并填回
                hidden_states=text_hidden_states, topk_output=text_topk_output
            ).flatten()

            vision_router_logits, _ = self.vision_experts_gate( # 视觉专家门控
                vision_hidden_states.to(dtype=torch.float32) # 转为float32计算
            )
            vision_topk_output = self.vision_experts_topk( # 视觉专家TopK选择
                vision_hidden_states, vision_router_logits
            )
            final_hidden_states[visual_token_mask] = self.vision_experts( # 视觉专家计算并填回
                hidden_states=vision_hidden_states, topk_output=vision_topk_output
            ).flatten()

        else: # 全部为文本token
            # text modal input processing directly # 文本模态输入直接处理
            text_router_logits, _ = self.text_experts_gate( # 文本专家门控
                hidden_states.to(dtype=torch.float32) # 转为float32计算
            )
            topk_output = self.text_experts_topk(hidden_states, text_router_logits) # 文本专家TopK选择
            final_hidden_states = self.text_experts( # 文本专家计算
                hidden_states=hidden_states, topk_output=topk_output
            )

        if shared_output is not None: # 如果存在共享专家输出
            final_hidden_states = final_hidden_states + shared_output # 加上共享专家输出

        if self.tp_size > 1: # 张量并行度大于1
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states) # 全归约同步

        return final_hidden_states.view(orig_shape) # 恢复原始形状并返回


class Ernie4_5_VLMoeDecoderLayer(nn.Module): # Ernie4.5 VL MoE解码器层
    """A single transformer layer. # 单个Transformer层

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size. # Transformer层接收[s, b, h]大小的输入并返回相同大小的输出
    """

    def __init__( # Ernie4.5 VL MoE解码器层初始化
        self,
        config, # 模型配置
        layer_id: int, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        rope_theta = config.rope_parameters["rope_theta"] # 获取RoPE基频
        rope_scaling = config.rope_parameters # 获取RoPE缩放参数
        rope_is_neox_style = getattr(config, "rope_is_neox_style", False) # 获取RoPE是否为Neox风格
        freq_allocation = getattr(config, "freq_allocation", 20) # 获取频率分配，默认20
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072) # 获取最大位置编码数，默认131072
        # Self attention. # 自注意力
        self.self_attn = Ernie4_5_VLMoeAttention( # 创建VL MoE注意力层
            config=config, # 模型配置
            hidden_size=config.hidden_size, # 隐藏维度
            num_heads=config.num_attention_heads, # 注意力头数
            num_kv_heads=config.num_key_value_heads, # KV头数
            layer_id=layer_id, # 层ID
            rope_theta=rope_theta, # RoPE基频
            rope_scaling=rope_scaling, # RoPE缩放
            rope_is_neox_style=rope_is_neox_style, # RoPE Neox风格
            freq_allocation=freq_allocation, # 频率分配
            max_position_embeddings=config.max_position_embeddings, # 最大位置编码数
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("self_attn", prefix), # 参数前缀
            bias=config.use_bias, # 是否使用偏置
        )

        # MoE # MoE层
        moe_layer_start_index = config.moe_layer_start_index # 获取MoE层起始索引列表
        min_moe_layer_start_index = min(moe_layer_start_index) # 最小MoE层起始索引
        moe_layer_end_index = getattr( # 获取MoE层结束索引
            config,
            "moe_layer_end_index",
            [config.num_hidden_layers - 1, config.num_hidden_layers - 1],
        )
        max_moe_layer_end_index = max(moe_layer_end_index) # 最大MoE层结束索引
        assert min_moe_layer_start_index <= max_moe_layer_end_index # 断言MoE层索引有效
        moe_num_experts = config.moe_num_experts # 获取各模态专家数列表
        max_moe_num_experts = max(moe_num_experts) # 最大专家数
        moe_layer_interval = getattr(config, "moe_layer_interval", 1) # 获取MoE层间隔，默认1
        use_moe = getattr(config, "use_moe", max_moe_num_experts > 0) # 是否使用MoE
        # MLP # MLP模块
        if ( # 满足MoE层条件
            use_moe
            and ((layer_id + 1) % moe_layer_interval == 0) # 层间隔条件
            and layer_id >= min_moe_layer_start_index # 不小于最小起始索引
            and layer_id <= max_moe_layer_end_index # 不大于最大结束索引
        ):
            self.mlp = Ernie4_5_VLMoeMoE( # 使用VL MoE
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else: # 使用普通MLP
            self.mlp = Ernie4_5_VLMoeMLP( # 使用普通MLP
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 输入层归一化
        self.post_attention_layernorm = RMSNorm( # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward( # Ernie4.5 VL MoE解码器层前向传播
        self,
        positions: torch.Tensor, # 位置编码
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批信息
        residual: Optional[torch.Tensor], # 残差连接
        visual_token_mask: torch.Tensor | None, # 视觉token掩码
        **kwargs: object, # 其他参数
    ) -> Tuple[torch.Tensor, torch.Tensor]: # 返回隐藏状态和残差
        # Self Attention # 自注意力
        if residual is None: # 无残差（第一层）
            residual = hidden_states # 保存隐藏状态作为残差
            hidden_states = self.input_layernorm(hidden_states) # 层归一化
        else: # 有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual) # 层归一化并更新残差
        hidden_states = self.self_attn( # 自注意力计算
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected # 全连接层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual) # 注意力后层归一化
        if isinstance(self.mlp, Ernie4_5_VLMoeMoE): # MLP为MoE类型
            hidden_states = self.mlp(hidden_states, visual_token_mask, **kwargs) # MoE前向传播，传入视觉掩码
        else: # 普通MLP
            hidden_states = self.mlp(hidden_states) # 普通MLP前向传播

        return hidden_states, residual # 返回隐藏状态和残差


# only used as text backbone for ernie4.5 vl # 仅用作Ernie4.5 VL的文本主干
class Ernie4_5_VLMoeModel(nn.Module): # Ernie4.5 VL MoE模型主体，支持流水线并行
    def __init__( # Ernie4.5 VL MoE模型初始化
        self,
        config: PretrainedConfig, # 预训练配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.pp_group = get_pp_group() # 获取流水线并行组

        if self.pp_group.is_first_rank: # 如果是流水线第一秩
            self.embed_tokens = VocabParallelEmbedding( # 创建词表并行嵌入层
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏维度
                enable_tp=not is_dp_attention_enabled(), # DP注意力未启用时开启TP
                prefix=add_prefix("embed_tokens", prefix), # 参数前缀
            )
        else: # 非第一秩
            self.embed_tokens = PPMissingLayer() # 使用缺失层占位

        self.layers, self.start_layer, self.end_layer = make_layers( # 创建解码器层列表，获取起止层索引
            config.num_hidden_layers, # 隐藏层数
            lambda idx, prefix: Ernie4_5_VLMoeDecoderLayer( # 每层的构造函数
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group, # 流水线并行秩
            pp_size=self.pp_group.world_size, # 流水线并行世界大小
            prefix=add_prefix("layers", prefix), # 参数前缀
        )
        if self.pp_group.is_last_rank: # 如果是流水线最后秩
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 创建最终归一化层
        else: # 非最后秩
            self.norm = PPMissingLayer(return_tuple=True) # 使用缺失层占位（返回元组）

    def get_input_embeddings(self) -> torch.Tensor: # 获取输入嵌入层
        return self.embed_tokens

    @torch.no_grad() # 禁用梯度计算
    def forward( # Ernie4.5 VL MoE模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None, # 流水线代理张量
        visual_token_mask: torch.Tensor | None = None, # 视觉token掩码
    ) -> Union[torch.Tensor, PPProxyTensors]: # 返回隐藏状态或流水线代理张量

        if self.pp_group.is_first_rank: # 流水线第一秩
            if input_embeds is None: # 无输入嵌入
                hidden_states = self.embed_tokens(input_ids) # 通过嵌入层获取隐藏状态
            else: # 有输入嵌入
                hidden_states = input_embeds # 直接使用输入嵌入
            residual = None # 初始化残差为None
        else: # 非第一秩
            assert pp_proxy_tensors is not None # 断言流水线代理张量不为None
            hidden_states = pp_proxy_tensors["hidden_states"] # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"] # 从代理张量获取残差

        for layer in islice(self.layers, self.start_layer, self.end_layer): # 遍历本秩负责的层
            hidden_states, residual = layer( # 前向传播每一层
                positions,
                hidden_states,
                forward_batch,
                residual,
                visual_token_mask,
            )

        if not self.pp_group.is_last_rank: # 非最后秩
            return PPProxyTensors( # 返回流水线代理张量
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if hidden_states.shape[0] != 0: # 隐藏状态非空
            if residual is None: # 无残差
                hidden_states = self.norm(hidden_states) # 归一化
            else: # 有残差
                hidden_states, _ = self.norm(hidden_states, residual) # 归一化并处理残差

        return hidden_states # 返回隐藏状态
