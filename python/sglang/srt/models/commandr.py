# Cohere Command-R模型实现文件 - 实现Cohere和Cohere2因果语言模型的SGLang推理
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/commandr.py  # 改编自vLLM的commandr实现

# Copyright 2023-2024 SGLang Team
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
# limitations under the License.
# ==============================================================================
# Copyright 2024 Cohere and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX  # 本代码基于EleutherAI的GPT-NeoX库和GPT-NeoX
# and OPT implementations in this library. It has been modified from its  # 和本库中的OPT实现。已从原始
# original forms to accommodate minor architectural differences compared  # 形式修改，以适应相比于
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.  # Meta AI团队训练模型所用的GPT-NeoX和OPT的微小架构差异
#
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
# limitations under the License.
# ==============================================================================

# Adapted from  # 改编自
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/commandr.py#L1  # vLLM commandr参考链接

# This file is based on the LLama model definition file in transformers  # 本文件基于transformers中的LLama模型定义文件
"""PyTorch Cohere model."""  # PyTorch Cohere模型

from typing import Iterable, Optional, Tuple # 导入类型提示模块

import torch # 导入PyTorch深度学习框架
import torch.utils.checkpoint # 导入梯度检查点工具
from torch import nn # 导入神经网络模块
from torch.nn.parameter import Parameter # 导入参数类
from transformers import Cohere2Config, CohereConfig, PretrainedConfig # 导入Cohere配置类

from sglang.srt.distributed import ( # 导入分布式相关函数
    get_tensor_model_parallel_rank, # 获取张量并行排名
    get_tensor_model_parallel_world_size, # 获取张量并行世界大小
)
from sglang.srt.layers.activation import SiluAndMul # 导入SiLU与乘法激活函数
from sglang.srt.layers.linear import ( # 导入并行线性层
    MergedColumnParallelLinear, # 合并列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入逻辑处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import ( # 导入权重加载工具
    default_weight_loader, # 默认权重加载器
    maybe_remap_kv_scale_name, # 可能重映射KV缩放名称
)
from sglang.srt.utils import add_prefix, get_compiler_backend, set_weight_attrs # 导入工具函数


@torch.compile(backend=get_compiler_backend()) # 使用编译器后端编译优化
def layer_norm_func(hidden_states, weight, variance_epsilon): # 层归一化函数：计算归一化后的隐藏状态
    input_dtype = hidden_states.dtype # 保存输入数据类型
    hidden_states = hidden_states.to(torch.float32) # 转换为FP32精度计算
    mean = hidden_states.mean(-1, keepdim=True) # 计算均值
    variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True) # 计算方差
    hidden_states = (hidden_states - mean) * torch.rsqrt(variance + variance_epsilon) # 归一化
    hidden_states = weight.to(torch.float32) * hidden_states # 乘以缩放权重
    return hidden_states.to(input_dtype) # 转换回原始数据类型


class LayerNorm(nn.Module): # 层归一化类，支持张量并行的权重加载
    def __init__(self, param_shape=None, eps=1e-5): # 初始化层归一化
        super().__init__() # 调用父类初始化
        self.weight = nn.Parameter(torch.ones(param_shape)) # 可学习缩放权重，初始化为1
        self.variance_epsilon = eps # 方差epsilon
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader}) # 设置权重加载器属性

    def forward(self, hidden_states, residuals=None): # 前向传播：层归一化
        hidden_states = layer_norm_func( # 调用编译优化的层归一化函数
            hidden_states, self.weight, self.variance_epsilon # 隐藏状态、权重和epsilon
        )
        return hidden_states, residuals # 返回归一化后的状态和残差

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor): # 权重加载器：支持张量并行分片
        tp_rank = get_tensor_model_parallel_rank() # 获取当前张量并行排名
        shard_dim = 0 if param.dim() != 1 else None # 如果参数不是1维则按第0维分片
        param_data = param.data # 获取参数数据
        if shard_dim is not None: # 如果需要分片
            shard_size = param_data.shape[shard_dim] # 获取分片大小
            start_idx = tp_rank * shard_size # 计算起始索引
            loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size) # 截取对应的分片
        assert param_data.shape == loaded_weight.shape # 确保形状匹配
        param_data.copy_(loaded_weight) # 复制权重数据


# Copied from transformers.models.llama.modeling_llama.LlamaMLP Llama->Cohere  # 从transformers的LlamaMLP复制，Llama改为Cohere
class CohereMLP(nn.Module): # Cohere MLP类，实现前馈神经网络
    def __init__( # 初始化Cohere MLP层
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.hidden_size = config.hidden_size # 隐藏维度
        self.intermediate_size = config.intermediate_size # 中间层维度
        self.gate_up_proj = MergedColumnParallelLinear( # gate和up合并投影层
            self.hidden_size, # 输入维度
            [self.intermediate_size] * 2, # 输出维度（gate和up各一个intermediate_size）
            bias=False, # 不使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("gate_up_proj", prefix), # 参数名前缀
        )
        self.down_proj = RowParallelLinear( # 降维投影层
            self.intermediate_size, # 输入维度
            self.hidden_size, # 输出维度
            bias=False, # 不使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("down_proj", prefix), # 参数名前缀
        )
        self.act_fn = SiluAndMul() # SiLU与乘法激活函数

    def forward(self, x): # 前向传播：gate_up→激活→降维
        gate_up, _ = self.gate_up_proj(x) # 通过gate_up投影
        x = self.act_fn(gate_up) # 通过激活函数
        x, _ = self.down_proj(x) # 通过降维投影
        return x # 返回输出


class CohereAttention(nn.Module): # Cohere注意力层类，实现多头注意力机制
    def __init__( # 初始化Cohere注意力层
        self,
        config: PretrainedConfig, # 模型配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        tp_size = get_tensor_model_parallel_world_size() # 获取张量并行大小
        self.config = config # 保存配置
        self.attention_dropout = config.attention_dropout # 注意力dropout率
        self.hidden_size = config.hidden_size # 隐藏维度
        self.total_num_heads = config.num_attention_heads # 总注意力头数
        self.num_heads = self.total_num_heads // tp_size # 每个并行分片的头数
        self.head_dim = self.hidden_size // self.total_num_heads # 每个头的维度
        self.total_num_kv_heads = config.num_key_value_heads # 总KV头数
        if self.total_num_kv_heads >= tp_size: # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此在多个张量并行GPU间分配
            # the KV heads across multiple tensor parallel GPUs.  # KV头
            assert self.total_num_kv_heads % tp_size == 0 # 确保可整除
        else: # 否则
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此在多个张量并行GPU间复制
            # the KV heads across multiple tensor parallel GPUs.  # KV头
            assert tp_size % self.total_num_kv_heads == 0 # 确保可整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size) # 每个并行分片的KV头数
        self.q_size = self.num_heads * self.head_dim # Q的总维度
        self.kv_size = self.num_kv_heads * self.head_dim # KV的总维度
        self.scaling = self.head_dim**-0.5 # 缩放因子
        self.max_position_embeddings = getattr( # 最大位置嵌入数
            config, "model_max_length", None # 优先使用model_max_length
        ) or getattr(config, "max_position_embeddings", 8192) # 其次使用max_position_embeddings，默认8192
        self.rope_theta = config.rope_parameters["rope_theta"] # RoPE的theta参数
        self.rope_scaling = config.rope_parameters # RoPE缩放参数
        self.use_qk_norm = getattr(config, "use_qk_norm", False) # 是否使用QK归一化
        self.qkv_proj = QKVParallelLinear( # QKV联合投影层
            self.hidden_size, # 输入维度
            self.head_dim, # 每个头的维度
            self.total_num_heads, # 总Q头数
            self.total_num_kv_heads, # 总KV头数
            bias=False, # 不使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("qkv_proj", prefix), # 参数名前缀
        )
        self.o_proj = RowParallelLinear( # 输出投影层
            self.total_num_heads * self.head_dim, # 输入维度
            self.hidden_size, # 输出维度
            bias=False, # 不使用偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("o_proj", prefix), # 参数名前缀
        )
        self.rotary_emb = get_rope( # 旋转位置编码
            self.head_dim, # 每个头的维度
            rotary_dim=self.head_dim, # 旋转维度
            max_position=self.max_position_embeddings, # 最大位置数
            base=self.rope_theta, # 基数
            rope_scaling=self.rope_scaling, # 缩放参数
            is_neox_style=False, # 非NeoX风格
        )

        self.v1 = isinstance(config, CohereConfig) # 是否为Cohere v1配置
        self.v2 = isinstance(config, Cohere2Config) # 是否为Cohere v2配置

        # Model v2 has interleaved sliding windows, v1 does not  # 模型v2有交错滑动窗口，v1没有
        if self.v2 and config.layer_types[layer_id] == "sliding_attention": # 如果是v2且当前层是滑动注意力
            self.sliding_window_size = config.sliding_window # 使用配置的滑动窗口大小
        else: # 否则
            self.sliding_window_size = -1 # 不使用滑动窗口

        self.attn = RadixAttention( # 基数注意力层
            self.num_heads, # 注意力头数
            self.head_dim, # 每个头的维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            sliding_window_size=self.sliding_window_size, # 滑动窗口大小
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数名前缀
        )
        if self.use_qk_norm: # 如果使用QK归一化
            self.q_norm = LayerNorm( # Q归一化
                param_shape=(self.num_heads, self.head_dim), eps=config.layer_norm_eps # 参数形状和epsilon
            )
            self.k_norm = LayerNorm( # K归一化
                param_shape=(self.num_kv_heads, self.head_dim), # 参数形状
                eps=config.layer_norm_eps, # epsilon
            )

    def _apply_qk_norm(self, q, k): # 对Q和K应用归一化
        q = q.view(*q.shape[:-1], -1, self.head_dim) # 重塑Q为多头格式
        k = k.view(*k.shape[:-1], -1, self.head_dim) # 重塑K为多头格式
        q, _ = self.q_norm(q) # 对Q归一化
        k, _ = self.k_norm(k) # 对K归一化
        q = q.view(*q.shape[:-2], -1) # 恢复Q的形状
        k = k.view(*k.shape[:-2], -1) # 恢复K的形状
        return q, k # 返回归一化后的Q和K

    def forward( # 前向传播：计算QKV→QK归一化→RoPE→注意力→输出投影
        self,
        positions: torch.Tensor, # 位置索引
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states) # 计算QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        if self.use_qk_norm: # 如果使用QK归一化
            q, k = self._apply_qk_norm(q, k) # 应用QK归一化
        # Model v1 uses RoPE throughout, Model v2 uses RoPE only for SWA layers  # 模型v1全程使用RoPE，模型v2仅滑动窗口注意力层使用RoPE
        if self.v1 or self.sliding_window_size > 0: # 如果是v1或使用滑动窗口
            q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch) # 执行注意力计算
        output, _ = self.o_proj(attn_output) # 通过输出投影层
        return output # 返回输出


class CohereDecoderLayer(nn.Module): # Cohere解码器层类，包含注意力和MLP
    def __init__( # 初始化Cohere解码器层
        self,
        config: PretrainedConfig, # 模型配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.hidden_size = config.hidden_size # 隐藏维度

        self.self_attn = CohereAttention( # 自注意力层
            config, # 模型配置
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("self_attn", prefix), # 参数名前缀
        )

        self.mlp = CohereMLP( # MLP层
            config, # 模型配置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("mlp", prefix), # 参数名前缀
        )
        self.input_layernorm = LayerNorm( # 输入层归一化
            param_shape=(config.hidden_size), eps=config.layer_norm_eps # 参数形状和epsilon
        )

    def forward( # 前向传播：层归一化→注意力+MLP并行→残差
        self,
        positions: torch.Tensor, # 位置索引
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批次信息
        residual: Optional[torch.Tensor], # 残差张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention  # 自注意力
        residual = hidden_states # 保存残差
        hidden_states, residual = self.input_layernorm(hidden_states, residual) # 通过层归一化
        hidden_states_attention = self.self_attn( # 通过自注意力层
            positions=positions, # 位置索引
            hidden_states=hidden_states, # 隐藏状态
            forward_batch=forward_batch, # 前向批次信息
        )
        hidden_states_mlp = self.mlp(hidden_states) # 通过MLP层
        # Add everything together  # 将所有结果相加
        hidden_states = residual + hidden_states_attention + hidden_states_mlp # 残差+注意力+MLP

        return hidden_states, residual # 返回隐藏状态和残差


class CohereModel(nn.Module): # Cohere模型类，包含词嵌入、多层解码器和最终归一化
    def __init__( # 初始化Cohere模型
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.vocab_size = config.vocab_size # 词表大小
        self.embed_tokens = VocabParallelEmbedding( # 词嵌入层
            config.vocab_size, config.hidden_size # 词表大小和隐藏维度
        )
        self.layers = nn.ModuleList( # 创建解码器层列表
            [
                CohereDecoderLayer(
                    config, # 模型配置
                    i, # 层索引
                    quant_config=quant_config, # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix), # 参数名前缀
                )
                for i in range(config.num_hidden_layers) # 遍历所有隐藏层
            ]
        )
        self.norm = LayerNorm( # 最终层归一化
            param_shape=(config.hidden_size), eps=config.layer_norm_eps # 参数形状和epsilon
        )

    def forward( # 前向传播：词嵌入→多层解码→归一化
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置索引
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) # 通过词嵌入层
        residual = None # 初始化残差为空
        for i in range(len(self.layers)): # 遍历所有层
            layer = self.layers[i] # 获取当前层
            hidden_states, residual = layer( # 通过当前解码器层
                positions, # 位置索引
                hidden_states, # 隐藏状态
                forward_batch, # 前向批次信息
                residual, # 残差
            )
        hidden_states, _ = self.norm(hidden_states, residual) # 通过最终归一化
        return hidden_states # 返回隐藏状态


class CohereForCausalLM(nn.Module): # Cohere因果语言模型类
    def __init__( # 初始化Cohere因果语言模型
        self,
        config: PretrainedConfig, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.logit_scale = getattr(config, "logit_scale", None) # 逻辑缩放因子
        self.logits_processor = LogitsProcessor(config, logit_scale=self.logit_scale) # 逻辑处理器
        self.model = CohereModel( # Cohere模型
            config, quant_config, prefix=add_prefix("model", prefix) # 配置、量化配置和前缀
        )

    @torch.no_grad() # 禁用梯度计算
    def forward( # 前向传播：通过模型和逻辑处理器
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置索引
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        hidden_states = self.model( # 通过Cohere模型
            input_ids, # 输入token ID
            positions, # 位置索引
            forward_batch, # 前向批次信息
        )
        return self.logits_processor( # 通过逻辑处理器
            input_ids, hidden_states, self.model.embed_tokens, forward_batch # 输入ID、隐藏状态、嵌入层和批次信息
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        stacked_params_mapping = [ # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"), # Q投影映射
            ("qkv_proj", "k_proj", "k"), # K投影映射
            ("qkv_proj", "v_proj", "v"), # V投影映射
            ("gate_up_proj", "gate_proj", 0), # gate投影映射
            ("gate_up_proj", "up_proj", 1), # up投影映射
        ]
        params_dict = dict(self.named_parameters()) # 获取参数字典
        loaded_params = set() # 已加载参数集合
        for name, loaded_weight in weights: # 遍历所有权重
            for param_name, shard_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射
                if shard_name not in name: # 如果分片名不在参数名中
                    continue # 跳过
                name = name.replace(shard_name, param_name) # 替换分片名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break # 跳出内层循环
            else: # 如果没有匹配到堆叠参数
                # lm_head is not used in vllm as it is tied with embed_token.  # vLLM中lm_head未使用，因为它与embed_token绑定
                # To prevent errors, skip loading lm_head.weight.  # 为防止错误，跳过加载lm_head.weight
                if "lm_head.weight" in name: # 如果是lm_head权重
                    continue # 跳过
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                # Remapping the name of FP8 kv-scale.  # 重映射FP8 KV缩放的名称
                name = maybe_remap_kv_scale_name(name, params_dict) # 可能重映射KV缩放名称
                if name is None: # 如果名称被过滤
                    continue # 跳过

                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重
            loaded_params.add(name) # 记录已加载的参数名


class Cohere2ForCausalLM(CohereForCausalLM): # Cohere2因果语言模型类，继承自CohereForCausalLM
    pass # 无额外实现


EntryClass = [CohereForCausalLM, Cohere2ForCausalLM] # 入口类列表，用于模型注册
