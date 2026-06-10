# 文件说明：Exaone模型推理实现，兼容HuggingFace权重
# 包含门控MLP、注意力层、解码器层、模型主体及因果语言模型

# Copyright 2024 The LGcns AI Engineering Team
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
# limitations under the License. # 详见License限制条件
# ==============================================================================

# Adapted from llama2.py # 改编自llama2.py
"""Inference-only Exaone model compatible with HuggingFace weights.""" # 仅推理的Exaone模型，兼容HuggingFace权重

from typing import Any, Dict, Iterable, Optional, Tuple # 导入类型提示工具

import torch # 导入PyTorch库
from torch import nn # 导入神经网络模块

from sglang.srt.distributed import get_tensor_model_parallel_world_size # 获取张量并行世界大小
from sglang.srt.layers.activation import SiluAndMul # 导入SiLU与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear, # 导入合并列并行线性层
    QKVParallelLinear, # 导入QKV并行线性层
    RowParallelLinear, # 导入行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput # 导入logits处理器和输出
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope # 导入RoPE获取函数
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead, # 导入并行语言模型头
    VocabParallelEmbedding, # 导入词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.utils import add_prefix # 导入添加前缀工具函数
from sglang.srt.utils.hf_transformers_utils import get_rope_config # 导入RoPE配置获取函数


class ExaoneGatedMLP(nn.Module): # Exaone门控MLP模块，使用SiLU激活和门控机制
    def __init__( # Exaone门控MLP初始化
        self,
        hidden_size: int, # 隐藏维度
        intermediate_size: int, # 中间维度
        hidden_act: str, # 激活函数名
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear( # gate和up合并投影
            hidden_size, # 输入维度
            [intermediate_size] * 2, # 输出维度列表（gate和up各一个中间维度）
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("gate_up_proj", prefix), # 参数前缀
        )
        self.c_proj = RowParallelLinear( # 下投影（输出投影）
            intermediate_size, # 输入维度
            hidden_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("c_proj", prefix), # 参数前缀
        )
        if hidden_act != "silu": # 检查激活函数是否为silu
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now." # 仅支持silu激活函数
            )
        self.act_fn = SiluAndMul() # 创建SiLU与乘法激活函数

    def forward(self, x): # Exaone门控MLP前向传播
        gate_up, _ = self.gate_up_proj(x) # gate和up投影
        x = self.act_fn(gate_up) # SiLU门控激活
        x, _ = self.c_proj(x) # 下投影
        return x # 返回输出


class ExaoneAttention(nn.Module): # Exaone注意力模块，支持GQA和旋转位置编码
    def __init__( # Exaone注意力初始化
        self,
        config, # 模型配置
        hidden_size: int, # 隐藏维度
        num_heads: int, # 注意力头数
        num_kv_heads: int, # KV头数
        layer_id: int = 0, # 层ID
        rope_theta: float = 500000, # RoPE基频
        rope_scaling: Optional[Dict[str, Any]] = None, # RoPE缩放配置
        rope_is_neox_style: bool = True, # RoPE是否为Neox风格
        max_position_embeddings: int = 4096, # 最大位置编码数
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
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
        self.rotary_dim = int( # 计算旋转维度
            self.head_dim * getattr(config, "partial_rotary_factor", 1)
        )
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
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("qkv_proj", prefix), # 参数前缀
        )
        self.out_proj = RowParallelLinear( # 创建输出投影
            self.total_num_heads * self.head_dim, # 输入维度
            hidden_size, # 输出维度
            bias=False, # 无偏置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("out_proj", prefix), # 参数前缀
        )

        self.rotary_emb = get_rope( # 创建旋转位置编码
            self.head_dim, # 头维度
            rotary_dim=self.rotary_dim, # 旋转维度
            max_position=max_position_embeddings, # 最大位置数
            base=rope_theta, # 基频
            rope_scaling=rope_scaling, # RoPE缩放
            is_neox_style=rope_is_neox_style, # Neox风格
        )
        self.attn = RadixAttention( # 创建基数注意力
            self.num_heads, # 注意力头数
            self.head_dim, # 头维度
            self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
        )

    def forward( # Exaone注意力前向传播
        self,
        positions: torch.Tensor, # 位置编码
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批信息
    ) -> torch.Tensor: # 返回注意力输出
        qkv, _ = self.qkv_proj(hidden_states) # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        q, k = self.rotary_emb(positions, q, k) # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch) # 计算注意力
        output, _ = self.out_proj(attn_output) # 输出投影
        return output # 返回输出


class ExaoneDecoderLayer(nn.Module): # Exaone解码器层，包含自注意力和门控MLP
    def __init__( # Exaone解码器层初始化
        self,
        config, # 模型配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.hidden_size = config.hidden_size # 保存隐藏维度
        rope_theta, rope_scaling = get_rope_config(config) # 获取RoPE配置
        if rope_scaling is not None and getattr( # 有RoPE缩放且有原始最大位置编码
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = ( # 设置原始最大位置编码
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True) # 获取RoPE是否为Neox风格
        max_position_embeddings = getattr(config, "max_position_embeddings", 4096) # 获取最大位置编码数
        self.self_attn = ExaoneAttention( # 创建自注意力层
            config=config, # 模型配置
            hidden_size=self.hidden_size, # 隐藏维度
            num_heads=config.num_attention_heads, # 注意力头数
            num_kv_heads=config.num_key_value_heads, # KV头数
            layer_id=layer_id, # 层ID
            rope_theta=rope_theta, # RoPE基频
            rope_scaling=rope_scaling, # RoPE缩放
            rope_is_neox_style=rope_is_neox_style, # RoPE Neox风格
            max_position_embeddings=max_position_embeddings, # 最大位置编码数
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("self_attn", prefix), # 参数前缀
        )
        self.mlp = ExaoneGatedMLP( # 创建门控MLP
            hidden_size=self.hidden_size, # 隐藏维度
            intermediate_size=config.intermediate_size, # 中间维度
            hidden_act=config.activation_function, # 激活函数
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("mlp", prefix), # 参数前缀
        )
        rms_norm_eps = config.layer_norm_epsilon # 获取RMS归一化epsilon
        self.ln_1 = RMSNorm(config.hidden_size, eps=rms_norm_eps) # 注意力前归一化
        self.ln_2 = RMSNorm(config.hidden_size, eps=rms_norm_eps) # MLP前归一化

    def forward( # Exaone解码器层前向传播
        self,
        positions: torch.Tensor, # 位置编码
        hidden_states: torch.Tensor, # 隐藏状态
        forward_batch: ForwardBatch, # 前向批信息
        residual: Optional[torch.Tensor], # 残差连接
    ) -> Tuple[torch.Tensor, torch.Tensor]: # 返回隐藏状态和残差
        # Self Attention # 自注意力
        if residual is None: # 无残差（第一层）
            residual = hidden_states # 保存隐藏状态作为残差
            hidden_states = self.ln_1(hidden_states) # 层归一化
        else: # 有残差
            hidden_states, residual = self.ln_1(hidden_states, residual) # 层归一化并更新残差
        hidden_states = self.self_attn( # 自注意力计算
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected # 全连接层
        hidden_states, residual = self.ln_2(hidden_states, residual) # MLP前层归一化
        hidden_states = self.mlp(hidden_states) # MLP计算
        return hidden_states, residual # 返回隐藏状态和残差


class ExaoneModel(nn.Module): # Exaone模型主体，包含嵌入层、解码器层和最终归一化
    def __init__( # Exaone模型初始化
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.padding_idx = config.pad_token_id # 保存填充token ID
        self.vocab_size = config.vocab_size # 保存词表大小
        self.wte = VocabParallelEmbedding( # 创建词表并行嵌入层
            config.vocab_size, # 词表大小
            config.hidden_size, # 隐藏维度
        )
        self.h = nn.ModuleList( # 创建解码器层列表
            [
                ExaoneDecoderLayer(
                    config, # 模型配置
                    i, # 层索引
                    quant_config=quant_config, # 量化配置
                    prefix=add_prefix(f"h.{i}", prefix), # 参数前缀
                )
                for i in range(config.num_hidden_layers) # 遍历所有隐藏层
            ]
        )
        rms_norm_eps = config.layer_norm_epsilon # 获取RMS归一化epsilon
        self.ln_f = RMSNorm(config.hidden_size, eps=rms_norm_eps) # 最终归一化层

    def forward( # Exaone模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
    ) -> torch.Tensor: # 返回隐藏状态
        if input_embeds is None: # 无输入嵌入
            hidden_states = self.wte(input_ids) # 通过嵌入层获取隐藏状态
        else: # 有输入嵌入
            hidden_states = input_embeds # 直接使用输入嵌入
        residual = None # 初始化残差为None
        for i in range(len(self.h)): # 遍历所有层
            layer = self.h[i] # 获取当前层
            hidden_states, residual = layer( # 前向传播当前层
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.ln_f(hidden_states, residual) # 最终归一化
        return hidden_states # 返回隐藏状态


class ExaoneForCausalLM(nn.Module): # Exaone因果语言模型
    def __init__( # Exaone因果语言模型初始化
        self,
        config, # 模型配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.transformer = ExaoneModel( # 创建Exaone模型主体
            config, quant_config=quant_config, prefix=add_prefix("transformer", prefix)
        )
        if self.config.tie_word_embeddings: # 如果绑定词嵌入
            self.lm_head = self.transformer.wte # 语言模型头共享嵌入层
        else: # 否则
            self.lm_head = ParallelLMHead( # 创建并行语言模型头
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏维度
                prefix=add_prefix("lm_head", prefix), # 参数前缀
            )
        self.logits_processor = LogitsProcessor(config) # 创建logits处理器

    @torch.no_grad() # 禁用梯度计算
    def forward( # Exaone因果语言模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
    ) -> LogitsProcessorOutput: # 返回logits处理器输出
        hidden_states = self.transformer( # 获取模型隐藏状态
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor( # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        stacked_params_mapping = [ # 堆叠参数映射
            # (param_name, shard_name, shard_id) # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"), # Q投影
            ("qkv_proj", "k_proj", "k"), # K投影
            ("qkv_proj", "v_proj", "v"), # V投影
            ("gate_up_proj", "c_fc_0", 0), # gate投影
            ("gate_up_proj", "c_fc_1", 1), # up投影
        ]
        params_dict = dict(self.named_parameters()) # 获取参数字典

        for name, loaded_weight in weights: # 遍历权重
            if "rotary_emb.inv_freq" in name or "projector" in name: # 跳过旋转嵌入逆频率和投影器
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name: # 跳过旋转嵌入缓存
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them. # 使用ColossalAI训练的模型可能包含这些张量，跳过
                continue
            if name.startswith("model.vision_tower") and name not in params_dict: # 跳过不在参数字典中的视觉塔
                continue

            name = name.replace("attn.attention", "self_attn") # 替换注意力名称
            for param_name, weight_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射
                if weight_name not in name: # 权重名不匹配则跳过
                    continue
                name = name.replace(weight_name, param_name) # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break
            else: # 非堆叠参数
                # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重


EntryClass = ExaoneForCausalLM # 模型入口类
