# Nemotron NAS (DeciLM) 因果语言模型实现
# 该文件实现了推理专用的 DeciLM/Nemotron-NAS 模型，兼容 HuggingFace 权重格式，
# 支持可变注意力头分组和 FFN 倍数，以及流水线并行。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# limitations under the License.
# ==============================================================================
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_nas.py

"""Inference-only deci model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple, Type, Union  # 导入类型提示 # 导入类型提示

import torch  # 导入 PyTorch # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块 # 导入神经网络模块
from transformers import LlamaConfig  # 导入 Llama 配置 # 导入 Llama 配置类

from sglang.srt.distributed import get_pp_group  # 导入流水线并行组 # 导入流水线并行组
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化 # 导入 RMS 归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入 logits 处理器 # 导入 logits 处理器和输出
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层 # 导入池化层和类型
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置 # 导入量化配置
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线缺失层 # 导入流水线缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入 # 导入词表并行嵌入层
    DEFAULT_VOCAB_PADDING_SIZE,  # 默认词表填充大小 # 默认词表填充大小
    ParallelLMHead,  # 并行语言模型头 # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入 # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息 # 导入前向批次信息和流水线代理张量
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具 # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器 # 默认权重加载器
    maybe_remap_kv_scale_name,  # KV 缩放名称重映射 # KV 缩放名称重映射
)
from sglang.srt.models.llama import LlamaAttention, LlamaMLP  # 导入 Llama 组件 # 导入 Llama 注意力和 MLP
from sglang.srt.utils import add_prefix, make_layers  # 导入工具函数 # 导入工具函数
from sglang.utils import logger  # 导入日志器 # 导入日志器


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:  # FFN 倍数转中间层大小 # 将 FFN 倍数转换为中间层大小
    """将 FFN 倍数转换为中间层大小（DeciLM 特有）"""
    # DeciLM-specific code # DeciLM 特有代码
    intermediate_size = int(2 * ffn_mult * n_embd / 3)  # 计算中间层大小 # 计算中间层大小
    return _find_multiple(intermediate_size, 256)  # 对齐到 256 的倍数 # 对齐到 256 的倍数


def _find_multiple(n: int, k: int) -> int:  # 找到最近的倍数 # 找到大于等于 n 的最近的 k 的倍数
    """找到大于等于 n 的最近的 k 的倍数"""
    # DeciLM-specific code # DeciLM 特有代码
    if n % k == 0:  # 如果已经是倍数 # 如果 n 已经是 k 的倍数
        return n  # 返回 n # 返回 n
    return n + k - (n % k)  # 返回下一个倍数 # 返回下一个 k 的倍数


class DeciLMDecoderLayer(nn.Module):  # DeciLM 解码器层 # DeciLM 解码器层
    """DeciLM 解码器层，支持可变注意力头分组和可选的 no-op 层"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: LlamaConfig,  # 模型配置 # Llama 配置
        layer_idx: int,  # 层索引 # 层索引
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        block_config = config.block_configs[layer_idx]  # 获取当前层的块配置 # 获取当前层的块配置
        self._is_no_op_attention = block_config.attention.no_op  # 是否为 no-op 注意力 # 是否为无操作注意力
        self._is_no_op_ffn = block_config.ffn.no_op  # 是否为 no-op FFN # 是否为无操作前馈

        self.hidden_size = config.hidden_size  # 隐藏层大小 # 隐藏层维度
        rope_theta = config.rope_parameters["rope_theta"]  # RoPE theta # 旋转位置编码的 theta 参数
        rope_scaling = config.rope_parameters  # RoPE 缩放 # 旋转位置编码缩放配置
        if rope_scaling is not None and getattr(  # 如果有缩放且有原始最大位置 # 如果有缩放配置且有原始最大位置
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 设置原始最大位置 # 设置原始最大位置
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置编码数 # 最大位置编码数
        # Support abacusai/Smaug-72B-v0.1 with attention_bias # 支持 attention_bias
        # Support internlm/internlm-7b with bias # 支持 bias
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)  # RoPE 风格 # RoPE 是否为 Neox 风格
        attention_bias = getattr(config, "attention_bias", False) or getattr(  # 注意力偏置 # 注意力偏置
            config, "bias", False
        )
        # support internlm/internlm3-8b with qkv_bias # 支持 qkv_bias
        if hasattr(config, "qkv_bias"):  # 如果有 qkv_bias # 如果有 qkv_bias 配置
            attention_bias = config.qkv_bias  # 使用 qkv_bias # 使用 qkv_bias 作为偏置

        if not self._is_no_op_attention:  # 如果不是 no-op 注意力 # 如果不是无操作注意力
            num_kv_heads = (  # KV 头数 # 计算 KV 头数
                config.num_attention_heads // block_config.attention.n_heads_in_group  # 总头数除以组内头数 # 总头数除以每组内的头数
            )
            self.self_attn = LlamaAttention(  # 自注意力 # 创建 Llama 注意力层
                config=config,  # 配置 # 模型配置
                hidden_size=self.hidden_size,  # 隐藏层大小 # 隐藏层维度
                num_heads=config.num_attention_heads,  # 注意力头数 # 注意力头数
                num_kv_heads=num_kv_heads,  # KV 头数 # KV 头数
                layer_id=layer_idx,  # 层 ID # 层 ID
                rope_theta=rope_theta,  # RoPE theta # 旋转位置编码 theta
                rope_scaling=rope_scaling,  # RoPE 缩放 # 旋转位置编码缩放
                rope_is_neox_style=rope_is_neox_style,  # RoPE 风格 # RoPE 风格
                max_position_embeddings=max_position_embeddings,  # 最大位置编码数 # 最大位置编码数
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=add_prefix("self_attn", prefix),  # 前缀 # 参数名前缀
                bias=attention_bias,  # 偏置 # 偏置
            )
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化 # 输入层 RMS 归一化

        if not self._is_no_op_ffn:  # 如果不是 no-op FFN # 如果不是无操作前馈
            ffn_mult = block_config.ffn.ffn_mult  # FFN 倍数 # FFN 倍数
            intermediate_size = _ffn_mult_to_intermediate_size(  # 计算中间层大小 # 计算中间层大小
                ffn_mult, config.hidden_size  # FFN 倍数和隐藏大小 # FFN 倍数和隐藏层维度
            )
            self.mlp = LlamaMLP(  # MLP # 创建 Llama MLP 层
                hidden_size=self.hidden_size,  # 隐藏层大小 # 隐藏层维度
                intermediate_size=intermediate_size,  # 中间层大小 # 中间层维度
                hidden_act=config.hidden_act,  # 隐藏激活函数 # 隐藏层激活函数
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=add_prefix("mlp", prefix),  # 前缀 # 参数名前缀
            )
            self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化 # 注意力后 RMS 归一化
                config.hidden_size, eps=config.rms_norm_eps  # 大小和 epsilon # 隐藏维度和 epsilon
            )

    def forward(  # 前向传播 # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码 # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差 # 残差张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回隐藏状态和残差 # 返回隐藏状态和残差
        # Self Attention # 自注意力

        if self._is_no_op_attention:  # 如果是 no-op 注意力 # 如果是无操作注意力
            pass  # 跳过 # 跳过
        else:  # 否则 # 正常注意力
            if residual is None:  # 如果没有残差 # 如果没有残差
                residual = hidden_states  # 保存残差 # 保存残差连接
                hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化 # 应用输入层归一化
            else:  # 否则 # 有残差
                hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 归一化并更新残差 # 归一化并更新残差
            hidden_states = self.self_attn(  # 自注意力前向 # 计算自注意力
                positions=positions,  # 位置编码 # 位置编码
                hidden_states=hidden_states,  # 隐藏状态 # 隐藏状态
                forward_batch=forward_batch,  # 前向批次 # 前向批次信息
            )

        # Fully Connected # 全连接层
        if not self._is_no_op_ffn:  # 如果不是 no-op FFN # 如果不是无操作前馈
            hidden_states, residual = self.post_attention_layernorm(  # 注意力后归一化 # 应用注意力后归一化
                hidden_states, residual  # 隐藏状态和残差 # 隐藏状态和残差
            )
            hidden_states = self.mlp(hidden_states)  # MLP 前向 # 计算 MLP
        return hidden_states, residual  # 返回隐藏状态和残差 # 返回隐藏状态和残差


class DeciModel(nn.Module):  # DeciLM 模型 # DeciLM 模型主体
    """DeciLM 模型主体，支持流水线并行"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        *,
        config: LlamaConfig,  # 模型配置 # Llama 配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
        layer_type: Type[DeciLMDecoderLayer] = DeciLMDecoderLayer,  # 层类型 # 解码器层类型
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化

        lora_config = None  # LoRA 配置 # LoRA 配置（暂未使用）
        self.config = config  # 保存配置 # 保存模型配置
        self.quant_config = quant_config  # 保存量化配置 # 保存量化配置
        self.padding_idx = config.pad_token_id  # 填充索引 # 填充标记 ID
        lora_vocab = (  # LoRA 额外词表大小 # LoRA 额外词表大小
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))  # 计算额外词表 # 计算额外词表大小
            if lora_config  # 如果有 LoRA # 如果有 LoRA 配置
            else 0  # 否则为 0 # 否则为 0
        )
        vocab_size = config.vocab_size + lora_vocab  # 总词表大小 # 总词表大小
        if get_pp_group().is_first_rank:  # 如果是第一个并行秩 # 如果是流水线并行的第一个秩
            self.embed_tokens = VocabParallelEmbedding(  # 词嵌入 # 创建词表并行嵌入层
                vocab_size,  # 词表大小 # 词表大小
                config.hidden_size,  # 隐藏层大小 # 隐藏层维度
                org_num_embeddings=config.vocab_size,  # 原始嵌入数 # 原始嵌入数
                quant_config=quant_config,  # 量化配置 # 量化配置
            )
        else:  # 否则 # 非第一个秩
            self.embed_tokens = PPMissingLayer()  # 缺失层 # 创建缺失层占位

        def get_layer(idx: int, prefix: str):  # 获取层 # 获取指定索引的解码器层
            return layer_type(  # 创建层 # 创建解码器层
                config,  # 配置 # 模型配置
                layer_idx=idx,  # 层索引 # 层索引
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=prefix,  # 前缀 # 参数名前缀
            )

        self.layers, self.start_layer, self.end_layer = make_layers(  # 构建解码器层 # 构建解码器层列表
            config.num_hidden_layers,  # 隐藏层数量 # 隐藏层数量
            get_layer,  # 层构建函数 # 层构建函数
            pp_rank=get_pp_group().rank_in_group,  # 流水线并行秩 # 流水线并行秩
            pp_size=get_pp_group().world_size,  # 流水线并行大小 # 流水线并行世界大小
            prefix=add_prefix("layers", prefix),  # 前缀 # 参数名前缀
        )
        if get_pp_group().is_last_rank:  # 如果是最后一个并行秩 # 如果是流水线并行的最后一个秩
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化 # 最终 RMS 归一化
        else:  # 否则 # 非最后一个秩
            self.norm = PPMissingLayer(return_tuple=True)  # 缺失层 # 创建缺失层占位

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入 # 获取输入嵌入
        """通过嵌入层获取输入 ID 的嵌入表示"""
        return self.embed_tokens(input_ids)  # 返回嵌入 # 返回嵌入

    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: Optional[torch.Tensor],  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入 # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量 # 流水线代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:  # 返回隐藏状态或代理张量 # 返回隐藏状态或代理张量
        if get_pp_group().is_first_rank:  # 如果是第一个并行秩 # 如果是流水线并行的第一个秩
            if inputs_embeds is not None:  # 如果提供了嵌入 # 如果提供了嵌入
                hidden_states = inputs_embeds  # 使用嵌入 # 使用嵌入
            else:  # 否则 # 没有提供嵌入
                hidden_states = self.get_input_embeddings(input_ids)  # 获取嵌入 # 通过嵌入层获取嵌入
            residual = None  # 初始化残差 # 初始化残差
        else:  # 否则 # 非第一个秩
            assert pp_proxy_tensors is not None  # 断言代理张量不为空 # 断言代理张量不为空
            hidden_states = pp_proxy_tensors["hidden_states"]  # 获取隐藏状态 # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 获取残差 # 从代理张量获取残差

        kv_cache_index = 0  # KV 缓存索引 # KV 缓存索引
        for i in range(self.start_layer, self.end_layer):  # 遍历层 # 遍历所有层
            layer = self.layers[i]  # 获取层 # 获取当前层
            if not layer._is_no_op_attention:  # 如果不是 no-op 注意力 # 如果不是无操作注意力
                hidden_states, residual = layer(  # 解码器层前向 # 解码器层前向传播
                    positions, hidden_states, forward_batch, residual  # 位置、隐藏状态、批次、残差 # 传入位置、隐藏状态、批次和残差
                )
                kv_cache_index += 1  # 增加 KV 缓存索引 # 增加 KV 缓存索引
            else:  # 否则 # no-op 注意力
                hidden_states, residual = layer(  # 解码器层前向（无 KV 缓存） # 解码器层前向传播
                    positions, hidden_states, forward_batch, residual  # 位置、隐藏状态、批次、残差 # 传入位置、隐藏状态、批次和残差
                )

        if not get_pp_group().is_last_rank:  # 如果不是最后一个并行秩 # 如果不是流水线并行的最后一个秩
            return PPProxyTensors(  # 返回代理张量 # 返回代理张量
                {"hidden_states": hidden_states, "residual": residual}  # 隐藏状态和残差 # 隐藏状态和残差
            )

        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化 # 应用最终归一化
        return hidden_states  # 返回隐藏状态 # 返回隐藏状态


class DeciLMForCausalLM(nn.Module):  # DeciLM 因果语言模型 # DeciLM 因果语言模型
    """DeciLM 因果语言模型，支持 LoRA 和 Mistral 格式检查点"""

    packed_modules_mapping = {  # 打包模块映射 # 打包模块映射
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],  # QKV 投影 # QKV 投影映射
        "gate_up_proj": ["gate_proj", "up_proj"],  # 门控上投影 # 门控上投影映射
    }

    # LoRA specific attributes # LoRA 特定属性
    supported_lora_modules = [  # 支持的 LoRA 模块 # 支持的 LoRA 模块
        "qkv_proj",  # QKV 投影 # QKV 投影
        "o_proj",  # 输出投影 # 输出投影
        "gate_up_proj",  # 门控上投影 # 门控上投影
        "down_proj",  # 下投影 # 下投影
        "embed_tokens",  # 嵌入层 # 嵌入层
        "lm_head",  # 语言模型头 # 语言模型头
    ]
    embedding_modules = {  # 嵌入模块映射 # 嵌入模块映射
        "embed_tokens": "input_embeddings",  # 输入嵌入 # 输入嵌入
        "lm_head": "output_embeddings",  # 输出嵌入 # 输出嵌入
    }
    embedding_padding_modules = ["lm_head"]  # 嵌入填充模块 # 嵌入填充模块

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints # Mistral 格式映射
    mistral_mapping = {  # Mistral 映射 # Mistral 格式检查点名称映射
        "layers": "model.layers",  # 层 # 层映射
        "attention": "self_attn",  # 注意力 # 注意力映射
        "wq": "q_proj",  # Q 投影 # Q 投影映射
        "wk": "k_proj",  # K 投影 # K 投影映射
        "wv": "v_proj",  # V 投影 # V 投影映射
        "wo": "o_proj",  # 输出投影 # 输出投影映射
        "attention_norm": "input_layernorm",  # 注意力归一化 # 注意力归一化映射
        "feed_forward": "mlp",  # 前馈 # 前馈映射
        "w1": "gate_proj",  # 门控投影 # 门控投影映射
        "w2": "down_proj",  # 下投影 # 下投影映射
        "w3": "up_proj",  # 上投影 # 上投影映射
        "ffn_norm": "post_attention_layernorm",  # 前馈归一化 # 前馈归一化映射
        "tok_embeddings": "model.embed_tokens",  # 词嵌入 # 词嵌入映射
        "output": "lm_head",  # 输出 # 输出映射
        "norm": "model.norm",  # 归一化 # 归一化映射
    }

    def __init__(  # 初始化方法 # 初始化方法
        self,
        *,
        config: LlamaConfig,  # 模型配置 # Llama 配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        lora_config = None  # LoRA 配置 # LoRA 配置（暂未使用）
        self.config = config  # 保存配置 # 保存模型配置
        self.lora_config = lora_config  # 保存 LoRA 配置 # 保存 LoRA 配置

        self.model = self._init_model(  # 初始化模型 # 初始化模型主体
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 配置和量化 # 传入配置和量化配置
        )
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入 # 如果绑定词嵌入权重
            self.lm_head = self.model.embed_tokens  # 语言模型头共享嵌入 # 语言模型头共享嵌入权重
        else:  # 否则 # 不绑定
            self.unpadded_vocab_size = config.vocab_size  # 未填充的词表大小 # 未填充的词表大小
            if lora_config:  # 如果有 LoRA # 如果有 LoRA 配置
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size  # 增加额外词表 # 增加额外词表大小
            self.lm_head = ParallelLMHead(  # 并行语言模型头 # 创建并行语言模型头
                self.unpadded_vocab_size,  # 词表大小 # 词表大小
                config.hidden_size,  # 隐藏层大小 # 隐藏层维度
                org_num_embeddings=config.vocab_size,  # 原始嵌入数 # 原始嵌入数
                padding_size=(  # 填充大小 # 填充大小
                    DEFAULT_VOCAB_PADDING_SIZE  # 默认填充 # 默认填充大小
                    # We need bigger padding if using lora for kernel
                    # compatibility # LoRA 需要更大的填充
                    if not lora_config  # 如果没有 LoRA # 如果没有 LoRA
                    else lora_config.lora_vocab_padding_size  # LoRA 填充 # LoRA 词表填充大小
                ),
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 前缀 # 参数名前缀
            )
        self.logits_processor = LogitsProcessor(config)  # logits 处理器 # 创建 logits 处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 池化层 # 创建池化层

    def _init_model(  # 初始化模型 # 初始化模型主体
        self,
        config: LlamaConfig,  # 模型配置 # Llama 配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        """初始化 DeciModel 实例"""
        return DeciModel(config=config, quant_config=quant_config, prefix=prefix)  # 返回 DeciModel # 返回 DeciModel 实例

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入 # 获取输入嵌入
        """获取语言模型的输入嵌入"""
        return self.model.get_input_embeddings(input_ids)  # 委托给模型 # 委托给模型

    @torch.no_grad()  # 禁用梯度 # 禁用梯度计算
    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入 # 输入嵌入（可选）
        get_embedding: bool = False,  # 是否获取嵌入 # 是否获取嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量 # 流水线代理张量
    ) -> LogitsProcessorOutput:  # 返回 logits 处理器输出 # 返回 logits 处理器输出
        hidden_states = self.model(  # 模型前向 # 模型前向传播
            input_ids,  # 输入 ID # 输入标记 ID
            positions,  # 位置编码 # 位置编码
            forward_batch,  # 前向批次 # 前向批次信息
            inputs_embeds,  # 输入嵌入 # 输入嵌入
            pp_proxy_tensors=pp_proxy_tensors,  # 流水线代理张量 # 流水线代理张量
        )
        if get_pp_group().is_last_rank:  # 如果是最后一个并行秩 # 如果是流水线并行的最后一个秩
            if not get_embedding:  # 如果不获取嵌入 # 如果不获取嵌入
                return self.logits_processor(  # logits 处理 # 通过 logits 处理器计算 logits
                    input_ids, hidden_states, self.lm_head, forward_batch  # 输入参数 # 输入参数
                )
            else:  # 否则 # 获取嵌入
                return self.pooler(hidden_states, forward_batch)  # 池化 # 通过池化层获取嵌入
        else:  # 否则 # 非最后一个秩
            return hidden_states  # 返回隐藏状态 # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:  # 加载权重 # 加载模型权重
        """加载模型权重，处理堆叠参数和专家参数"""
        stacked_params_mapping = [  # 堆叠参数映射 # 堆叠参数映射
            # (param_name, shard_name, shard_id) # (参数名, 分片名, 分片 ID)
            (".qkv_proj", ".q_proj", "q"),  # Q 投影 # Q 投影映射
            (".qkv_proj", ".k_proj", "k"),  # K 投影 # K 投影映射
            (".qkv_proj", ".v_proj", "v"),  # V 投影 # V 投影映射
            (".gate_up_proj", ".gate_proj", 0),  # 门控投影 # 门控投影映射
            (".gate_up_proj", ".up_proj", 1),  # 上投影 # 上投影映射
        ]

        params_dict = dict(self.named_parameters())  # 参数字典 # 获取模型参数字典

        for name, loaded_weight in weights:  # 遍历权重 # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码的逆频率 # 跳过旋转位置编码的逆频率
                continue  # 继续 # 跳过
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的余弦/正弦 # 跳过缓存的余弦/正弦值
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue  # 继续 # 跳过
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的 lm_head 权重 # 跳过绑定的语言模型头权重
                continue  # 继续 # 跳过
            if self.model.quant_config is not None and (  # 如果有量化配置且 # 如果有量化配置且
                scale_name := self.model.quant_config.get_cache_scale(name)  # 获取缓存缩放名称 # 获取缓存缩放名称
            ):
                # Loading kv cache quantization scales # 加载 KV 缓存量化缩放
                param = params_dict[scale_name]  # 获取参数 # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器 # 获取权重加载器
                loaded_weight = (  # 处理权重 # 处理权重
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]  # 0D 保持，1D 取第一个 # 0D 保持，1D 取第一个
                )
                weight_loader(param, loaded_weight)  # 加载权重 # 加载权重
                continue  # 继续 # 跳过
            if "scale" in name:  # 如果是缩放参数 # 如果是缩放参数
                name = maybe_remap_kv_scale_name(name, params_dict)  # 重映射名称 # 重映射 KV 缩放名称
                if name is None:  # 如果名称为空 # 如果名称为空
                    continue  # 继续 # 跳过

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射 # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中 # 如果权重名不在参数名中
                    continue  # 继续 # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名 # 替换权重名
                # Skip loading extra bias for GPTQ models. # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 跳过不存在的偏置
                    continue  # 继续 # 跳过
                if name not in params_dict:  # 如果参数不存在 # 如果参数不存在
                    continue  # 继续 # 跳过
                param = params_dict[name]  # 获取参数 # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器 # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重 # 加载权重分片
                break  # 跳出内层循环 # 跳出内层循环
            else:  # 非堆叠参数 # 非堆叠参数
                # Skip loading extra bias for GPTQ models. # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 跳过不存在的偏置
                    continue  # 继续 # 跳过
                if name in params_dict.keys():  # 如果参数存在 # 如果参数存在
                    param = params_dict[name]  # 获取参数 # 获取参数
                    weight_loader = getattr(  # 获取权重加载器 # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认使用标准加载器 # 默认使用标准权重加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重 # 加载权重
                else:  # 否则 # 参数不存在
                    logger.warning(f"Parameter {name} not found in params_dict")  # 记录警告 # 记录警告


EntryClass = [DeciLMForCausalLM]  # 入口类 # 模型入口类列表
