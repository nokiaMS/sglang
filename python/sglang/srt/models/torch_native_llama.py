# PyTorch原生LLaMA模型实现
# 本文件实现了使用PyTorch原生张量并行包的LLaMA推理模型。
# 支持HuggingFace权重，使用PyTorch的分布式张量并行而非自定义并行层。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""
Inference-only LLaMA model compatible with HuggingFace weights.

This model supports tensor parallelism (TP) using the PyTorch tensor parallel package.
Reference: https://pytorch.org/docs/stable/distributed.tensor.parallel.html

Here is a quick example to enable TP:
```python
from sglang.srt.layers.model_parallel import tensor_parallel

device_mesh = torch.distributed.init_device_mesh("cuda", (tp_size,))
tensor_parallel(model, device_mesh)
```

An end-to-end example can be found in `python/sglang/bench_one_batch.py`.
You can run it with the following command:
```bash
$ python3 -m sglang.bench_one_batch --correct \
  --model meta-llama/Meta-Llama-3-8B \
  --json-model-override-args '{"architectures": ["TorchNativeLlamaForCausalLM"]}' \
  --tensor-parallel-size 2 \
  --disable-cuda-graph
```
We will enable CUDA Graph support soon.
"""

import types  # 导入types模块，用于动态方法绑定
from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch核心库
from torch import nn  # 导入神经网络模块
from torch.nn.parameter import Parameter  # 导入参数类
from transformers import LlamaConfig  # 导入LLaMA配置

from sglang.srt.distributed import (  # 导入分布式通信函数
    get_tensor_model_parallel_rank,  # 获取张量并行秩
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU和乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

tp_size: Optional[int] = None  # 全局张量并行大小
tp_rank: Optional[int] = None  # 全局张量并行秩


def gate_up_proj_weight_loader(
    self,
    param: Parameter,  # 目标参数
    loaded_weight: torch.Tensor,  # 加载的权重
    loaded_shard_id: int,  # 加载的分片ID
):
    """门控和上投影的权重加载器，支持张量并行"""
    # shard_id: (shard_offset, shard_size)
    gate_up_offsets = {}  # 分片偏移字典
    current_shard_offset = 0  # 当前分片偏移
    for i, output_size in enumerate(self.output_sizes):  # 遍历输出大小
        # Everything shrinks by tp_size if TP enabled
        output_size = output_size // tp_size  # TP缩放输出大小
        gate_up_offsets[i] = (current_shard_offset, output_size)  # 记录偏移和大小
        current_shard_offset += output_size  # 更新偏移
    # Re-size the param to the size after TP
    if current_shard_offset != param.shape[0]:  # 如果大小不匹配
        # The clone will free the original, full tensor
        param.data = param.data.narrow(0, 0, current_shard_offset).clone()  # 裁剪并克隆

    # Now load gate or up
    assert loaded_shard_id < len(self.output_sizes)  # 断言分片ID有效
    param_data = param.data  # 获取参数数据
    shard_offset, shard_size = gate_up_offsets[loaded_shard_id]  # 获取分片偏移和大小
    param_data = param_data.narrow(0, shard_offset, shard_size)  # 裁剪参数数据
    loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)  # 裁剪加载的权重
    assert param_data.shape == loaded_weight.shape  # 断言形状匹配
    param_data.copy_(loaded_weight)  # 复制权重


class LlamaMLP(nn.Module):
    """LLaMA模型的MLP模块"""
    _tp_plan = {  # 张量并行计划
        "gate_up_proj": "Colwise_Sharded",  # 门控上投影列分片
        "down_proj": "Rowwise",  # 下投影行分片
    }

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = torch.nn.Linear(  # 门控上投影（PyTorch原生线性层）
            hidden_size,  # 输入大小
            intermediate_size * 2,  # 输出大小（gate和up合并）
            bias=False,  # 不使用偏置
        )
        self.gate_up_proj.output_sizes = [intermediate_size] * 2  # 设置输出大小列表
        self.gate_up_proj.weight_loader = types.MethodType(  # 绑定权重加载器方法
            gate_up_proj_weight_loader, self.gate_up_proj
        )
        self.gate_up_proj.weight.weight_loader = self.gate_up_proj.weight_loader  # 设置权重的加载器
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)  # 下投影层
        if hidden_act != "silu":  # 如果不是SiLU激活
            raise ValueError(  # 抛出异常
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU和乘法激活函数

    def forward(self, x):
        """MLP前向传播：gate_up -> SiLU激活 -> down"""
        gate_up = self.gate_up_proj(x)  # 通过门控上投影
        x = self.act_fn(gate_up)  # 应用SiLU和乘法
        x = self.down_proj(x)  # 通过下投影
        return x  # 返回输出


def qkv_proj_weight_loader(
    self,
    param: Parameter,  # 目标参数
    loaded_weight: torch.Tensor,  # 加载的权重
    loaded_shard_id: str,  # 加载的分片ID（q/k/v）
):
    """QKV投影的权重加载器，支持张量并行"""
    num_heads = self.num_heads // tp_size  # TP后的头数
    num_kv_heads = self.num_kv_heads // tp_size  # TP后的KV头数
    # shard_id: (shard_offset, shard_size)
    qkv_offsets = {  # QKV偏移字典
        "q": (0, num_heads * self.head_size),  # Q的偏移和大小
        "k": (num_heads * self.head_size, num_kv_heads * self.head_size),  # K的偏移和大小
        "v": (  # V的偏移和大小
            (num_heads + num_kv_heads) * self.head_size,
            num_kv_heads * self.head_size,
        ),
    }
    total_size = qkv_offsets["v"][0] + qkv_offsets["v"][1]  # 总大小
    # Re-size the param to the size after TP
    if total_size != param.shape[0]:  # 如果大小不匹配
        # The clone will free the original, full tensor
        param.data = param.data.narrow(0, 0, total_size).clone()  # 裁剪并克隆

    # Now load q, k or v
    shard_offset, shard_size = qkv_offsets[loaded_shard_id]  # 获取分片偏移和大小
    param_data = param.data  # 获取参数数据
    param_data = param_data.narrow(0, shard_offset, shard_size)  # 裁剪参数数据
    loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)  # 裁剪加载的权重
    assert param_data.shape == loaded_weight.shape  # 断言形状匹配
    param_data.copy_(loaded_weight)  # 复制权重


class LlamaAttention(nn.Module):
    """LLaMA注意力模块，使用PyTorch原生线性层"""
    _tp_plan = {  # 张量并行计划
        "qkv_proj": "Colwise_Sharded",  # QKV投影列分片
        "o_proj": "Rowwise",  # 输出投影行分片
    }

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID
        rope_theta: float = 10000,  # RoPE theta
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放
        rope_is_neox_style: bool = True,  # 是否使用Neox风格RoPE
        max_position_embeddings: int = 8192,  # 最大位置编码
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.total_num_heads = num_heads  # 保存总头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # TP后的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0  # 断言可整除
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0  # 断言可整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # TP后的KV头数
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(  # 获取头维度
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim  # Q大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 保存RoPE theta
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置编码

        self.qkv_proj = torch.nn.Linear(  # QKV投影（PyTorch原生）
            hidden_size,  # 输入大小
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,  # 输出大小
            bias=False,  # 不使用偏置
        )
        self.qkv_proj.head_size = self.head_dim  # 设置头大小
        self.qkv_proj.num_heads = self.total_num_heads  # 设置总头数
        self.qkv_proj.num_kv_heads = self.total_num_kv_heads  # 设置总KV头数
        self.qkv_proj.weight_loader = types.MethodType(  # 绑定权重加载器
            qkv_proj_weight_loader, self.qkv_proj
        )
        self.qkv_proj.weight.weight_loader = self.qkv_proj.weight_loader  # 设置权重的加载器
        self.qkv_proj.weight.output_dim = 0  # 设置输出维度
        self.o_proj = torch.nn.Linear(  # 输出投影
            self.total_num_heads * self.head_dim,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
        )
        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基数
            rope_scaling=rope_scaling,  # 缩放
            is_neox_style=rope_is_neox_style,  # Neox风格
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """注意力前向传播：QKV投影 -> 分离 -> RoPE -> 注意力 -> 输出投影"""
        qkv = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分离Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力层
        output = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class LlamaDecoderLayer(nn.Module):
    """LLaMA解码器层，包含自注意力和MLP"""

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        rope_theta = config.rope_parameters["rope_theta"]  # 获取RoPE theta
        rope_scaling = config.rope_parameters  # 获取RoPE缩放参数
        if rope_scaling is not None and getattr(  # 如果有缩放且有原始最大位置
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 设置原始最大位置
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)  # 获取RoPE风格
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 获取最大位置编码
        self.self_attn = LlamaAttention(  # 自注意力层
            config=config,  # 配置
            hidden_size=self.hidden_size,  # 隐藏大小
            num_heads=config.num_attention_heads,  # 头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放
            rope_is_neox_style=rope_is_neox_style,  # RoPE风格
            max_position_embeddings=max_position_embeddings,  # 最大位置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )
        self.mlp = LlamaMLP(  # MLP层
            hidden_size=self.hidden_size,  # 隐藏大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播：自注意力 + MLP"""
        # Self Attention
        if residual is None:  # 如果没有残差
            residual = hidden_states  # 设置残差为隐藏状态
            hidden_states = self.input_layernorm(hidden_states)  # 归一化
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 带残差归一化
        hidden_states = self.self_attn(  # 通过自注意力
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP
        return hidden_states, residual  # 返回隐藏状态和残差


class LlamaModel(nn.Module):
    """LLaMA模型主体，包含嵌入层、解码器层和归一化"""

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ) -> None:
        super().__init__()  # 调用父类初始化

        global tp_size, tp_rank  # 声明全局变量
        if tp_size is None:  # 如果未初始化
            tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        if tp_rank is None:  # 如果未初始化
            tp_rank = get_tensor_model_parallel_rank()  # 获取TP秩

        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏大小
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                LlamaDecoderLayer(
                    config, i, quant_config=quant_config, prefix=f"model.layers.{i}"  # 每层配置
                )
                for i in range(config.num_hidden_layers)  # 遍历层数
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> torch.Tensor:
        """模型主体前向传播：嵌入 -> 解码器层 -> 归一化"""
        if input_embeds is None:  # 如果没有输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层
        else:
            hidden_states = input_embeds  # 使用输入嵌入
        residual = None  # 初始化残差
        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                positions,  # 位置
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
                residual,  # 残差
            )
        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class TorchNativeLlamaForCausalLM(nn.Module):
    """使用PyTorch原生TP的LLaMA因果语言模型"""

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.supports_torch_tp = True  # 支持PyTorch原生TP
        self.model = LlamaModel(config, quant_config=quant_config)  # 创建模型主体
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens  # 使用嵌入层作为LM头
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)  # 并行LM头
        self.logits_processor = LogitsProcessor(config)  # logits处理器

        # turning off autotune for fp8dq since it doesn't give speedup and
        # increases compile time significantly
        torch._inductor.config.max_autotune_gemm_backends = "ATEN"  # 禁用FP8自调优

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> LogitsProcessorOutput:
        """因果语言模型前向传播：模型主体 -> logits处理"""
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型
        return self.logits_processor(  # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_module_name_from_weight_name(self, name):
        """从权重名称获取模块名称"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id, num_shard)
            ("qkv_proj", "q_proj", "q", 3),  # Q投影
            ("qkv_proj", "k_proj", "k", 3),  # K投影
            ("qkv_proj", "v_proj", "v", 3),  # V投影
            ("gate_up_proj", "gate_proj", 0, 2),  # 门控投影
            ("gate_up_proj", "up_proj", 1, 2),  # 上投影
        ]
        for param_name, weight_name, shard_id, num_shard in stacked_params_mapping:  # 遍历映射
            if weight_name in name:  # 如果包含权重名
                return (  # 返回模块名和分片数
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1  # 返回原始模块名和分片数1

    def get_num_params(self):
        """获取模型参数数量"""
        params_dict = dict(self.named_parameters())  # 参数字典
        return len(params_dict)  # 返回参数数量

    def load_weights_to_module(
        self,
        fqn: str,  # 模块完全限定名
        weights: Iterable[Tuple[str, torch.Tensor]],  # 权重迭代器
    ):
        """Load weights onto submodule pointed by path `fqn`."""  # 加载权重到指定子模块
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),  # Q投影
            (".qkv_proj", ".k_proj", "k"),  # K投影
            (".qkv_proj", ".v_proj", "v"),  # V投影
            (".gate_up_proj", ".gate_proj", 0),  # 门控投影
            (".gate_up_proj", ".up_proj", 1),  # 上投影
        ]
        module = self.get_submodule(fqn)  # 获取子模块
        params_dict = dict(module.named_parameters(prefix=fqn, recurse=False))  # 参数字典

        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转频率和投影器
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过视觉塔
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的LM头
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射
                if weight_name not in name:  # 不包含权重名
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") or name not in params_dict:  # 跳过GPTQ偏置
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内循环
            else:  # 没有匹配到堆叠参数
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") or name not in params_dict:  # 跳过GPTQ偏置
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取加载器
                weight_loader(param, loaded_weight)  # 加载权重

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],  # 权重迭代器
    ):
        """Load weights onto the full model."""  # 加载权重到完整模型
        self.load_weights_to_module("", weights)  # 加载到根模块


class TorchNativePhi3ForCausalLM(TorchNativeLlamaForCausalLM):
    """Phi3模型，继承自TorchNativeLlamaForCausalLM"""
    pass  # 无额外实现


EntryClass = [TorchNativeLlamaForCausalLM, TorchNativePhi3ForCausalLM]  # 入口类列表
