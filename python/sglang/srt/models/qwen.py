# QWen 语言模型实现
# 本文件实现了 QWen（通义千问）基础语言模型，包含 MLP、注意力、Transformer 块、
# 模型主体和语言模型头部等组件。QWen 采用 GPT 风格的 Transformer 解码器架构，
# 使用旋转位置编码（RoPE）和 SiLU 激活函数。
# SPDX-License-Identifier: Apache-2.0  # 许可证声明
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
# Copyright 2023-2024 SGLang Team  # SGLang 团队版权
# Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 不得违反许可证使用
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何担保
# See the License for the specific language governing permissions and  # 查看许可证获取权限
# limitations under the License.  # 许可证限制
# ==============================================================================

# Adapted from  # 适配自
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/qwen.py#L1  # vLLM 项目中的 QWen 实现

from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入张量并行世界大小获取函数
from sglang.srt.layers.activation import SiluAndMul  # 导入 SiLU 与乘法融合激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化层
from sglang.srt.layers.linear import (  # 导入线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV 并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入 Radix 注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入 RoPE 配置获取工具


class QWenMLP(nn.Module):
    """QWen 模型的 MLP（多层感知机）模块，使用门控 SiLU 激活"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化 QWen MLP 模块"""
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并线性层
            hidden_size,  # 输入隐藏维度
            2 * [intermediate_size],  # 输出为两倍中间维度（gate 和 up）
            bias=False,  # 不使用偏置
            gather_output=False,  # 不收集输出
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.c_proj = RowParallelLinear(  # 输出投影层
            intermediate_size,  # 输入中间维度
            hidden_size,  # 输出隐藏维度
            bias=False,  # 不使用偏置
            input_is_parallel=True,  # 输入已并行分区
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_proj", prefix),  # 参数前缀
        )
        if hidden_act != "silu":  # 检查激活函数是否为 SiLU
            raise ValueError(  # 不支持则抛出异常
                f"Unsupported activation: {hidden_act}. "  # 不支持的激活函数提示
                "Only silu is supported for now."  # 仅支持 SiLU
            )
        self.act_fn = SiluAndMul()  # 创建 SiLU 与乘法融合激活函数

    def forward(self, x):
        """MLP 前向传播：门控投影 -> 激活 -> 输出投影"""
        gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影层
        x = self.act_fn(gate_up)  # 应用 SiLU 与乘法激活
        x, _ = self.c_proj(x)  # 通过输出投影层
        return x  # 返回输出


class QWenAttention(nn.Module):
    """QWen 模型的注意力模块，使用旋转位置编码和 Radix 注意力"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化 QWen 注意力模块"""
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 隐藏维度
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tensor_model_parallel_world_size == 0  # 断言头数可被并行度整除
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size  # 每个并行单元的头数
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度

        # pylint: disable=invalid-name  # 禁用无效名称检查
        self.c_attn = QKVParallelLinear(  # QKV 投影层
            hidden_size,  # 输入维度
            self.head_dim,  # 每头维度
            self.total_num_heads,  # 总头数
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_attn", prefix),  # 参数前缀
        )
        self.c_proj = RowParallelLinear(  # 输出投影层
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            input_is_parallel=True,  # 输入已并行分区
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_proj", prefix),  # 参数前缀
        )
        self.rotary_emb = get_rope(  # 创建旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 旋转基底
            rope_scaling=rope_scaling,  # 旋转缩放配置
        )
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.attn = RadixAttention(  # 创建 Radix 注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_heads,  # KV 头数（与 Q 相同，无 GQA）
            layer_id=layer_id,  # 层 ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """注意力前向传播：QKV 投影 -> 旋转编码 -> 注意力计算 -> 输出投影"""
        qkv, _ = self.c_attn(hidden_states)  # 计算 QKV
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # 分割 Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力
        output, _ = self.c_proj(attn_output)  # 输出投影
        return output  # 返回输出


class QWenBlock(nn.Module):
    """QWen 模型的 Transformer 解码器块，包含注意力和 MLP"""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化 QWen 解码器块"""
        super().__init__()  # 调用父类初始化
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)  # 第一个 RMS 归一化

        rope_theta, rope_scaling = get_rope_config(config)  # 获取 RoPE 配置
        self.attn = QWenAttention(  # 创建注意力模块
            config.hidden_size,  # 隐藏维度
            config.num_attention_heads,  # 注意力头数
            config.max_position_embeddings,  # 最大位置数
            rope_theta=rope_theta,  # 旋转基底
            rope_scaling=rope_scaling,  # 旋转缩放
            layer_id=layer_id,  # 层 ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)  # 第二个 RMS 归一化

        self.mlp = QWenMLP(  # 创建 MLP 模块
            config.hidden_size,  # 隐藏维度
            config.intermediate_size // 2,  # 中间维度（注意 QWen 使用一半）
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """解码器块前向传播：注意力残差连接 + MLP 残差连接"""
        # Self Attention  # 自注意力
        residual = hidden_states  # 保存残差
        hidden_states = self.ln_1(hidden_states)  # 归一化
        hidden_states = self.attn(  # 注意力计算
            positions=positions,  # 位置信息
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = residual + hidden_states  # 残差连接

        # Fully Connected  # 全连接层
        residual = hidden_states  # 保存残差
        hidden_states = self.ln_2(hidden_states)  # 归一化
        hidden_states = self.mlp(hidden_states)  # MLP 计算
        hidden_states = residual + hidden_states  # 残差连接
        return hidden_states  # 返回输出


class QWenModel(nn.Module):
    """QWen 模型主体，包含嵌入层、多层 Transformer 块和最终归一化"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化 QWen 模型主体"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 词表大小

        vocab_size = ((config.vocab_size + 63) // 64) * 64  # 向上取整到 64 的倍数
        self.wte = VocabParallelEmbedding(  # 词嵌入层
            vocab_size,  # 词表大小
            config.hidden_size,  # 嵌入维度
            prefix=add_prefix("wte", prefix),  # 参数前缀
        )
        self.h = nn.ModuleList(  # Transformer 块列表
            [
                QWenBlock(  # 创建解码器块
                    config,  # 配置
                    i,  # 层 ID
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"h.{i}", prefix),  # 参数前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历所有层
            ]
        )
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)  # 最终 RMS 归一化

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """模型主体前向传播：嵌入 -> 多层 Transformer -> 归一化"""
        hidden_states = self.wte(input_ids)  # 词嵌入
        for i in range(len(self.h)):  # 遍历所有层
            layer = self.h[i]  # 获取当前层
            hidden_states = layer(  # 通过当前层
                positions,  # 位置信息
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
            )
        hidden_states = self.ln_f(hidden_states)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class QWenLMHeadModel(nn.Module):
    """QWen 语言模型头部，在模型主体基础上添加语言模型头用于生成 logits"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """初始化 QWen 语言模型头部"""
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.transformer = QWenModel(  # 创建模型主体
            config, quant_config=quant_config, prefix=add_prefix("transformer", prefix)  # 传入配置、量化配置和前缀
        )
        vocab_size = ((config.vocab_size + 63) // 64) * 64  # 向上取整到 64 的倍数
        self.lm_head = ParallelLMHead(  # 语言模型头
            vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)  # 词表大小、隐藏维度和前缀
        )
        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """语言模型前向传播：模型主体 -> logits 处理"""
        hidden_states = self.transformer(input_ids, positions, forward_batch)  # 通过模型主体
        return self.logits_processor(  # 处理 logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 输入 ID、隐藏状态、语言模型头和批次
        )

    @torch.no_grad()  # 禁用梯度计算
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based  # 分割区间 [起始, 结束)，从 0 开始
    ):
        """分离预填充前向传播，支持逐层分割执行"""
        start, end = split_interval  # 解包起始和结束索引
        # embed  # 嵌入
        if start == 0:  # 如果从第 0 层开始
            forward_batch.hidden_states = self.transformer.wte(input_ids)  # 计算词嵌入

        # decoder layer  # 解码器层
        for i in range(start, end):  # 遍历指定范围的层
            layer = self.transformer.h[i]  # 获取当前层
            forward_batch.hidden_states = layer(  # 通过当前层
                positions,  # 位置信息
                forward_batch.hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
            )

        if end == self.transformer.config.num_hidden_layers:  # 如果到达最后一层
            # norm  # 归一化
            forward_batch.hidden_states = self.transformer.ln_f(  # 最终归一化
                forward_batch.hidden_states  # 隐藏状态
            )
            # logits process  # logits 处理
            result = self.logits_processor(  # 处理 logits
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch  # 输入 ID、隐藏状态、语言模型头和批次
            )
        else:  # 未到达最后一层
            result = None  # 结果为空

        return result  # 返回结果

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理门控上投影的堆叠参数映射"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("gate_up_proj", "w2", 0),  # gate 投影
            ("gate_up_proj", "w1", 1),  # up 投影
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入逆频率
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数处理
                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = QWenLMHeadModel  # 模型入口类
