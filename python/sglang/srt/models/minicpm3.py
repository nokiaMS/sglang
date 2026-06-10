# MiniCPM3模型：基于多头潜在注意力(MLA)的推理优化实现
# 该模块实现了MiniCPM3因果语言模型，采用MLA（Multi-head Latent Attention）机制
# 核心特性：使用低秩压缩的QKV投影、FP8量化支持、KV缓存压缩
# 主要组件：MiniCPM3MLP、MiniCPM3AttentionMLA、MiniCPM3DecoderLayer、MiniCPM3ForCausalLM

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
"""Inference-only MiniCPM3 model compatible with HuggingFace weights."""

import math  # 导入数学模块
from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入TP世界大小
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU和乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入RoPE获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix, is_cuda  # 导入工具函数
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入RoPE配置获取

if is_cuda():  # 如果是CUDA设备
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8  # 导入FP8批量矩阵乘法

    from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

    # TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
    # Wrap bmm_fp8 as a custom op so torch.compile does not trace into
    # torch.cuda.current_blas_handle() (which returns a non-Tensor).
    @register_custom_op(mutates_args=["out"])  # 注册自定义算子
    def _bmm_fp8_op(  # FP8批量矩阵乘法算子
        A: torch.Tensor,  # 矩阵A
        B: torch.Tensor,  # 矩阵B
        out: torch.Tensor,  # 输出张量
        A_scale: torch.Tensor,  # A的缩放因子
        B_scale: torch.Tensor,  # B的缩放因子
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)  # 调用原始FP8批量矩阵乘法

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        """FP8批量矩阵乘法封装函数"""
        if out is None:  # 如果没有提供输出张量
            out = torch.empty(  # 创建输出张量
                (A.shape[0], A.shape[1], B.shape[2]),  # 输出形状
                device=A.device,  # 设备
                dtype=dtype,  # 数据类型
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)  # 调用FP8算子
        return out  # 返回输出


class MiniCPM3MLP(nn.Module):
    """MiniCPM3多层感知机模块"""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控上投影（合并的gate和up）
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小（gate和up各一）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 前缀
        )
        self.down_proj = RowParallelLinear(  # 下投影
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 前缀
        )
        if hidden_act != "silu":  # 如果激活函数不是SiLU
            raise ValueError(  # 抛出异常
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU和乘法激活函数

    def forward(self, x):
        """MLP前向传播"""
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影
        x = self.act_fn(gate_up)  # 激活函数
        x, _ = self.down_proj(x)  # 下投影
        return x  # 返回输出


def input_to_float8(x, dtype=torch.float8_e4m3fn):
    """将输入张量转换为FP8格式，返回量化后的张量和缩放因子"""
    finfo = torch.finfo(dtype)  # 获取FP8数据类型信息
    min_val, max_val = x.aminmax()  # 计算最小最大值
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)  # 计算绝对最大值
    scale = finfo.max / amax  # 计算缩放因子
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)  # 缩放并截断
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()  # 返回FP8张量和逆缩放因子


class MiniCPM3AttentionMLA(nn.Module):
    """MiniCPM3多头潜在注意力(MLA)模块，使用低秩压缩的KV投影"""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 层ID
        self.hidden_size = hidden_size  # 隐藏层大小
        self.qk_nope_head_dim = qk_nope_head_dim  # QK非旋转头维度
        self.qk_rope_head_dim = qk_rope_head_dim  # QK旋转头维度
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # QK头总维度
        self.v_head_dim = v_head_dim  # V头维度
        self.q_lora_rank = q_lora_rank  # Q低秩
        self.kv_lora_rank = kv_lora_rank  # KV低秩
        self.num_heads = num_heads  # 注意力头数
        tp_size = get_tensor_model_parallel_world_size()  # TP大小
        assert num_heads % tp_size == 0  # 断言头数可被TP大小整除
        self.num_local_heads = num_heads // tp_size  # 本地头数
        self.scaling = self.qk_head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # RoPE theta
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入

        if self.q_lora_rank is not None:  # 如果使用Q低秩投影
            self.q_a_proj = ReplicatedLinear(  # Q压缩投影
                self.hidden_size,  # 输入大小
                self.q_lora_rank,  # 输出大小（低秩）
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("q_a_proj", prefix),  # 前缀
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)  # Q压缩归一化
            self.q_b_proj = ColumnParallelLinear(  # Q展开投影
                q_lora_rank,  # 输入大小（低秩）
                self.num_heads * self.qk_head_dim,  # 输出大小
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("q_b_proj", prefix),  # 前缀
            )
        else:  # 不使用Q低秩投影
            self.q_proj = ColumnParallelLinear(  # 直接Q投影
                self.hidden_size,  # 输入大小
                self.num_heads * self.qk_head_dim,  # 输出大小
                bias=False,  # 不使用偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("q_proj", prefix),  # 前缀
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(  # KV压缩投影（多查询注意力）
            self.hidden_size,  # 输入大小
            self.kv_lora_rank + self.qk_rope_head_dim,  # 输出大小（KV低秩+旋转维度）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),  # 前缀
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)  # KV压缩归一化
        self.kv_b_proj = ColumnParallelLinear(  # KV展开投影
            self.kv_lora_rank,  # 输入大小（低秩）
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("kv_b_proj", prefix),  # 前缀
        )
        # O projection.
        self.o_proj = RowParallelLinear(  # 输出投影
            self.num_heads * self.v_head_dim,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 前缀
        )
        self.rotary_emb = get_rope(  # 旋转位置嵌入
            qk_rope_head_dim,  # 旋转维度
            rotary_dim=qk_rope_head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基础频率
            rope_scaling=rope_scaling,  # 缩放参数
        )

        self.attn = RadixAttention(  # 基数注意力
            self.num_local_heads,  # 本地头数
            self.kv_lora_rank + self.qk_rope_head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=1,  # KV头数为1（多查询注意力）
            layer_id=layer_id,  # 层ID
            v_head_dim=self.kv_lora_rank,  # V头维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 前缀
        )

        self.w_kc = None  # K压缩权重矩阵
        self.w_vc = None  # V压缩权重矩阵
        self.w_scale = None  # FP8缩放因子

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """MLA注意力前向传播：压缩QKV、应用RoPE、计算注意力、输出投影"""
        q_len = hidden_states.shape[0]  # 查询长度
        q_input = hidden_states.new_empty(  # 创建查询输入张量
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:  # 如果使用Q低秩投影
            q = self.q_a_proj(hidden_states)[0]  # Q压缩投影
            q = self.q_a_layernorm(q)  # Q压缩归一化
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)  # Q展开投影
        else:  # 不使用Q低秩投影
            q = self.q_proj(hidden_states)[0].view(  # 直接Q投影
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # 分离非旋转和旋转部分

        if self.w_kc.dtype == torch.float8_e4m3fn:  # 如果K压缩权重为FP8
            q_nope_val, q_nope_scale = input_to_float8(  # 将Q非旋转部分转为FP8
                q_nope.transpose(0, 1), torch.float8_e4m3fn
            )
            q_nope_out = bmm_fp8(  # FP8批量矩阵乘法
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:  # 非FP8
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)  # 普通批量矩阵乘法
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)  # 填入Q的KV低秩部分

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]  # KV压缩投影
        v_input = latent_cache[..., : self.kv_lora_rank]  # 取V部分
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)  # V归一化
        k_input = latent_cache.unsqueeze(1)  # KV潜在缓存
        k_input[..., : self.kv_lora_rank] = v_input  # 填入K的V部分
        k_pe = k_input[..., self.kv_lora_rank :]  # 取K的旋转部分

        original_shapes = [q_pe.shape, k_pe.shape]  # 保存原始形状
        q_pe, k_pe = self.rotary_emb(  # 应用旋转位置嵌入
            positions,  # 位置
            q_pe.reshape(-1, q_pe.shape[1] * q_pe.shape[2]),  # 展平Q旋转部分
            k_pe.reshape(-1, k_pe.shape[1] * k_pe.shape[2]),  # 展平K旋转部分
        )
        q_pe, k_pe = q_pe.view(original_shapes[0]), k_pe.view(original_shapes[1])  # 恢复原始形状
        q_input[..., self.kv_lora_rank :] = q_pe  # 填入Q的旋转部分
        k_input[..., self.kv_lora_rank :] = k_pe  # 填入K的旋转部分

        attn_output = self.attn(q_input, k_input, v_input, forward_batch)  # 计算注意力
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)  # 重塑输出

        if self.w_vc.dtype == torch.float8_e4m3fn:  # 如果V压缩权重为FP8
            attn_output_val, attn_output_scale = input_to_float8(  # 将注意力输出转为FP8
                attn_output.transpose(0, 1), torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(  # FP8批量矩阵乘法
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:  # 非FP8
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)  # 普通批量矩阵乘法
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)  # 转置并展平
        output, _ = self.o_proj(attn_output)  # 输出投影

        return output  # 返回输出


class MiniCPM3DecoderLayer(nn.Module):
    """MiniCPM3解码器层，包含MLA注意力和MLP"""
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏层大小
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入
        self.self_attn = MiniCPM3AttentionMLA(  # MLA自注意力
            config=config,  # 配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            qk_nope_head_dim=config.qk_nope_head_dim,  # QK非旋转头维度
            qk_rope_head_dim=config.qk_rope_head_dim,  # QK旋转头维度
            v_head_dim=self.hidden_size // config.num_attention_heads,  # V头维度
            q_lora_rank=(  # Q低秩
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,  # KV低秩
            rope_theta=rope_theta,  # RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入
            quant_config=quant_config,  # 量化配置
            layer_id=layer_id,  # 层ID
            prefix=add_prefix("self_attn", prefix),  # 前缀
        )

        self.mlp = MiniCPM3MLP(  # MLP层
            hidden_size=self.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播"""
        # Self Attention
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入归一化
        hidden_states = self.self_attn(  # 自注意力计算
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
        )
        hidden_states = residual + hidden_states * (  # 残差连接，带缩放因子
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        # Fully Connected
        residual = hidden_states  # 更新残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化
        hidden_states = self.mlp(hidden_states)  # MLP计算
        hidden_states = residual + hidden_states * (  # 残差连接，带缩放因子
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        return hidden_states, None  # 返回隐藏状态，残差为None


class MiniCPM3Model(nn.Module):
    """MiniCPM3模型主体"""
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充索引
        self.vocab_size = config.vocab_size  # 词表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            self.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层大小
            org_num_embeddings=config.vocab_size,  # 原始嵌入数
            prefix=add_prefix("embed_tokens", prefix),  # 前缀
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                MiniCPM3DecoderLayer(
                    config,  # 配置
                    i,  # 层ID
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历层数
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """模型主体前向传播"""
        if input_embeds is None:  # 如果没有预计算的嵌入
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb  # 通过词嵌入获取并缩放
        else:  # 有预计算的嵌入
            hidden_states = input_embeds  # 直接使用
        residual = None  # 初始化残差

        for i in range(len(self.layers)):  # 遍历每一层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过解码器层
                positions,  # 位置
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次
                residual,  # 残差
            )
        hidden_states = self.norm(hidden_states)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class MiniCPM3ForCausalLM(nn.Module):
    """MiniCPM3因果语言模型"""
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        self.num_experts = getattr(self.config, "num_experts", 0)  # 专家数量
        self.quant_config = quant_config  # 量化配置
        self.model = MiniCPM3Model(  # 模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if not self.config.tie_word_embeddings:  # 如果不绑定词嵌入
            self.lm_head = ParallelLMHead(  # 语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                org_num_embeddings=config.vocab_size,  # 原始嵌入数
                prefix=add_prefix("lm_head", prefix),  # 前缀
            )

        self.scale_width = self.config.hidden_size / self.config.dim_model_base  # 宽度缩放因子

        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """因果语言模型前向传播"""
        if input_embeds is not None:  # 如果有预计算的嵌入
            input_embeds = input_embeds * self.config.scale_emb  # 缩放嵌入
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 模型主体前向
        hidden_states = hidden_states / self.scale_width  # 宽度缩放
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入
            lm_head = self.model.embed_tokens  # 使用词嵌入作为LM头
        else:  # 不绑定
            lm_head = self.lm_head  # 使用独立的LM头
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)  # 返回logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，包括MLA的KV压缩权重预处理"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),  # 门控投影
            ("gate_up_proj", "up_proj", 1),  # 上投影
        ]
        expert_params_mapping = [  # 专家参数映射
            # (param_name, weight_name, expert_id)
            (
                "ws" if weight_name in ["w1", "w3"] else "w2s",  # 根据权重名确定参数名
                f"experts.{expert_id}.{weight_name}.weight",  # 权重名称
                expert_id,  # 专家ID
            )
            for expert_id in range(self.num_experts)  # 遍历专家
            for weight_name in ["w1", "w2", "w3"]  # 遍历权重名
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入逆频率
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过旋转嵌入缓存
                # Models trained using ColossalAI may include these tensors in the
                # checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的词嵌入权重
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数
                if weight_name not in name:  # 如果权重名不在名称中
                    continue
                name = name.replace(weight_name, param_name)  # 替换权重名
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 非堆叠参数
                for param_name, weight_name, expert_id in expert_params_mapping:  # 遍历专家参数
                    if weight_name not in name:  # 如果权重名不在名称中
                        continue
                    name = name.replace(weight_name, param_name)  # 替换权重名
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载专家权重
                        param, loaded_weight, weight_name, expert_id=expert_id
                    )
                    break
                else:  # 非专家参数
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重

        for layer_id in range(self.config.num_hidden_layers):  # 遍历每一层
            self_attn = self.model.layers[layer_id].self_attn  # 获取自注意力层
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(  # 从KV展开投影中分离K和V压缩权重
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)  # K压缩权重
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)  # V压缩权重
            if hasattr(self_attn.kv_b_proj, "weight_scale"):  # 如果有FP8缩放因子
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale  # 保存缩放因子
            del self_attn.kv_b_proj  # 删除原始KV展开投影


EntryClass = MiniCPM3ForCausalLM  # 入口类
