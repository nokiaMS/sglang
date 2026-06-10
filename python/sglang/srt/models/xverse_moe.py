# XVERSE MoE（混合专家）模型实现
# 本文件实现了仅推理的XVERSE MoE模型，包含MLP、MoE、注意力层、解码器层和完整模型。
# 支持共享专家和路由专家，以及NPU和CUDA后端。

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
"""Inference-only XVERSE MoE model."""  # 仅推理的XVERSE MoE模型

from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import (  # 导入分布式通信函数
    get_tensor_model_parallel_rank,  # 获取TP秩
    get_tensor_model_parallel_world_size,  # 获取TP世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (  # 导入NPU MoE方法
    fused_moe_npu,
)
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU和乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # 导入MoE运行器配置
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe  # 导入融合MoE
from sglang.srt.layers.moe.topk import TopK  # 导入Top-K选择
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix, is_npu  # 导入工具函数
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入RoPE配置工具


class XverseMLP(nn.Module):
    """XVERSE MoE模型的MLP模块"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控上投影合并层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小列表
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # 下投影层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            reduce_results=reduce_results,  # 是否归约结果
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
        )
        if hidden_act != "silu":  # 仅支持SiLU
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU和乘法激活

    def forward(self, x):
        """MLP前向传播"""
        gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影
        x = self.act_fn(gate_up)  # 应用激活
        x, _ = self.down_proj(x)  # 通过下投影
        return x  # 返回输出


class XverseMoE(nn.Module):
    """XVERSE MoE模块，包含路由专家和可选的共享专家"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.rank = get_tensor_model_parallel_rank()  # 获取TP秩
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.n_routed_experts = config.num_experts  # 路由专家数
        self.top_k = config.moe_top_k  # Top-K值
        if self.tp_size > self.n_routed_experts:  # TP大小超过专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )

        self.experts = nn.ModuleList(  # 专家模块列表
            [
                XverseMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,  # 不归约，后续统一归约
                    prefix=add_prefix(f"experts.{i}", prefix),
                )
                for i in range(self.n_routed_experts)
            ]
        )
        self.pack_params()  # 打包专家参数
        self.moe_runner_config = MoeRunnerConfig(inplace=True)  # MoE运行器配置

        self.router = ReplicatedLinear(  # 路由器
            config.hidden_size,
            self.n_routed_experts,  # 输出大小等于专家数
            bias=False,
            quant_config=None,
            prefix=add_prefix("router", prefix),
        )
        self.topk = TopK(  # Top-K选择
            top_k=self.top_k,
            renormalize=getattr(self.config, "norm_topk_prob", False),  # 是否重归一化
        )

        if config.num_shared_experts is not None:  # 如果有共享专家
            intermediate_size = config.intermediate_size * config.num_shared_experts  # 共享专家中间大小
            self.shared_experts = XverseMLP(  # 共享专家MLP
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,  # 不归约
                prefix=add_prefix("shared_experts", prefix),
            )
        self.fused_moe_method = fused_moe if not is_npu() else fused_moe_npu  # 选择MoE方法

    def pack_params(self):
        """打包专家参数为融合格式"""
        w1 = []  # gate_up权重列表
        w2 = []  # down权重列表
        for expert in self.experts:  # 遍历专家
            w1.append(expert.gate_up_proj.weight)  # 添加gate_up权重
            w2.append(expert.down_proj.weight)  # 添加down权重
        self.w1 = torch._utils._flatten_dense_tensors(w1)  # 展平w1
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)  # 反展平
        for data, param in zip(w1s, w1):  # 同步参数数据
            param.data = data
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)  # 重塑为3D

        self.w2 = torch._utils._flatten_dense_tensors(w2)  # 展平w2
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)  # 反展平
        for data, param in zip(w2s, w2):  # 同步参数数据
            param.data = data

        self.w2 = self.w2.view(len(w2), *w2s[0].shape)  # 重塑为3D

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE前向传播：路由 -> Top-K -> 专家计算 -> 合并"""
        num_tokens, hidden_dim = hidden_states.shape  # 获取token数和隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑为2D
        if self.config.num_shared_experts is not None:  # 如果有共享专家
            shared_output = self.shared_experts(hidden_states)  # 共享专家输出
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.router(hidden_states)  # 路由logits
        topk_output = self.topk(hidden_states, router_logits)  # Top-K选择
        final_hidden_states = self.fused_moe_method(  # 融合MoE计算
            hidden_states,
            self.w1,  # gate_up权重
            self.w2,  # down权重
            topk_output,
            self.moe_runner_config,  # MoE配置
        )

        if self.config.num_shared_experts is not None:  # 合并共享专家输出
            final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 全归约

        return final_hidden_states.view(num_tokens, hidden_dim)  # 返回重塑后的输出


class XverseAttention(nn.Module):
    """XVERSE MoE模型的注意力模块"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID
        rope_theta: float = 10000,  # RoPE theta
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放
        max_position_embeddings: int = 8192,  # 最大位置编码
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.total_num_heads = num_heads  # 总头数
        assert self.total_num_heads % tp_size == 0  # 断言可整除
        self.num_heads = self.total_num_heads // tp_size  # TP后头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0  # 断言可整除
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0  # 断言可整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # TP后KV头数
        self.head_dim = hidden_size // self.total_num_heads  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 保存RoPE theta
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置编码

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """注意力前向传播"""
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分离Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力
        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class XverseDecoderLayer(nn.Module):
    """XVERSE MoE解码器层，包含注意力和MLP/MoE"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置
        num_key_value_heads = getattr(  # KV头数
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.self_attn = XverseAttention(  # 自注意力层
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        if config.num_experts is not None:  # 如果有专家（MoE层）
            self.mlp = XverseMoE(  # MoE MLP
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:  # 普通MLP层
            self.mlp = XverseMLP(  # 普通MLP
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
    ) -> torch.Tensor:
        """解码器层前向传播"""
        # Self Attention
        if residual is None:  # 无残差
            residual = hidden_states  # 设置残差
            hidden_states = self.input_layernorm(hidden_states)  # 归一化
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 带残差归一化
        hidden_states = self.self_attn(  # 通过自注意力
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP/MoE
        return hidden_states, residual  # 返回隐藏状态和残差


class XverseModel(nn.Module):
    """XVERSE MoE模型主体"""

    fall_back_to_pt_during_load = False  # 不回退到PyTorch加载

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                XverseDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """模型主体前向传播"""
        hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层
        residual = None  # 初始化残差
        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class XverseMoeForCausalLM(nn.Module):
    """XVERSE MoE因果语言模型"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = XverseModel(  # 创建模型主体
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 并行LM头
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """因果语言模型前向传播"""
        hidden_states = self.model(input_ids, positions, forward_batch)  # 通过模型
        return self.logits_processor(  # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),  # Q投影
            ("qkv_proj", "k_proj", "k"),  # K投影
            ("qkv_proj", "v_proj", "v"),  # V投影
            ("gate_up_proj", "gate_proj", 0),  # 门控投影
            ("gate_up_proj", "up_proj", 1),  # 上投影
        ]
        params_dict = dict(self.named_parameters())  # 参数字典

        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转频率
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射
                if weight_name not in name:  # 不匹配
                    continue
                name = name.replace(weight_name, param_name)  # 替换名称
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ偏置
                    continue
                # Skip experts that are not assigned to this worker.
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:  # 跳过非本秩专家
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 非堆叠参数
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ偏置
                    continue
                # Skip experts that are not assigned to this worker.
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:  # 跳过非本秩专家
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = XverseMoeForCausalLM  # 入口类
