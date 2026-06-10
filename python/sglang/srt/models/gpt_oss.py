# GptOss模型推理实现文件
# 本文件实现了仅用于推理的GptOss模型，兼容HuggingFace权重格式
# 包含稀疏MoE块、注意力层、解码器层、模型主体及因果语言模型等核心组件
# 支持张量并行、专家并行、流水线并行及MXFP4量化等特性

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


"""Inference-only GptOss model compatible with HuggingFace weights."""  # 仅推理用的GptOss模型，兼容HuggingFace权重

import logging  # 导入日志模块
import math  # 导入数学模块
import re  # 导入正则表达式模块
from collections.abc import Iterable  # 导入可迭代类型
from functools import partial  # 导入偏函数工具
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置基类

from sglang.srt.compilation.piecewise_context_manager import (  # 导入分段编译上下文管理器
    get_forward_context,  # 获取前向上下文
    is_in_piecewise_cuda_graph,  # 判断是否在分段CUDA图中
)
from sglang.srt.distributed import (  # 导入分布式通信模块
    get_moe_expert_parallel_rank,  # 获取MoE专家并行排名
    get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
    get_moe_tensor_parallel_rank,  # 获取MoE张量并行排名
    get_moe_tensor_parallel_world_size,  # 获取MoE张量并行世界大小
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_rank,  # 获取张量并行排名
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入全局专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置模型配置
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes  # 导入层通信器和层散射模式
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力模块
    get_attention_tp_rank,  # 获取注意力TP排名
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 判断是否启用数据并行注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入线性层模块
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe import get_moe_a2a_backend  # 导入MoE全互连后端获取函数
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类获取函数
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE Triton层
from sglang.srt.layers.moe.topk import TopK  # 导入TopK选择器
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert  # 导入MoE权重参数过滤函数
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.quantization.fp8_utils import dequant_mxfp4  # 导入MXFP4反量化工具
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入流水线缺失层和层ID获取工具
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和流水线代理张量
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.utils import (  # 导入模型工具函数
    create_fused_set_kv_buffer_arg,  # 创建融合设置KV缓冲区参数
    enable_fused_set_kv_buffer,  # 启用融合设置KV缓冲区
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import (  # 导入工具函数
    LazyValue,  # 懒加载值
    add_prefix,  # 添加前缀
    get_cuda_version,  # 获取CUDA版本
    is_blackwell_supported,  # 判断是否支持Blackwell架构
    is_cpu,  # 判断是否为CPU
    is_cuda,  # 判断是否为CUDA
    is_flashinfer_available,  # 判断FlashInfer是否可用
    is_npu,  # 判断是否为NPU
    is_sm90_supported,  # 判断是否支持SM90
    make_layers,  # 创建层
)
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册函数

_is_cpu = is_cpu()  # 是否为CPU环境
_is_npu = is_npu()  # 是否为NPU环境
_is_cuda = is_cuda()  # 是否为CUDA环境
_is_tinygemm_supported = (  # 是否支持TinyGEMM
    _is_cuda  # 需要是CUDA环境
    and is_flashinfer_available()  # FlashInfer可用
    and (is_sm90_supported() or is_blackwell_supported())  # 支持SM90或Blackwell
)

if _is_tinygemm_supported and get_cuda_version()[0] < 13:  # 如果支持TinyGEMM且CUDA版本小于13
    try:  # 尝试导入
        from flashinfer.gemm import tinygemm_bf16  # 导入FlashInfer的TinyGEMM BF16实现
    except ImportError:  # 导入失败
        tinygemm_bf16 = None  # 设为None
        _is_tinygemm_supported = False  # 标记不支持
else:  # 否则不支持TinyGEMM
    tinygemm_bf16 = None  # 设为None
    _is_tinygemm_supported = False  # 标记不支持


class GptOssConfig(PretrainedConfig):  # GptOss配置类，继承自预训练配置基类
    model_type = "gpt_oss"  # 模型类型标识

    def __init__(self, **kwargs):  # 初始化函数
        super().__init__(**kwargs)  # 调用父类初始化


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


# Aligned with HF's implementation, using sliding window inclusive with the last token  # 与HuggingFace实现对齐，滑动窗口包含最后一个token
# SGLang assumes exclusive  # SGLang假设滑动窗口不包含最后一个token
def get_attention_sliding_window_size(config):  # 获取注意力滑动窗口大小
    return config.sliding_window - 1  # 返回滑动窗口大小减1（转换为SGLang的排他模式）


class TinyGemmLinear(ReplicatedLinear):  # TinyGEMM线性层类，继承自复制线性层
    """ReplicatedLinear with a FlashInfer tinygemm BF16 fast path."""  # 带FlashInfer tinygemm BF16快速路径的复制线性层

    def __init__(self, *args, **kwargs):  # 初始化函数
        super().__init__(*args, **kwargs)  # 调用父类初始化
        self._use_tinygemm = (  # 是否使用TinyGEMM
            _is_tinygemm_supported  # 环境支持TinyGEMM
            and not self.skip_bias_add  # 不跳过偏置添加
            and self.weight.is_contiguous()  # 权重张量是连续的
            and self.weight.shape[0] % 16 == 0  # 权重行数可被16整除
            and self.weight.shape[1] % 64 == 0  # 权重列数可被64整除
            and self.weight.dtype == torch.bfloat16  # 权重数据类型为BF16
            and (  # 并且
                self.bias is None  # 偏置为None
                or (  # 或者
                    self.bias.dtype == torch.bfloat16  # 偏置数据类型为BF16
                    and self.bias.is_contiguous()  # 偏置张量是连续的
                    and self.bias.shape[0] == self.weight.shape[0]  # 偏置形状与权重行数一致
                )
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 前向传播函数
        if (  # 如果满足TinyGEMM使用条件
            self._use_tinygemm  # 启用TinyGEMM
            and x.ndim == 2  # 输入为2维
            and x.is_cuda  # 输入在CUDA上
            and x.shape[0] <= 128  # 批次大小不超过128
            and x.is_contiguous()  # 输入是连续的
            and x.shape[1] == self.weight.shape[1]  # 输入特征维度与权重列数一致
            and x.dtype == torch.bfloat16  # 输入数据类型为BF16
        ):
            out = x.new_empty((x.shape[0], self.output_size))  # 创建输出张量
            tinygemm_bf16(x, self.weight, out, self.bias)  # 使用TinyGEMM BF16核函数计算
            return out, None  # 返回输出和None（无偏置）

        return super().forward(x)  # 否则使用父类前向传播


class GptOssSparseMoeBlock(nn.Module):  # GptOss稀疏MoE块类
    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: GptOssConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行世界大小
        self.layer_id = layer_id  # 层ID
        self.activation = config.hidden_act  # 激活函数名称
        self.gemm1_alpha = getattr(config, "hidden_act_alpha", 1.702)  # GEMM1的alpha参数，默认1.702
        self.gemm1_clamp_limit = config.swiglu_limit  # SwiGLU限幅值

        self.topk = TopK(  # TopK选择器
            top_k=config.num_experts_per_tok,  # 每个token选择的专家数
            renormalize=True,  # 重归一化
            layer_id=layer_id,  # 层ID
        )

        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数
        experts_type = get_moe_impl_class(quant_config)  # 获取MoE实现类
        extra_kwargs = {}  # 额外参数字典
        if experts_type.__name__ == "FusedMoE":  # 如果是融合MoE实现
            quant_config_name = (  # 获取量化配置名称
                quant_config.get_name() if quant_config is not None else None  # 有量化配置则获取名称，否则为None
            )
            extra_kwargs = {  # 设置额外参数
                # for moe gate_up_proj and down_proj and their bias loading  # 用于MoE gate_up_proj和down_proj及其偏置加载
                "use_weight_loader_fused": quant_config_name  # 是否使用融合权重加载器
                != "mxfp4"  # 量化配置不是mxfp4时使用
            }

        self.experts = experts_type(  # MoE专家层
            num_experts=config.num_local_experts  # 本地专家数量
            + get_global_server_args().ep_num_redundant_experts,  # 加上冗余专家数量
            top_k=config.num_experts_per_tok,  # TopK值
            layer_id=layer_id,  # 层ID
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            quant_config=quant_config,  # 量化配置
            activation=self.activation,  # 激活函数
            gemm1_alpha=self.gemm1_alpha,  # GEMM1 alpha参数
            gemm1_clamp_limit=self.gemm1_clamp_limit,  # GEMM1限幅值
            with_bias=True,  # 使用偏置
            prefix=add_prefix("experts", prefix),  # 参数前缀
            **extra_kwargs,  # 额外参数
        )

        self.router = TinyGemmLinear(  # 路由器线性层
            config.hidden_size,  # 输入维度
            config.num_local_experts,  # 输出维度（专家数）
            bias=True,  # 使用偏置
            quant_config=None,  # 路由器不使用量化
            prefix=add_prefix("gate", prefix),  # 参数前缀
            params_dtype=config.dtype,  # 参数数据类型
        )

    def forward(  # 前向传播函数
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次信息，可选
        should_allreduce_fusion: bool = False,  # 是否融合全归约，默认False
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep():  # 如果不是DeepEP后端
            return self.forward_normal(hidden_states, should_allreduce_fusion)  # 使用普通前向传播
        else:  # 否则
            raise Exception("forward_deepep branch not implemented yet")  # 抛出异常，DeepEP分支未实现

    def get_moe_weights(self):  # 获取MoE权重
        return [  # 返回权重列表
            x.data  # 获取参数数据
            for name, x in self.experts.named_parameters()  # 遍历专家层参数
            if name not in ["correction_bias"]  # 排除修正偏置
            and filter_moe_weight_param_global_expert(  # 过滤全局专家权重参数
                name, x, self.experts.num_local_experts  # 传入名称、参数和本地专家数
            )
        ]

    def forward_normal(  # 普通前向传播函数
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        should_allreduce_fusion: bool = False,  # 是否融合全归约，默认False
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape  # 获取token数和隐藏维度
        if is_in_piecewise_cuda_graph():  # 如果在分段CUDA图中
            final_hidden_states = moe_impl(self.layer_id, hidden_states)  # 使用分段CUDA图的MoE实现
        else:  # 否则
            router_logits, _ = self.router(hidden_states)  # 计算路由器logits
            topk_output = self.topk(hidden_states, router_logits)  # 执行TopK选择
            final_hidden_states = self.experts(hidden_states, topk_output)  # 执行专家计算

        if self.tp_size > 1 and not should_allreduce_fusion:  # 如果TP大小>1且不融合全归约
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 执行张量并行全归约

        ans = final_hidden_states.view(num_tokens, hidden_dim)  # 重新整形输出
        return ans  # 返回结果


@register_custom_op(out_shape="hidden_states")  # 注册自定义算子，输出形状为hidden_states
def moe_impl(layer_id: int, hidden_states: torch.Tensor) -> torch.Tensor:  # MoE实现函数（用于分段CUDA图）
    forward_context = get_forward_context()  # 获取前向上下文
    moe_fusion = forward_context.moe_fusions[layer_id]  # 获取当前层的MoE融合对象
    router_logits, _ = moe_fusion.router(hidden_states)  # 计算路由器logits
    topk_output = moe_fusion.topk(hidden_states, router_logits)  # 执行TopK选择
    final_hidden_states = moe_fusion.experts(hidden_states, topk_output)  # 执行专家计算
    return final_hidden_states  # 返回最终隐藏状态


class GptOssAttention(nn.Module):  # GptOss注意力层类
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID，默认0
        rope_theta: float = 10000,  # RoPE基准角度，默认10000
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置，默认None
        max_position_embeddings: int = 8192,  # 最大位置嵌入数，默认8192
        head_dim: Optional[int] = None,  # 头维度，默认None
        rms_norm_eps: float = 1e-06,  # RMS归一化epsilon，默认1e-6
        attention_bias: bool = False,  # 是否使用注意力偏置，默认False
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
        sliding_window_size: int = -1,  # if -1, normal attention, else, window attention.  # 滑动窗口大小，-1为普通注意力，否则为窗口注意力
        layer_type: str = "",  # 层类型，默认空字符串
        params_dtype: torch.dtype = torch.bfloat16,  # 参数数据类型，默认BF16
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.sliding_window_size = sliding_window_size  # 保存滑动窗口大小

        attn_tp_rank = get_attention_tp_rank()  # 获取注意力TP排名
        attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小

        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % attn_tp_size == 0  # 断言总头数可被TP大小整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 当前分片的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= attn_tp_size:  # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此我们进行分区
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头分配到多个张量并行GPU上
            assert self.total_num_kv_heads % attn_tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此我们进行复制
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头复制到多个张量并行GPU上
            assert attn_tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 当前分片的KV头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # RoPE基准角度
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.tp_rank = get_tensor_model_parallel_rank()  # 张量并行排名

        self.qkv_proj = QKVParallelLinear(  # QKV投影线性层
            hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=attention_bias,  # 是否使用偏置
            params_dtype=params_dtype,  # 参数数据类型
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # 注意力TP排名
            tp_size=attn_tp_size,  # 注意力TP大小
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )

        # Choose dtype of sinks based on attention backend: trtllm_mha requires float32,  # 根据注意力后端选择sinks的数据类型：trtllm_mha需要float32
        # others can use bfloat16  # 其他后端可以使用bfloat16
        attn_backend = get_global_server_args().attention_backend  # 获取注意力后端
        sinks_dtype = torch.float32 if attn_backend == "trtllm_mha" else torch.bfloat16  # 根据后端选择数据类型
        self.sinks = nn.Parameter(  # 注意力汇聚点参数
            torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False  # 不需要梯度
        )

        self.o_proj = RowParallelLinear(  # 输出投影线性层（行并行）
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=attention_bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # 注意力TP排名
            tp_size=attn_tp_size,  # 注意力TP大小
            reduce_results=False,  # 不自动归约结果
            params_dtype=params_dtype,  # 参数数据类型
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基准角度
            rope_scaling=rope_scaling,  # RoPE缩放配置
        )

        assert layer_type in {"sliding_attention", "full_attention"}  # 断言层类型合法
        use_sliding_window = layer_type == "sliding_attention"  # 是否使用滑动窗口
        self.attn = RadixAttention(  # 基数注意力实现
            self.num_heads,  # 注意力头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            prefix=add_prefix("attn", prefix),  # 参数前缀
            sliding_window_size=(sliding_window_size if use_sliding_window else -1),  # 滑动窗口大小
        )
        self.layer_id = layer_id  # 保存层ID

    def forward_prepare(  # 前向传播准备函数
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ):
        if hidden_states.shape[0] == 0:  # 如果没有token
            return hidden_states, forward_batch, None  # 直接返回
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影层
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割为Q、K、V

        extra_args = {}  # 额外参数字典
        if not _is_npu:  # 如果不是NPU
            extra_args = {  # 设置额外参数
                "fused_set_kv_buffer_arg": (  # 融合设置KV缓冲区参数
                    create_fused_set_kv_buffer_arg(  # 创建融合参数
                        value=v,  # V值
                        layer=self.attn,  # 注意力层
                        forward_batch=forward_batch,  # 批次信息
                    )
                    if enable_fused_set_kv_buffer(forward_batch)  # 如果启用了融合设置KV缓冲区
                    else None  # 否则为None
                ),
            }
        q, k = self.rotary_emb(positions, q, k, **extra_args)  # 应用旋转位置编码
        inner_state = q, k, v, forward_batch  # 内部状态
        return None, forward_batch, inner_state  # 返回准备结果

    def forward_core(self, intermediate_state):  # 前向传播核心函数
        hidden_states, forward_batch, inner_state = intermediate_state  # 解包中间状态
        if inner_state is None:  # 如果内部状态为None
            return hidden_states  # 直接返回隐藏状态
        attn_output = self.attn(  # 执行注意力计算
            *inner_state,  # 展开内部状态
            sinks=self.sinks,  # 汇聚点参数
            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),  # 是否保存KV缓存
        )
        output, _ = self.o_proj(attn_output)  # 通过输出投影层
        return output  # 返回输出

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        s = self.forward_prepare(  # 前向传播准备
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 批次信息
        )
        return self.forward_core(s)  # 执行前向传播核心


class GptOssDecoderLayer(nn.Module):  # GptOss解码器层类
    def __init__(  # 初始化函数
        self,
        config: GptOssConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
        sliding_window_size: int | None = None,  # 滑动窗口大小，默认None
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏层大小
        rope_theta = config.rope_parameters["rope_theta"]  # RoPE基准角度
        rope_scaling = config.rope_parameters  # RoPE缩放配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入数
        head_dim = getattr(  # 头维度
            config, "head_dim", config.hidden_size // config.num_attention_heads  # 默认为隐藏大小除以头数
        )
        rms_norm_eps = config.rms_norm_eps  # RMS归一化epsilon
        attention_bias = config.attention_bias  # 注意力偏置

        if sliding_window_size is None:  # 如果没有指定滑动窗口大小
            self.sliding_window_size = get_attention_sliding_window_size(self.config)  # 从配置获取
        else:  # 否则
            self.sliding_window_size = sliding_window_size  # 使用指定值

        self.self_attn = GptOssAttention(  # 自注意力层
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE基准角度
            rope_scaling=rope_scaling,  # RoPE缩放配置
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入数
            head_dim=head_dim,  # 头维度
            rms_norm_eps=rms_norm_eps,  # RMS归一化epsilon
            attention_bias=attention_bias,  # 注意力偏置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
            sliding_window_size=self.sliding_window_size,  # 滑动窗口大小
            layer_type=config.layer_types[layer_id],  # 层类型
            params_dtype=config.dtype,  # 参数数据类型
        )

        self.layer_id = layer_id  # 层ID

        self.attn_tp_size = get_attention_tp_size()  # 注意力TP大小
        self.attn_tp_rank = get_attention_tp_rank()  # 注意力TP排名

        # GptOss all layers are sparse and have no nextn now  # GptOss所有层都是稀疏的，目前没有nextn
        self.is_layer_sparse = True  # 所有层都是稀疏MoE层
        self.is_nextn = False  # 不是nextn层
        is_previous_layer_sparse = True  # 前一层是稀疏的
        is_next_layer_sparse = True  # 下一层是稀疏的

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 初始化层散射模式
            layer_id=layer_id,  # 层ID
            num_layers=config.num_hidden_layers,  # 总层数
            is_layer_sparse=self.is_layer_sparse,  # 当前层是否稀疏
            is_previous_layer_sparse=is_previous_layer_sparse,  # 前一层是否稀疏
            is_next_layer_sparse=is_next_layer_sparse,  # 下一层是否稀疏
        )

        if self.is_layer_sparse:  # 如果是稀疏层
            self.mlp = GptOssSparseMoeBlock(  # 使用稀疏MoE块
                layer_id=self.layer_id,  # 层ID
                config=config,  # 模型配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 参数前缀
            )
        else:  # 否则
            raise NotImplementedError(  # 抛出未实现错误
                "Dense MLP is not implemented for GptOssDecoderLayer. "  # GptOss解码器层未实现稠密MLP
                "Please use GptOssSparseMoeBlock instead."  # 请使用GptOssSparseMoeBlock
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )

        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,  # 层散射模式
            input_layernorm=self.input_layernorm,  # 输入层归一化
            post_attention_layernorm=self.post_attention_layernorm,  # 注意力后层归一化
            is_last_layer=(  # 是否为最后一层
                self.is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)  # nextn层或最后一层
            ),
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差张量，可选
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力计算
            hidden_states, residual, forward_batch  # 传入隐藏状态、残差和批次信息
        )

        if hidden_states.shape[0] != 0:  # 如果有token
            hidden_states = self.self_attn(  # 执行自注意力计算
                positions=positions,  # 位置
                hidden_states=hidden_states,  # 隐藏状态
                forward_batch=forward_batch,  # 批次信息
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP计算
            hidden_states, residual, forward_batch  # 传入隐藏状态、残差和批次信息
        )

        should_allreduce_fusion = (  # 是否融合全归约
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(  # 判断是否将MLP全归约与下一层融合
                forward_batch  # 批次信息
            )
        )

        hidden_states = self.mlp(hidden_states, forward_batch, should_allreduce_fusion)  # 执行MoE MLP计算

        if should_allreduce_fusion:  # 如果需要融合全归约
            hidden_states._sglang_needs_allreduce_fusion = True  # 标记需要融合全归约

        if not should_allreduce_fusion:  # 如果不需要融合全归约
            hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理层
                hidden_states, residual, forward_batch  # 传入隐藏状态、残差和批次信息
            )

        return hidden_states, residual  # 返回隐藏状态和残差


class GptOssModel(nn.Module):  # GptOss模型类
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
        decoder_layer_type: type[nn.Module] = GptOssDecoderLayer,  # 解码器层类型，默认GptOss解码器层
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词汇表大小
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if _is_npu:  # 如果是NPU
            config.hidden_act = "npu_swiglu_oai"  # 使用NPU专用的SwiGLU激活函数

        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个rank
            self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
                config.vocab_size,  # 词汇表大小
                config.hidden_size,  # 嵌入维度
                use_attn_tp_group=is_dp_attention_enabled(),  # 是否使用注意力TP组
                prefix=add_prefix("embed_tokens", prefix),  # 参数前缀
            )
        else:  # 否则
            self.embed_tokens = PPMissingLayer()  # 使用流水线缺失层占位

        # Use the provided decoder layer type or default to GptOssDecoderLayer  # 使用提供的解码器层类型或默认GptOss解码器层
        decoder_layer_type = decoder_layer_type or GptOssDecoderLayer  # 确保解码器层类型
        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建层列表，获取起始和结束层索引
            config.num_hidden_layers,  # 总层数
            lambda idx, prefix: decoder_layer_type(  # 层创建函数
                layer_id=idx,  # 层ID
                config=config,  # 模型配置
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 参数前缀
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行rank
            pp_size=self.pp_group.world_size,  # 流水线并行世界大小
            prefix=add_prefix("layers", prefix),  # 参数前缀
        )
        if self.pp_group.is_last_rank:  # 如果是最后一个rank
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终RMS归一化
        else:  # 否则
            self.norm = PPMissingLayer(return_tuple=True)  # 使用流水线缺失层占位（返回元组）

        self.layers_to_capture = []  # 需要捕获隐藏状态的层列表

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，默认None
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，可选
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:  # 如果是第一个rank
            if input_embeds is None:  # 如果没有提供输入嵌入
                hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入
            else:  # 否则
                hidden_states = input_embeds  # 使用提供的输入嵌入
            residual = None  # 残差初始化为None
        else:  # 否则
            assert pp_proxy_tensors is not None  # 断言流水线代理张量不为None
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理张量获取残差

        aux_hidden_states = []  # 辅助隐藏状态列表
        for i in range(self.start_layer, self.end_layer):  # 遍历当前rank负责的层
            with get_global_expert_distribution_recorder().with_current_layer(i):  # 记录专家分布
                if i in self.layers_to_capture:  # 如果当前层需要捕获
                    aux_hidden_states.append(hidden_states + residual)  # 添加隐藏状态和残差之和
                layer = self.layers[i]  # 获取当前层
                hidden_states, residual = layer(  # 执行当前层前向传播
                    positions, hidden_states, forward_batch, residual  # 传入位置、隐藏状态、批次和残差
                )
        if not self.pp_group.is_last_rank:  # 如果不是最后一个rank
            return PPProxyTensors(  # 返回流水线代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:  # 否则
            if hidden_states.shape[0] != 0:  # 如果有token
                if residual is None:  # 如果没有残差
                    hidden_states = self.norm(hidden_states)  # 直接归一化
                else:  # 否则
                    hidden_states, _ = self.norm(hidden_states, residual)  # 归一化隐藏状态和残差
        if len(aux_hidden_states) == 0:  # 如果没有辅助隐藏状态
            return hidden_states  # 直接返回隐藏状态

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态


class GptOssForCausalLM(nn.Module):  # GptOss因果语言模型类
    fall_back_to_pt_during_load = False  # 加载时不回退到PyTorch默认方式

    _lora_pattern_moe = re.compile(  # MoE LoRA匹配正则表达式
        r"^(?:model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:  # 判断是否应对指定模块应用LoRA
        return bool(self._lora_pattern_moe.match(module_name))  # 匹配MoE LoRA模式

    def __init__(  # 初始化函数
        self,
        config: GptOssConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = GptOssModel(  # GptOss模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词汇表大小
            config.hidden_size,  # 隐藏层大小
            # quant_config=quant_config,  # 量化配置（已注释）
            prefix=add_prefix("lm_head", prefix),  # 参数前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力TP组
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

        self._routed_experts_weights_of_layer = LazyValue(  # 懒加载每层路由专家权重
            lambda: {  # lambda函数
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()  # 获取每层MoE权重
                for layer_id in range(self.start_layer, self.end_layer)  # 遍历当前rank负责的层
                if isinstance(self.model.layers[layer_id].mlp, GptOssSparseMoeBlock)  # 只取稀疏MoE层
            }
        )

    @property
    def routed_experts_weights_of_layer(self):  # 路由专家权重属性
        return self._routed_experts_weights_of_layer.value  # 返回懒加载的值

    @torch.no_grad()  # 禁用梯度计算装饰器
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，默认None
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，可选
    ) -> torch.Tensor:
        hidden_states = self.model(  # 通过模型主体获取隐藏状态
            input_ids,  # 输入ID
            positions,  # 位置
            forward_batch,  # 批次信息
            input_embeds,  # 输入嵌入
            pp_proxy_tensors=pp_proxy_tensors,  # 流水线代理张量
        )

        aux_hidden_states = None  # 辅助隐藏状态初始化为None
        if self.capture_aux_hidden_states:  # 如果需要捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包隐藏状态和辅助隐藏状态

        if self.pp_group.is_last_rank:  # 如果是最后一个rank
            return self.logits_processor(  # 通过logits处理器获取logits
                input_ids,  # 输入ID
                hidden_states,  # 隐藏状态
                self.lm_head,  # 语言模型头
                forward_batch,  # 批次信息
                aux_hidden_states,  # 辅助隐藏状态
            )
        else:  # 否则
            return hidden_states  # 直接返回隐藏状态

    @property
    def start_layer(self):  # 起始层属性
        return self.model.start_layer  # 返回模型起始层

    @property
    def end_layer(self):  # 结束层属性
        return self.model.end_layer  # 返回模型结束层

    def _get_default_weight_mapping(self):  # 获取默认权重名称映射
        """Generate default weight name mapping for GptOss safetensors."""  # 生成GptOss safetensors的默认权重名称映射
        weight_mapping = {}  # 权重映射字典

        # Map router weights to gate  # 将路由器权重映射到gate
        weight_mapping["embedding.weight"] = "model.embed_tokens.weight"  # 嵌入权重映射
        weight_mapping["unembedding.weight"] = "lm_head.weight"  # 反嵌入权重映射
        weight_mapping["norm.scale"] = "model.norm.weight"  # 归一化权重映射
        for layer_id in range(self.config.num_hidden_layers):  # 遍历所有层
            weight_mapping[f"block.{layer_id}.attn.q_proj.weight"] = (  # Q投影权重映射
                f"model.layers.{layer_id}.self_attn.q_proj.weight"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.attn.q_proj.bias"] = (  # Q投影偏置映射
                f"model.layers.{layer_id}.self_attn.q_proj.bias"  # 目标名称
            )

            weight_mapping[f"block.{layer_id}.attn.k_proj.weight"] = (  # K投影权重映射
                f"model.layers.{layer_id}.self_attn.k_proj.weight"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.attn.k_proj.bias"] = (  # K投影偏置映射
                f"model.layers.{layer_id}.self_attn.k_proj.bias"  # 目标名称
            )

            weight_mapping[f"block.{layer_id}.attn.v_proj.weight"] = (  # V投影权重映射
                f"model.layers.{layer_id}.self_attn.v_proj.weight"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.attn.v_proj.bias"] = (  # V投影偏置映射
                f"model.layers.{layer_id}.self_attn.v_proj.bias"  # 目标名称
            )

            weight_mapping[f"block.{layer_id}.attn.out.weight"] = (  # 输出投影权重映射
                f"model.layers.{layer_id}.self_attn.o_proj.weight"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.attn.out.bias"] = (  # 输出投影偏置映射
                f"model.layers.{layer_id}.self_attn.o_proj.bias"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.attn.sinks"] = (  # 注意力汇聚点映射
                f"model.layers.{layer_id}.self_attn.sinks"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.attn.norm.scale"] = (  # 注意力归一化权重映射
                f"model.layers.{layer_id}.input_layernorm.weight"  # 目标名称
            )

            weight_mapping[f"block.{layer_id}.mlp.gate.weight"] = (  # 路由器权重映射
                f"model.layers.{layer_id}.mlp.router.weight"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.mlp.gate.bias"] = (  # 路由器偏置映射
                f"model.layers.{layer_id}.mlp.router.bias"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.mlp.norm.scale"] = (  # MLP归一化权重映射
                f"model.layers.{layer_id}.post_attention_layernorm.weight"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.mlp.experts.gate_up_proj"] = (  # 专家gate_up投影映射
                f"model.layers.{layer_id}.mlp.experts.gate_up_proj"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.mlp.gate_up_proj_bias"] = (  # 专家gate_up投影偏置映射
                f"model.layers.{layer_id}.mlp.experts.gate_up_proj_bias"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.mlp.down_proj"] = (  # 专家down投影映射
                f"model.layers.{layer_id}.mlp.experts.mlp2_weight"  # 目标名称
            )
            weight_mapping[f"block.{layer_id}.mlp.down_proj_bias"] = (  # 专家down投影偏置映射
                f"model.layers.{layer_id}.mlp.experts.mlp2_bias"  # 目标名称
            )

        return weight_mapping  # 返回权重映射

    # TODO beautify code  # TODO: 美化代码
    def load_weights(  # 加载权重函数
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],  # 权重迭代器
        is_nextn: bool = False,  # 是否为nextn，默认False
        weight_name_mapping: dict = None,  # 权重名称映射，默认None
    ):
        quant_config_name = (  # 获取量化配置名称
            self.quant_config.get_name() if self.quant_config is not None else None  # 有量化配置则获取名称，否则为None
        )
        if quant_config_name != "mxfp4":  # 如果不是mxfp4量化
            self._load_normal_weights(  # 加载普通权重
                weights, is_nextn=is_nextn, weight_name_mapping=weight_name_mapping  # 传入权重和参数
            )
        else:  # 否则
            self._load_weights_mxfp4(  # 加载mxfp4权重
                weights, is_nextn=is_nextn, weight_name_mapping=weight_name_mapping  # 传入权重和参数
            )

    def _load_weights_mxfp4(self, weights, is_nextn, weight_name_mapping):  # 加载MXFP4权重
        mxfp4_weights = []  # MXFP4权重列表
        normal_weights = []  # 普通权重列表

        for name, weight in weights:  # 遍历所有权重
            if (  # 如果是专家权重且使用mxfp4量化
                ".experts" in name  # 名称包含".experts"
                and self.quant_config is not None  # 有量化配置
                and self.quant_config.get_name() == "mxfp4"  # 量化配置为mxfp4
            ):
                mxfp4_weights.append((name, weight))  # 添加到MXFP4权重列表
            else:  # 否则
                normal_weights.append((name, weight))  # 添加到普通权重列表

        mxfp4_loaded_params = self._load_mxfp4_experts_weights(mxfp4_weights)  # 加载MXFP4专家权重
        self._load_normal_weights(  # 加载普通权重
            normal_weights,  # 普通权重列表
            is_nextn=is_nextn,  # 是否nextn
            weight_name_mapping=weight_name_mapping,  # 权重名称映射
            other_loaded_param_names=mxfp4_loaded_params,  # 已加载的MXFP4参数名
        )

    def _load_mxfp4_experts_weights(self, weights):  # 加载MXFP4专家权重

        params_dict = dict(self.named_parameters())  # 获取参数字典
        loaded_params: set[str] = set()  # 已加载参数集合
        mxfp4_block = 32  # MXFP4块大小

        moe_tp_rank = get_moe_tensor_parallel_rank()  # MoE张量并行排名
        moe_tp_size = get_moe_tensor_parallel_world_size()  # MoE张量并行世界大小
        moe_ep_rank = get_moe_expert_parallel_rank()  # MoE专家并行排名
        moe_ep_size = get_moe_expert_parallel_world_size()  # MoE专家并行世界大小

        intermediate_size = self.config.intermediate_size  # 中间层大小
        assert (  # 断言中间层大小可被mxfp4块大小整除
            intermediate_size % mxfp4_block == 0
        ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"  # 中间层大小必须可被mxfp4块大小整除
        intermediate_size_block = intermediate_size // mxfp4_block  # 每块中间层大小

        per_rank_intermediate_size_block = math.ceil(  # 每个rank的中间层块数
            intermediate_size_block / moe_tp_size  # 向上取整
        )

        per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block  # 每个rank的中间层大小

        # Calculate common slicing bounds for current rank  # 计算当前rank的通用切片边界
        assert self.config.num_local_experts % moe_ep_size == 0  # 断言专家数可被EP大小整除
        moe_num_global_experts = self.config.num_local_experts  # 全局专家数
        moe_num_local_experts = self.config.num_local_experts // moe_ep_size  # 本地专家数

        moe_tp_rank_start = moe_tp_rank * per_rank_intermediate_size  # TP排名起始索引
        moe_tp_rank_end = min(  # TP排名结束索引
            (moe_tp_rank + 1) * per_rank_intermediate_size, intermediate_size  # 不超过中间层大小
        )

        moe_ep_rank_start = moe_ep_rank * moe_num_local_experts  # EP排名起始索引
        moe_ep_rank_end = (moe_ep_rank + 1) * moe_num_local_experts  # EP排名结束索引

        for name, weight in weights:  # 遍历所有MXFP4权重
            if _is_cuda:  # 如果是CUDA环境
                weight = weight.cuda()  # 将权重移到GPU

            if "gate_up_proj_blocks" in name:  # 如果是gate_up投影块权重
                # Handle MLP gate and up projection weights  # 处理MLP门控和上投影权重
                new_name = name.replace("gate_up_proj_blocks", "w13_weight")  # 替换名称

                # flat weight from (E, 2 * N, block_size, entry_per_block)  # 将权重从(E, 2*N, block_size, entry_per_block)展平
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous  # 到(E, 2*N, -1)，对于连续张量不应触发复制
                weight = weight.view(  # 重塑权重形状
                    moe_num_global_experts, 2 * intermediate_size, -1  # (专家数, 2*中间层大小, -1)
                ).contiguous()  # 确保连续

                narrow_weight = weight[  # 按TP和EP排名切片权重
                    moe_ep_rank_start:moe_ep_rank_end,  # EP维度切片
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,  # TP维度切片（乘2因为是gate和up）
                    ...,  # 其他维度保持不变
                ]

                param = params_dict[new_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 目标参数
                    narrow_weight,  # 切片后的权重
                    weight_name=new_name,  # 权重名称
                    shard_id=None,  # 分片ID为None
                    expert_id=None,  # 专家ID为None
                )
                loaded_params.add(new_name)  # 添加到已加载参数集合

            elif "down_proj_blocks" in name:  # 如果是down投影块权重
                # Handle MLP down projection weights  # 处理MLP下投影权重
                new_name = name.replace("down_proj_blocks", "w2_weight")  # 替换名称
                # same flatten here, but since 2 mx4 value are packed in 1  # 同样展平，但因为2个mx4值打包在1个
                # uint8, divide by 2  # uint8中，所以除以2
                weight = weight.view(  # 重塑权重形状
                    moe_num_global_experts, -1, intermediate_size // 2  # (专家数, -1, 中间层大小/2)
                ).contiguous()  # 确保连续
                narrow_weight = weight[  # 按EP和TP排名切片权重
                    moe_ep_rank_start:moe_ep_rank_end,  # EP维度切片
                    ...,  # 中间维度保持不变
                    moe_tp_rank_start // 2 : moe_tp_rank_end // 2,  # TP维度切片（除以2因为打包）
                ]

                param = params_dict[new_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 目标参数
                    narrow_weight,  # 切片后的权重
                    weight_name=new_name,  # 权重名称
                    shard_id=None,  # 分片ID为None
                    expert_id=None,  # 专家ID为None
                )
                loaded_params.add(new_name)  # 添加到已加载参数集合

            elif "gate_up_proj_scales" in name:  # 如果是gate_up投影缩放因子
                # Handle MLP gate and up projection weights scale  # 处理MLP门控和上投影权重缩放因子
                new_name = name.replace("gate_up_proj_scales", "w13_weight_scale")  # 替换名称
                narrow_weight = weight[  # 按EP和TP排名切片权重
                    moe_ep_rank_start:moe_ep_rank_end,  # EP维度切片
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,  # TP维度切片
                    ...,  # 其他维度保持不变
                ]

                param = params_dict[new_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 目标参数
                    narrow_weight,  # 切片后的权重
                    weight_name=new_name,  # 权重名称
                    shard_id=None,  # 分片ID为None
                    expert_id=None,  # 专家ID为None
                )
                loaded_params.add(new_name)  # 添加到已加载参数集合

            elif "down_proj_scales" in name:  # 如果是down投影缩放因子
                # Handle MLP down projection weights  # 处理MLP下投影权重缩放因子
                new_name = name.replace("down_proj_scales", "w2_weight_scale")  # 替换名称
                narrow_weight = weight[  # 按EP和TP排名切片权重
                    moe_ep_rank_start:moe_ep_rank_end,  # EP维度切片
                    ...,  # 中间维度保持不变
                    moe_tp_rank_start // mxfp4_block : moe_tp_rank_end // mxfp4_block,  # TP维度切片（除以mxfp4块大小）
                ]

                param = params_dict[new_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 目标参数
                    narrow_weight,  # 切片后的权重
                    weight_name=new_name,  # 权重名称
                    shard_id=None,  # 分片ID为None
                    expert_id=None,  # 专家ID为None
                )
                loaded_params.add(new_name)  # 添加到已加载参数集合
            elif "gate_up_proj_bias" in name:  # 如果是gate_up投影偏置
                # Handle MLP gate and up projection biases  # 处理MLP门控和上投影偏置
                new_name = name.replace("gate_up_proj_bias", "w13_weight_bias")  # 替换名称

                narrow_weight = weight[  # 按EP和TP排名切片权重
                    moe_ep_rank_start:moe_ep_rank_end,  # EP维度切片
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,  # TP维度切片
                ]

                param = params_dict[new_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 目标参数
                    narrow_weight,  # 切片后的权重
                    weight_name=new_name,  # 权重名称
                    shard_id=None,  # 分片ID为None
                    expert_id=None,  # 专家ID为None
                )
                loaded_params.add(new_name)  # 添加到已加载参数集合

            elif "down_proj_bias" in name:  # 如果是down投影偏置
                narrow_weight = weight[moe_ep_rank_start:moe_ep_rank_end, ...]  # 按EP排名切片偏置
                if moe_tp_rank != 0:  # 如果不是第一个TP rank
                    narrow_weight = torch.zeros_like(narrow_weight)  # 用零填充偏置（只在一个rank上加载）

                # Handle MLP down projection bias  # 处理MLP下投影偏置
                new_name = name.replace("down_proj_bias", "w2_weight_bias")  # 替换名称
                param = params_dict[new_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(  # 加载权重
                    param,  # 目标参数
                    narrow_weight,  # 切片后的权重
                    weight_name=new_name,  # 权重名称
                    shard_id=None,  # 分片ID为None
                    expert_id=None,  # 专家ID为None
                )
                loaded_params.add(new_name)  # 添加到已加载参数集合

        return loaded_params  # 返回已加载参数集合

    def _load_normal_weights(  # 加载普通权重函数
        self,
        weights,  # 权重列表
        is_nextn: bool,  # 是否为nextn
        weight_name_mapping: dict,  # 权重名称映射
        other_loaded_param_names=[],  # 其他已加载参数名列表，默认空
    ):
        tp_rank = get_tensor_model_parallel_rank()  # 获取张量并行排名
        if is_nextn:  # 如果是nextn
            logging.warning(  # 输出警告日志
                "Loading weights for nextn is currently not supported in GptOssForCausalLM. "  # GptOssForCausalLM目前不支持加载nextn权重
            )
            return  # 直接返回
        weights = _canonicalize_weights(self.config, weights)  # 规范化权重
        weights = sorted(weights, key=lambda x: x[0])  # Sort by name for consistency  # 按名称排序以保持一致性

        new_weights = []  # 新权重列表
        for name, p in weights:  # 遍历所有权重
            if "qkv.weight" in name:  # 如果是QKV合并权重
                q_proj, k_proj, v_proj = p.split(  # 按头数分割为Q、K、V
                    [
                        self.config.num_attention_heads * self.config.head_dim,  # Q维度
                        self.config.num_key_value_heads * self.config.head_dim,  # K维度
                        self.config.num_key_value_heads * self.config.head_dim,  # V维度
                    ],
                    dim=0,  # 在第0维分割
                )
                new_weights.append(  # 添加Q权重
                    (f"{name.replace('qkv.weight', 'q_proj.weight')}", q_proj)  # 重命名为q_proj
                )
                new_weights.append(  # 添加K权重
                    (f"{name.replace('qkv.weight', 'k_proj.weight')}", k_proj)  # 重命名为k_proj
                )
                new_weights.append(  # 添加V权重
                    (f"{name.replace('qkv.weight', 'v_proj.weight')}", v_proj)  # 重命名为v_proj
                )
            elif "qkv.bias" in name:  # 如果是QKV合并偏置
                q_bias, k_bias, v_bias = p.split(  # 按头数分割为Q、K、V偏置
                    [
                        self.config.num_attention_heads * self.config.head_dim,  # Q偏置维度
                        self.config.num_key_value_heads * self.config.head_dim,  # K偏置维度
                        self.config.num_key_value_heads * self.config.head_dim,  # V偏置维度
                    ],
                    dim=0,  # 在第0维分割
                )
                new_weights.append(  # 添加Q偏置
                    (f"{name.replace('qkv.bias', 'q_proj.bias')}", q_bias)  # 重命名为q_proj
                )
                new_weights.append(  # 添加K偏置
                    (f"{name.replace('qkv.bias', 'k_proj.bias')}", k_bias)  # 重命名为k_proj
                )
                new_weights.append(  # 添加V偏置
                    (f"{name.replace('qkv.bias', 'v_proj.bias')}", v_bias)  # 重命名为v_proj
                )
            else:  # 否则直接添加
                new_weights.append((name, p))  # 保持原名添加
        weights = new_weights  # 使用新权重列表

        # Use provided weight name mapping if available, otherwise use default  # 如果提供了权重名称映射则使用，否则使用默认
        if weight_name_mapping is None:  # 如果没有提供权重名称映射
            weight_name_mapping = self._get_default_weight_mapping()  # 使用默认映射
        else:  # 否则
            # Merge with default mapping  # 与默认映射合并
            default_mapping = self._get_default_weight_mapping()  # 获取默认映射
            default_mapping.update(weight_name_mapping)  # 用提供的映射更新默认映射
            weight_name_mapping = default_mapping  # 使用合并后的映射

        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q投影映射
            ("qkv_proj", "k_proj", "k"),  # K投影映射
            ("qkv_proj", "v_proj", "v"),  # V投影映射
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping_fused(  # 创建专家参数映射
            ckpt_gate_up_proj_name="gate_up_proj",  # 检查点gate_up投影名
            ckpt_down_proj_name="down_proj",  # 检查点down投影名
            ckpt_gate_up_proj_bias_name="gate_up_proj_bias",  # 检查点gate_up投影偏置名
            ckpt_down_proj_bias_name="down_proj_bias",  # 检查点down投影偏置名
        )

        params_dict = dict(self.named_parameters())  # 获取参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            loaded_weight = _WeightCreator.maybe_materialize(loaded_weight)  # 可能物化懒加载权重

            # Apply weight name mapping if provided  # 如果提供了权重名称映射则应用
            if weight_name_mapping and name in weight_name_mapping:  # 如果有权重映射且名称在映射中
                name = weight_name_mapping[name]  # 应用映射

            layer_id = get_layer_id(name)  # 获取层ID
            if (  # 如果层ID不在当前rank负责的范围内
                layer_id is not None  # 层ID不为None
                and hasattr(self.model, "start_layer")  # 模型有start_layer属性
                and (  # 并且
                    layer_id < self.model.start_layer  # 层ID小于起始层
                    or layer_id >= self.model.end_layer  # 或层ID大于等于结束层
                )
            ):
                continue  # 跳过

            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码的逆频率
                continue  # 继续
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在名称中
                    continue  # 继续
                if "mlp.experts" in name:  # 如果是专家层参数（由专家映射处理）
                    continue  # 继续

                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                if name.endswith(".bias") and name not in params_dict:  # 跳过不在字典中的偏置
                    continue  # 继续
                if name not in params_dict:  # 跳过不在字典中的参数
                    continue  # 继续

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 按分片加载权重
                break  # 跳出内层循环
            else:  # 如果没有匹配的堆叠参数映射
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果权重名不在名称中
                        continue  # 继续
                    name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                    if name not in params_dict:  # 如果不在参数字典中
                        continue  # 继续
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    if "bias" not in name:  # 如果不是偏置
                        loaded_weight = loaded_weight.transpose(-2, -1)  # 转置权重
                    if "w2_weight_bias" in name and get_moe_tensor_parallel_rank() != 0:  # 如果是down投影偏置且不是第一个TP rank
                        loaded_weight = loaded_weight.zero_()  # 将偏置置零

                    weight_loader(  # 加载权重
                        param,  # 目标参数
                        loaded_weight,  # 加载的权重
                        name,  # 权重名称
                        shard_id=shard_id,  # 分片ID
                    )
                    break  # 跳出内层循环
                else:  # 如果没有匹配专家参数映射
                    if name.endswith(".bias") and name not in params_dict:  # 跳过不在字典中的偏置
                        continue  # 继续
                    if name not in params_dict:  # 跳过不在字典中的参数
                        continue  # 继续
                    if name in params_dict.keys():  # 如果名称在参数字典中
                        param = params_dict[name]  # 获取参数
                        if "sinks" in name:  # 如果是注意力汇聚点参数
                            start = get_attention_tp_rank() * param.numel()  # 计算起始索引
                            tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
                            full_shard_size = param.numel() * tp_size  # 完整分片大小
                            # This handles TP padding: if the checkpoint dim is not divisible by tp_size,  # 处理TP填充：如果检查点维度不能被tp_size整除
                            # the last TP shard extends beyond `loaded_weight`, pad with zeros before slicing.  # 最后一个TP分片超出loaded_weight，在切片前用零填充
                            if (  # 如果需要填充
                                _is_cpu  # 是CPU环境
                                and full_shard_size > loaded_weight.size(0)  # 完整分片大于权重大小
                                and start + param.numel() >= loaded_weight.size(0)  # 当前分片超出权重范围
                            ):
                                pad_size = start + param.numel() - loaded_weight.size(0)  # 计算填充大小
                                pad_tensor = torch.zeros(pad_size).to(  # 创建填充张量
                                    loaded_weight.dtype  # 使用相同数据类型
                                )
                                loaded_weight = torch.cat(  # 拼接权重和填充
                                    [loaded_weight, pad_tensor], dim=0  # 在第0维拼接
                                ).to(loaded_weight.dtype)  # 转换数据类型
                            param.data.copy_(  # 复制数据到参数
                                loaded_weight[start : start + param.numel()]  # 切片对应分片
                            )
                        else:  # 否则普通参数
                            weight_loader = getattr(  # 获取权重加载器
                                param, "weight_loader", default_weight_loader  # 默认使用默认权重加载器
                            )
                            weight_loader(param, loaded_weight)  # 加载权重
                    else:  # 否则
                        logger.warning(f"Parameter {name} not found in params_dict")  # 记录参数未找到警告

    def get_embed_and_head(self):  # 获取嵌入层和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回词嵌入权重和语言模型头权重

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.model.embed_tokens  # 返回词嵌入层

    def set_embed_and_head(self, embed, head):  # 设置嵌入层和语言模型头权重
        del self.model.embed_tokens.weight  # 删除旧词嵌入权重
        del self.lm_head.weight  # 删除旧语言模型头权重
        self.model.embed_tokens.weight = embed  # 设置新词嵌入权重
        self.lm_head.weight = head  # 设置新语言模型头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):  # 设置EAGLE3需要捕获隐藏状态的层
        if not self.pp_group.is_last_rank:  # 如果不是最后一个rank
            return  # 直接返回

        if layer_ids is None:  # 如果没有指定层ID
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            num_layers = self.config.num_hidden_layers  # 总层数
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]  # 默认捕获第2层、中间层和倒数第3层
        else:  # 否则
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            # we plus 1 here because in sglang, for the ith layer, it takes the output  # 这里加1是因为在sglang中，第i层取
            # of the (i-1)th layer as aux hidden state  # 第(i-1)层的输出作为辅助隐藏状态
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 层ID加1

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):  # 设置DFLASH需要捕获隐藏状态的层
        if not self.pp_group.is_last_rank:  # 如果不是最后一个rank
            return  # 直接返回

        if layer_ids is None:  # 如果没有指定层ID
            raise ValueError(  # 抛出值错误
                "DFLASH requires explicit layer_ids for aux hidden capture."  # DFLASH需要显式指定层ID
            )

        self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
        self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 层ID加1

    @classmethod
    def get_model_config_for_expert_location(cls, config):  # 获取专家位置的模型配置
        return ModelConfigForExpertLocation(  # 返回专家位置模型配置
            num_layers=config.num_hidden_layers,  # 层数
            num_logical_experts=config.num_local_experts,  # 逻辑专家数
            num_groups=None,  # 分组数为None
        )

    def get_attention_sliding_window_size(self):  # 获取注意力滑动窗口大小
        return get_attention_sliding_window_size(self.config)  # 返回配置中的滑动窗口大小


def _canonicalize_weights(config, weights_in: Iterable[Tuple[str, torch.Tensor]]):  # 规范化权重函数
    weights_out_dict = dict(weights_in)  # 将输入权重转换为字典

    for layer_id in range(config.num_hidden_layers):  # 遍历所有层
        for name_chunk in ["mlp1_weight", "mlp2_weight"]:  # 遍历MLP权重名称
            name_prefix = f"block.{layer_id}.mlp.{name_chunk}"  # 构建名称前缀
            w_blocks = weights_out_dict.pop(f"{name_prefix}.blocks", None)  # 弹出块权重
            w_scales = weights_out_dict.pop(f"{name_prefix}.scales", None)  # 弹出缩放权重
            if w_blocks is not None:  # 如果有块权重
                weights_out_dict[name_prefix] = _WeightCreator(  # 创建懒加载权重
                    partial(  # 使用偏函数
                        _dequant_mlp_weight,  # 反量化MLP权重函数
                        debug_name=name_prefix,  # 调试名称
                        w_blocks=w_blocks,  # 块权重
                        w_scales=w_scales,  # 缩放权重
                    )
                )

    return list(weights_out_dict.items())  # 返回权重列表


def _dequant_mlp_weight(debug_name, w_blocks, w_scales):  # 反量化MLP权重函数
    if get_tensor_model_parallel_rank() == 0:  # 如果是第一个TP rank
        logger.info(f"Dequantize {debug_name} start")  # 记录反量化开始

    original_device = w_blocks.device  # 记录原始设备

    w_blocks = w_blocks.cuda()  # 将块权重移到GPU
    w_scales = w_scales.cuda()  # 将缩放权重移到GPU

    w_bf16 = dequant_mxfp4(w_block=w_blocks, w_scale=w_scales, out_dtype=torch.bfloat16)  # 反量化MXFP4权重为BF16
    w_bf16 = w_bf16.transpose(-2, -1).contiguous()  # 转置并确保连续

    if get_tensor_model_parallel_rank() == 0:  # 如果是第一个TP rank
        logger.info(  # 记录反量化完成
            f"Dequantize {debug_name} end {w_blocks.shape=} {w_scales.shape=} {w_bf16.shape=}"  # 包含形状信息
        )

    return w_bf16.to(original_device)  # 将权重移回原始设备


class _WeightCreator:  # 懒加载权重创建器类
    def __init__(self, fn):  # 初始化函数
        self._fn = fn  # 保存懒加载函数

    @staticmethod
    def maybe_materialize(obj):  # 可能物化函数
        if isinstance(obj, _WeightCreator):  # 如果是权重创建器对象
            output = obj._fn()  # 执行懒加载函数
            obj._fn = None  # 清除函数引用
            return output  # 返回物化结果

        return obj  # 否则直接返回对象


EntryClass = GptOssForCausalLM  # 入口类为GptOssForCausalLM
