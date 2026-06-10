# Transformers模型后端实现
# 本文件实现了基于HuggingFace Transformers的通用模型后端，支持自动模块替换（Linear->TP、RMSNorm->融合、MoE覆盖）。
# 包含因果语言模型、嵌入模型、序列分类模型及其MoE和多模态变体。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
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
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/transformers
"""Wrapper around `transformers` models."""  # HuggingFace Transformers模型的包装器

import inspect  # 导入inspect模块，用于检查函数签名
import logging  # 导入日志模块
import re  # 导入正则表达式模块
from collections.abc import Iterable, Mapping  # 导入集合抽象基类
from contextlib import contextmanager  # 导入上下文管理器
from typing import List, Literal, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
import transformers  # 导入transformers库
from torch import nn  # 导入神经网络模块
from transformers import AutoModel, PretrainedConfig, PreTrainedModel  # 导入transformers模型类
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # 导入动态模块工具
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # 导入注意力函数映射

from sglang.srt.distributed import (  # 导入分布式通信函数
    divide,  # 除法工具
    get_moe_expert_parallel_world_size,  # 获取MoE EP世界大小
    get_pp_group,  # 获取PP组
    get_pp_indices,  # 获取PP索引
    get_tensor_model_parallel_world_size,  # 获取TP世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置配置
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm  # 导入归一化层
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE
from sglang.srt.layers.moe.topk import StandardTopKOutput  # 导入标准Top-K输出
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert  # 导入MoE权重过滤
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType  # 导入池化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.utils import PPMissingLayer  # 导入PP缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行LM头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态填充模式
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs  # 导入多模态输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper  # 导入权重加载工具
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import get_device  # 导入设备获取工具
from sglang.srt.utils.common import direct_register_custom_op  # 导入自定义算子注册
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config  # 导入HF文本配置


def can_enable_torch_compile(config: PretrainedConfig) -> bool:
    """Check whether the model config is compatible with torch.compile.
    
    Dynamic rope scaling triggers data-dependent control flow that prevents
    capturing a single computation graph, so we disable compilation for it.
    """  # 检查模型配置是否与torch.compile兼容
    text_config = getattr(config, "text_config", config)  # 获取文本配置
    rope_scaling = getattr(text_config, "rope_scaling", None)  # 获取RoPE缩放
    if isinstance(rope_scaling, dict):  # 如果是字典
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))  # 获取类型
        if rope_type == "dynamic":  # 动态类型不支持编译
            return False
    rope_params = getattr(text_config, "rope_parameters", None)  # 获取RoPE参数
    if isinstance(rope_params, dict):  # 如果是字典
        if isinstance(next(iter(rope_params.values()), None), dict):  # 值也是字典
            return not any(  # 检查是否有动态类型
                rp.get("rope_type") == "dynamic" for rp in rope_params.values()
            )
        if rope_params.get("rope_type") == "dynamic":  # 动态类型不支持
            return False
    return True  # 其他情况支持编译


logger = logging.getLogger(__name__)  # 获取日志记录器

_TRANSFORMERS_MOE_LAYERS: dict[str, "TransformersFusedMoE"] = {}  # MoE层全局注册表


def maybe_prefix(prefix: str, name: str) -> str:
    """如果前缀非空则在名称前添加前缀"""
    return name if not prefix else f"{prefix}.{name}"  # 添加前缀


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    """记录模块替换日志"""
    logger.debug("%s: %s -> %s", name, old_module, new_module)  # 调试级别日志


def _getattr_first(obj, names, default=None):
    """Return the first existing attribute from *names*, else *default*."""  # 从名称列表中返回第一个存在的属性
    for name in names:  # 遍历名称列表
        value = getattr(obj, name, None)  # 尝试获取属性
        if value is not None:  # 如果存在
            return value  # 返回值
    return default  # 全部不存在返回默认值


def _resolve_attention_backend_model_cls(config: PretrainedConfig):
    """解析注意力后端模型类"""
    model_cls = getattr(transformers, getattr(config, "architectures", [""])[0], None)  # 从transformers获取
    if model_cls is not None:  # 如果找到
        return model_cls  # 返回模型类

    auto_map = getattr(config, "auto_map", {}) or {}  # 获取auto_map
    for key in ("AutoModel", "AutoModelForCausalLM"):  # 遍历自动映射键
        if key not in auto_map:  # 键不存在
            continue  # 跳过
        try:
            return get_class_from_dynamic_module(  # 从动态模块获取类
                auto_map[key],
                getattr(config, "_name_or_path", ""),  # 模型路径
            )
        except Exception as e:  # 获取失败
            logger.warning(  # 记录警告
                "Failed to load dynamic module from auto_map[%s]: %s.",
                key,
                e,
            )
    return None  # 返回None


def _encoder_accepts_feature_kwarg(encoder, feature_kwarg: str) -> bool:
    """检查编码器是否接受特征关键字参数"""
    try:
        sig = inspect.signature(encoder)  # 获取函数签名
    except (TypeError, ValueError):  # 获取签名失败
        return False  # 返回False

    if feature_kwarg in sig.parameters:  # 参数名在签名中
        return True  # 接受

    has_var_keyword = any(  # 检查是否有可变关键字参数
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_keyword:  # 没有可变关键字
        return False  # 不接受

    required_positional_params = [  # 必需位置参数列表
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    return len(required_positional_params) == 0  # 没有必需位置参数时接受


@contextmanager
def _init_on_device_without_buffers(device: torch.device):
    """Initialize model parameters on *device* while leaving buffers on CPU.
    Adapted from ``accelerate``."""  # 在指定设备上初始化参数，但buffer留在CPU上
    old_register_parameter = nn.Module.register_parameter  # 保存原始注册参数方法

    def register_empty_parameter(module, name, param):  # 注册空参数
        old_register_parameter(module, name, param)  # 调用原始方法
        if param is not None:  # 如果参数非空
            param_cls = type(module._parameters[name])  # 获取参数类
            kwargs = module._parameters[name].__dict__  # 获取参数字典
            kwargs["requires_grad"] = param.requires_grad  # 设置梯度需求
            module._parameters[name] = param_cls(  # 在目标设备上创建参数
                module._parameters[name].to(device), **kwargs
            )

    try:
        nn.Module.register_parameter = register_empty_parameter  # 替换注册方法
        yield  # 执行上下文
    finally:
        nn.Module.register_parameter = old_register_parameter  # 恢复原始方法


Style = Literal["colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"]  # TP风格类型


def replace_linear_class(
    linear: nn.Linear,  # 原始线性层
    style: Style = "replicate",  # TP风格
    quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    *,  # 以下为关键字参数
    prefix: str = "",  # 参数前缀
) -> Union[ColumnParallelLinear, RowParallelLinear, ReplicatedLinear]:
    """将PyTorch线性层替换为SGLang并行线性层"""
    if not isinstance(style, str):  # 检查风格类型
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    sglang_linear_cls, linear_kwargs = {  # 根据风格选择类和参数
        "colwise": (ColumnParallelLinear, {}),  # 列并行
        "colwise_rep": (ColumnParallelLinear, {"gather_output": True}),  # 列并行+收集输出
        "rowwise": (RowParallelLinear, {}),  # 行并行
        "rowwise_rep": (RowParallelLinear, {"input_is_parallel": False}),  # 行并行+非并行输入
        "replicate": (ReplicatedLinear, {}),  # 复制
    }.get(style, (ReplicatedLinear, {}))  # 默认复制

    class HFCompatibleLinear(sglang_linear_cls):  # HF兼容线性层
        @property
        def parent_cls(self) -> type:  # 获取父类
            return sglang_linear_cls

        def forward(self, input: torch.Tensor) -> torch.Tensor:  # 前向传播
            return super().forward(input)[0]  # 只返回第一个输出

    return HFCompatibleLinear(  # 创建替换后的线性层
        input_size=linear.in_features,  # 输入特征数
        output_size=linear.out_features,  # 输出特征数
        bias=linear.bias is not None,  # 是否有偏置
        quant_config=quant_config,  # 量化配置
        prefix=prefix,  # 参数前缀
        **linear_kwargs,  # 额外参数
    )


def _normalize_tp_style(style: str) -> Style:
    """归一化TP风格名称"""
    style = style.lower().replace("-", "_")  # 转小写并替换连字符
    style = {  # 风格名称映射
        "colwiseparallel": "colwise",
        "packed_colwise": "colwise",
        "local_colwise": "colwise",
        "rowwiseparallel": "rowwise",
        "packed_rowwise": "rowwise",
        "local_rowwise": "rowwise",
        "local_packed_rowwise": "rowwise",
        "isolated": "replicate",
        "local": "replicate",
        "replicated_with_grad_allreduce": "replicate",
        "moe_tp_experts": "replicate",
    }.get(style, style)  # 默认保持原样
    if style not in {"colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"}:  # 验证
        raise ValueError(f"Unsupported TP style '{style}' for Transformers backend.")
    return style  # 返回归一化后的风格


def replace_rms_norm_class(rms_norm: nn.Module, hidden_size: int) -> nn.Module:
    """将HF RMSNorm替换为SGLang融合RMSNorm"""
    eps = _getattr_first(rms_norm, ("eps", "variance_epsilon"), 1e-6)  # 获取eps
    kwargs = {"hidden_size": hidden_size, "eps": eps}  # 构建参数
    weight_meta = getattr(rms_norm, "weight", None)  # 获取权重元数据
    if weight_meta is not None:  # 如果有权重
        kwargs["hidden_size"] = weight_meta.size(0)  # 使用权重大小

    try:
        with torch.device("cpu"):  # 在CPU上创建测试实例
            weight_test = getattr(rms_norm.__class__(1), "weight", None)  # 测试权重
    except Exception:
        weight_test = None  # 失败设为None
    is_gemma = weight_test is not None and torch.all(weight_test == 0)  # 检测Gemma RMSNorm

    if is_gemma:  # 如果是Gemma风格
        base_cls = GemmaRMSNorm  # 使用GemmaRMSNorm
        norm = base_cls(  # 创建归一化层
            **{k: v for k, v in kwargs.items() if k in ("hidden_size", "eps")}
        )
    else:  # 普通RMSNorm
        kwargs["has_weight"] = getattr(rms_norm, "with_scale", True)  # 是否有权重
        if weight_meta is not None:  # 如果有权重元数据
            kwargs["weight_dtype"] = weight_meta.dtype  # 设置权重数据类型
        else:
            kwargs["has_weight"] = False  # 无权重
        kwargs["cast_x_before_out_mul"] = (  # 匹配HF fp16语义
            True  # match HF fp16-weight-multiply semantics
        )
        base_cls = RMSNorm  # 使用RMSNorm
        norm = base_cls(**kwargs)  # 创建归一化层

    # Wrap to handle 3D inputs from Transformers backbone (batch dim)
    class HFCompatibleRMSNorm(norm.__class__):  # HF兼容RMSNorm
        def forward(self, x, *args, **kwargs):  # 前向传播
            orig_shape = x.shape  # 保存原始形状
            if x.ndim > 2:  # 如果是3D以上输入
                x = x.reshape(-1, x.shape[-1]).contiguous()  # 展平为2D
            result = super().forward(x, *args, **kwargs)  # 调用父类前向
            if isinstance(result, tuple):  # 如果结果是元组
                return tuple(  # 对每个元素恢复形状
                    (
                        r.reshape(orig_shape)
                        if torch.is_tensor(r) and r.shape != orig_shape
                        else r
                    )
                    for r in result
                )
            if torch.is_tensor(result) and result.shape != orig_shape:  # 单个张量
                return result.reshape(orig_shape)  # 恢复形状
            return result  # 返回结果

    norm.__class__ = HFCompatibleRMSNorm  # 替换类
    return norm  # 返回归一化层


def sglang_flash_attention_forward(
    module: torch.nn.Module,  # 注意力模块
    query: torch.Tensor,  # 查询张量
    key: torch.Tensor,  # 键张量
    value: torch.Tensor,  # 值张量
    attention_mask: torch.Tensor,  # 注意力掩码
    scaling: float = None,  # 缩放因子
    attention_instances: Optional[Mapping[int, RadixAttention]] = None,  # 注意力实例映射
    forward_batch: Optional[ForwardBatch] = None,  # 前向批次
    **kwargs,
):
    """SGLang Flash注意力前向传播，作为HF注意力函数注册"""
    self_attn: RadixAttention = attention_instances[module.layer_idx]  # 获取对应层的注意力实例
    if scaling is not None:  # 如果有缩放因子
        self_attn.scaling = float(scaling)  # 设置缩放
    hidden = query.shape[-2]  # 序列长度
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))  # 转置
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))  # 重塑
    return self_attn.forward(query, key, value, forward_batch=forward_batch), None  # 调用注意力


ALL_ATTENTION_FUNCTIONS["sglang"] = sglang_flash_attention_forward  # 注册SGLang注意力函数


class TransformersFusedMoE(nn.Module):
    """FusedMoE wrapper for the Transformers modeling backend.

    Wraps SGLang's native MoE implementation and exposes the
    ``(hidden_states, topk_ids, topk_weights)`` signature expected by
    Transformers' ``experts.forward()``.  A registered custom op
    (``torch.ops.sglang.transformers_moe_forward``) is used so that
    ``torch.compile`` can properly graph-break around the MoE kernel.
    """  # Transformers后端的融合MoE包装器

    def __init__(
        self,
        *,  # 强制关键字参数
        num_experts: int,  # 专家数
        top_k: int,  # Top-K值
        hidden_size: int,  # 隐藏大小
        intermediate_size: int,  # 中间层大小
        layer_id: int,  # 层ID
        reduce_results: bool,  # 是否归约结果
        quant_config: Optional[QuantizationConfig],  # 量化配置
        prefix: str,  # 参数前缀
        activation: str,  # 激活函数
        with_bias: bool,  # 是否有偏置
        expert_mapping: list,  # 专家映射
    ) -> None:
        super().__init__()  # 调用父类初始化
        num_redundant = get_global_server_args().ep_num_redundant_experts  # 冗余专家数
        experts_cls = get_moe_impl_class(quant_config)  # 获取MoE实现类
        self.experts = experts_cls(  # 创建专家层
            num_experts=num_experts + num_redundant,  # 加上冗余专家
            top_k=top_k,  # Top-K值
            layer_id=layer_id,  # 层ID
            hidden_size=hidden_size,  # 隐藏大小
            intermediate_size=intermediate_size,  # 中间层大小
            reduce_results=reduce_results,  # 是否归约
            quant_config=quant_config,  # 量化配置
            activation=activation,  # 激活函数
            with_bias=with_bias,  # 偏置
            prefix=prefix,  # 前缀
        )
        self.layer_name = prefix  # 层名
        self.num_experts = num_experts  # 专家数
        self.top_k = top_k  # Top-K值
        self._expert_mapping = expert_mapping  # 专家映射
        _TRANSFORMERS_MOE_LAYERS[prefix] = self  # 注册到全局表

    @property
    def tp_size(self) -> int:
        """获取MoE张量并行大小"""
        return getattr(self.experts, "moe_tp_size", 1)  # 返回TP大小

    @property
    def ep_size(self) -> int:
        """获取MoE专家并行大小"""
        return getattr(self.experts, "moe_ep_size", 1)  # 返回EP大小

    def maybe_all_reduce_tensor_model_parallel(
        self, output: torch.Tensor
    ) -> torch.Tensor:
        """如果TP大小>1，执行张量并行全归约"""
        if self.tp_size > 1:  # 需要全归约
            return tensor_model_parallel_all_reduce(output)  # 全归约
        return output  # 不需要全归约

    def get_expert_weights(self):
        """获取专家权重"""
        return getattr(self.experts, "get_expert_weights", lambda: None)()  # 调用专家方法

    def get_moe_weights(self) -> list[torch.Tensor]:
        """获取MoE权重列表"""
        num_local = getattr(self.experts, "num_local_experts", self.num_experts)  # 本地专家数
        return [
            x.data  # 参数数据
            for name, x in self.experts.named_parameters()  # 遍历参数
            if name not in ("correction_bias",)  # 排除校正偏置
            and filter_moe_weight_param_global_expert(name, x, num_local)  # 过滤全局专家
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        topk_ids: torch.Tensor,  # Top-K ID
        topk_weights: torch.Tensor,  # Top-K权重
        **kwargs,
    ) -> torch.Tensor:
        """MoE前向传播，通过自定义算子以支持torch.compile"""
        topk_ids = topk_ids.to(torch.int32)  # 转换ID类型
        topk_weights = topk_weights.to(torch.float32)  # 转换权重类型
        if hidden_states.is_cuda:  # 如果是CUDA
            return torch.ops.sglang.transformers_moe_forward(  # 使用自定义算子
                hidden_states,
                topk_ids,
                topk_weights,
                self.layer_name,
            )
        return _transformers_moe_forward(  # 非CUDA使用Python实现
            hidden_states,
            topk_ids,
            topk_weights,
            self.layer_name,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载MoE权重"""
        loaded: set[str] = set()  # 已加载参数集合
        param_dict = dict(self.named_parameters())  # 参数字典
        for name, loaded_weight in weights:  # 遍历权重
            matched = False  # 是否匹配
            for param_name, weight_name, expert_id, shard_id in self._expert_mapping:  # 遍历专家映射
                if weight_name not in name:  # 不匹配
                    continue  # 跳过
                mapped_name = name.replace(weight_name, param_name)  # 映射名称
                param = param_dict.get(mapped_name)  # 获取参数
                if param is None:  # 参数不存在
                    continue  # 跳过
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 加载器
                try:
                    weight_loader(  # 尝试带专家ID加载
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                except TypeError:  # 不支持专家ID
                    weight_loader(param, loaded_weight)  # 普通加载
                loaded.add(name)  # 添加到已加载集合
                matched = True  # 标记匹配
                break  # 跳出内循环
            if not matched:  # 没有匹配专家映射
                direct_name = name if name in param_dict else f"experts.{name}"  # 尝试直接名称
                if direct_name in param_dict:  # 直接名称存在
                    param = param_dict[direct_name]  # 获取参数
                    weight_loader = getattr(  # 获取加载器
                        param, "weight_loader", default_weight_loader
                    )
                    try:
                        weight_loader(param, loaded_weight)  # 尝试加载
                    except TypeError:
                        default_weight_loader(param, loaded_weight)  # 使用默认加载
                    loaded.add(name)  # 添加到已加载集合
                else:
                    logger.warning(  # 记录警告
                        "MoE weight '%s' in layer '%s' could not be matched to any "
                        "parameter and will be skipped.",
                        name,
                        self.layer_name,
                    )
        return loaded  # 返回已加载集合


def _transformers_moe_forward(
    hidden_states: torch.Tensor,  # 隐藏状态
    topk_ids: torch.Tensor,  # Top-K ID
    topk_weights: torch.Tensor,  # Top-K权重
    layer_name: str,  # 层名
) -> torch.Tensor:
    """Transformers MoE前向传播的Python实现"""
    self = _TRANSFORMERS_MOE_LAYERS[layer_name]  # 获取MoE层实例
    # Record expert distribution for EPLB
    from sglang.srt.eplb.expert_distribution import (  # 导入专家分布记录器
        get_global_expert_distribution_recorder,
    )

    recorder = get_global_expert_distribution_recorder()  # 获取记录器
    with recorder.with_current_layer(self.experts.layer_id):  # 设置当前层
        recorder.on_select_experts(topk_ids=topk_ids)  # 记录专家选择
    topk_output = StandardTopKOutput(  # 创建标准Top-K输出
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=topk_weights,
    )
    return self.experts(hidden_states.clone(), topk_output)  # 通过专家层


def _transformers_moe_forward_fake(
    hidden_states: torch.Tensor,  # 隐藏状态
    topk_ids: torch.Tensor,  # Top-K ID
    topk_weights: torch.Tensor,  # Top-K权重
    layer_name: str,  # 层名
) -> torch.Tensor:
    """MoE前向传播的假实现，用于torch.compile的shape推断"""
    return torch.empty_like(hidden_states)  # 返回空张量


direct_register_custom_op(  # 注册自定义算子
    op_name="transformers_moe_forward",  # 算子名
    op_func=_transformers_moe_forward,  # 实现函数
    mutates_args=["hidden_states"],  # 可变参数
    fake_impl=_transformers_moe_forward_fake,  # 假实现
)

try:
    from sglang.srt.compilation.compilation_config import SPLIT_OPS  # 导入分裂算子列表

    _MOE_SPLIT_OP = "sglang.transformers_moe_forward"  # MoE分裂算子名
    if _MOE_SPLIT_OP not in SPLIT_OPS:  # 如果未注册
        SPLIT_OPS.append(_MOE_SPLIT_OP)  # 添加到列表
except ImportError:
    pass  # 忽略导入错误


_BASE_DYNAMIC_ARG_DIMS: dict[str, int] = {  # 基础动态参数维度
    "input_ids": 0,  # 输入ID的batch维度
    "positions": 0,  # 位置的batch维度
    "input_embeds": 0,  # 输入嵌入的batch维度
}

_MULTIMODAL_DYNAMIC_ARG_DIMS: dict[str, int] = {  # 多模态动态参数维度
    "input_ids": 0,  # 输入ID的batch维度
    "positions": -1,  # last dim to support M-RoPE (Qwen2.5-VL 3×seq layout) # 位置的最后维度，支持M-RoPE
    "input_embeds": 0,  # 输入嵌入的batch维度
}


class TransformersBase(nn.Module):
    """Transformers后端模型基类，处理模块替换、PP、权重加载等"""

    torch_compile_dynamic_arg_dims: dict[str, int] = _BASE_DYNAMIC_ARG_DIMS  # 动态维度配置

    hf_to_sglang_mapper = WeightsMapper(  # HF到SGLang权重名称映射
        orig_to_new_prefix={
            "language_model.model.": "model.language_model.",  # 语言模型前缀
            "model.transformer.": "model.",  # transformer前缀
            "model.model.": "model.",  # 模型前缀
            "model.lm_head.": "lm_head.",  # LM头前缀
            "model.score.": "classifier.",  # 分数到分类器
            "model.classifier.": "classifier.",  # 分类器前缀
            "transformer.": "model.",  # transformer前缀
            "model.": "model.",  # 模型前缀
            "lm_head.": "lm_head.",  # LM头前缀
            "score.": "classifier.",  # 分数到分类器
            "classifier.": "classifier.",  # 分类器前缀
            "": "model.",  # 空前缀映射到模型
        }
    )

    def __init_subclass__(cls, *args, **kwargs):  # 子类初始化钩子
        super().__init_subclass__(*args, **kwargs)  # 调用父类
        mapper = WeightsMapper()  # 创建空映射器
        for base in cls.__mro__:  # 遍历方法解析顺序
            base_mapper = getattr(base, "hf_to_sglang_mapper", None)  # 获取基类映射器
            if base_mapper is not None:  # 如果有映射器
                mapper = mapper | base_mapper  # 合并映射器
        cls.hf_to_sglang_mapper = mapper  # 设置合并后的映射器

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        logger.info("Using Transformers backend.")  # 记录日志

        self.quant_config = quant_config  # 保存量化配置
        self.config = config  # 保存配置
        self.text_config = get_hf_text_config(config)  # 获取文本配置
        self.weight_mapper = self.hf_to_sglang_mapper  # 保存权重映射器
        self.pp_group = get_pp_group()  # 获取PP组

        # Weight loading attrs
        self.skip_prefixes: list[str] = []  # 跳过的前缀列表
        self.skip_substrs: list[str] = []  # 跳过的子串列表
        self.ignore_unexpected_prefixes: list[str] = []  # 忽略的意外前缀
        self.ignore_unexpected_suffixes: list[str] = []  # 忽略的意外后缀
        self.skip_substrs.extend([".attn.bias", ".attn.masked_bias", ".masked_bias"])  # 添加跳过子串
        self.ignore_unexpected_prefixes.extend(["classifier.", "score."])  # 添加忽略前缀

        if self.quant_config is not None:  # 如果有量化配置
            quant_method_name = self.quant_config.get_name()  # 获取量化方法名
            if "gptq" in quant_method_name:  # GPTQ量化
                self.ignore_unexpected_suffixes.append(".bias")  # 忽略偏置
            if "fp8" in quant_method_name:  # FP8量化
                fp8_suffix_map = {".activation_scale": ".input_scale"}  # FP8后缀映射
                use_mxfp8 = bool(getattr(self.quant_config, "use_mxfp8", False))  # 是否MXFP8
                weight_block_size = getattr(  # 权重块大小
                    self.quant_config, "weight_block_size", None
                )
                if not use_mxfp8 and weight_block_size is None:  # 非MXFP8且无块大小
                    fp8_suffix_map[".weight_scale_inv"] = ".weight_scale"  # 映射权重缩放
                self.weight_mapper = self.weight_mapper | WeightsMapper(  # 合并映射器
                    orig_to_new_suffix=fp8_suffix_map
                )

        # Resolve model class for _supports_attention_backend check
        model_cls = _resolve_attention_backend_model_cls(config)  # 解析模型类

        supports_backend = (  # 是否支持自定义注意力后端
            getattr(model_cls, "_supports_attention_backend", True)
            if model_cls
            else True
        )

        # Initialize on meta device to avoid premature GPU allocation
        self.text_config._attn_implementation = "sglang"  # 设置注意力实现为sglang
        if supports_backend:  # 如果支持后端
            with _init_on_device_without_buffers(torch.device("meta")):  # 在meta设备上初始化
                self.model: PreTrainedModel = AutoModel.from_config(  # 从配置创建模型
                    self.config,
                    torch_dtype=torch.get_default_dtype(),  # 默认数据类型
                    trust_remote_code=True,  # 信任远程代码
                )
        else:
            raise ValueError(  # 不支持则抛出异常
                f"Model {model_cls} does not support custom attention backends "
                "(_supports_attention_backend=False). The Transformers backend "
                "requires custom attention support."
            )

        self.vocab_size = getattr(  # 获取词表大小
            self.text_config,
            "vocab_size",
            self.model.get_input_embeddings().num_embeddings,  # 从嵌入层获取
        )
        self.unpadded_vocab_size = self.vocab_size  # 未填充词表大小

        # Embedding scale (e.g. Whisper)
        input_embeddings = self.model.get_input_embeddings()  # 获取输入嵌入
        self.embed_scale = getattr(input_embeddings, "embed_scale", None)  # 获取嵌入缩放

        self.start_layer = 0  # 起始层索引
        self.end_layer = getattr(self.text_config, "num_hidden_layers", 0)  # 结束层索引

        # Pipeline parallel
        self.pipeline_parallel()  # 执行流水线并行
        # Module replacement (Linear → TP, RMSNorm → fused, MoE overridden by MoEMixin)
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.recursive_replace()  # 递归替换模块
        # Attention instances
        self.attention_instances = self._create_attention_instances(tp_size)  # 创建注意力实例
        # Vocab embeddings
        self.replace_vocab_embed_class(self.model)  # 替换词表嵌入类

        # Initialize remaining meta-device parameters to real device tensors
        self._init_parameters(self.model)  # 初始化参数到真实设备

        self.lm_head: Optional[ParallelLMHead] = None  # LM头
        self.logits_processor: Optional[LogitsProcessor] = None  # logits处理器
        self.pooler: Optional[Pooler] = None  # 池化层

        self._compile_compatible = can_enable_torch_compile(config)  # 是否可编译

    @property
    def _can_torch_compile(self) -> bool:
        """Whether this model instance is safe to wrap with torch.compile."""  # 是否可以安全使用torch.compile
        return self._compile_compatible  # 返回编译兼容性

    def _init_parameters(self, module: nn.Module):
        """Materialize any parameters still on the meta device."""  # 将meta设备上的参数实体化
        for name, param in module.named_parameters(recurse=False):  # 遍历模块参数
            if param.device == torch.device("meta"):  # 如果在meta设备上
                new_param = nn.Parameter(  # 创建新参数
                    torch.empty_like(
                        param.data,
                        device=get_device(),  # 在真实设备上
                    )
                )
                setattr(module, name, new_param)  # 设置参数
        for child in module.children():  # 遍历子模块
            self._init_parameters(child)  # 递归初始化

    def log_replacement(self, name: str, old_module: nn.Module, new_module: nn.Module):
        """记录模块替换"""
        logger.debug("%s: %s -> %s", name, old_module, new_module)  # 调试日志

    # -- TP plan handling ---------------------------------------------------
    def _get_model_tp_plan(self) -> Mapping[str, str]:
        """获取模型的TP计划"""
        plan = (  # 尝试多种来源获取TP计划
            getattr(self.model, "tp_plan", None)
            or getattr(self.model, "_tp_plan", None)
            or getattr(self.model.config, "base_model_tp_plan", None)
            or getattr(self.text_config, "base_model_tp_plan", None)
        )
        if plan:  # 如果有计划
            return plan  # 返回计划

        plan = self._infer_tp_plan_from_children()  # 从子模块推断
        return plan if plan else {}  # 返回计划或空字典

    _LANGUAGE_MODEL_CHILD_NAMES = frozenset(  # 语言模型子模块名称集合
        {"language_model", "text_model", "model", "lm"}
    )

    def _infer_tp_plan_from_children(self) -> dict[str, str]:
        """从子模块推断TP计划"""
        plan: dict[str, str] = {}  # TP计划字典
        for child_name, child_module in self.model.named_children():  # 遍历子模块
            child_plan = getattr(child_module, "_tp_plan", None)  # 获取子模块计划
            if child_plan:  # 如果有计划
                plan.update({f"{child_name}.{k}": v for k, v in child_plan.items()})  # 更新计划
                continue  # 继续下一个

            child_config = getattr(child_module, "config", None)  # 获取子模块配置
            if child_config is not None:  # 如果有配置
                child_tp = getattr(child_config, "base_model_tp_plan", None)  # 获取配置TP计划
                if child_tp:  # 如果有计划
                    plan.update({f"{child_name}.{k}": v for k, v in child_tp.items()})  # 更新
                    continue  # 继续

            if child_name not in self._LANGUAGE_MODEL_CHILD_NAMES:  # 非语言模型子模块
                continue  # 跳过
            if child_config is None:  # 无配置
                continue  # 跳过
            model_type = getattr(child_config, "model_type", "")  # 获取模型类型
            base_type = (  # 去除VL和_text后缀
                model_type.replace("_vl_text", "")
                .replace("_vl", "")
                .replace("_text", "")
            )
            if base_type and base_type != model_type:  # 有后缀被去除
                try:
                    from transformers import AutoConfig  # 导入自动配置

                    base_cfg = AutoConfig.for_model(base_type)  # 获取基础配置
                    base_tp = getattr(base_cfg, "base_model_tp_plan", None)  # 获取基础TP计划
                    if base_tp:  # 如果有计划
                        plan.update(  # 更新计划
                            {f"{child_name}.{k}": v for k, v in base_tp.items()}
                        )
                except Exception as e:  # 获取失败
                    logger.debug(  # 调试日志
                        "Could not infer TP plan from base model type '%s': %s",
                        base_type,
                        e,
                    )
        return plan  # 返回TP计划

    def _normalize_tp_plan(self, tp_plan: Mapping[str, str]) -> dict[str, Style]:
        """归一化TP计划"""
        normalized = {}  # 归一化后的计划
        for pattern, style in tp_plan.items():  # 遍历计划
            if pattern.startswith("^model\\."):  # 以^model\.开头
                pattern = "^" + pattern[len("^model\\.") :]  # 移除model\.前缀
            elif pattern.startswith("model\\."):  # 以model\.开头
                pattern = pattern[len("model\\.") :]  # 移除model\.前缀
            elif pattern.startswith("model."):  # 以model.开头
                pattern = pattern[len("model.") :]  # 移除model.前缀
            normalized[pattern] = _normalize_tp_style(style)  # 归一化风格
        return normalized  # 返回归一化后的计划

    # -- Recursive module replacement (Linear + RMSNorm) --------------------
    def recursive_replace(self):
        """递归替换模块：Linear -> TP线性层，RMSNorm -> 融合RMSNorm"""
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        tp_plan = self._normalize_tp_plan(self._get_model_tp_plan())  # 获取归一化TP计划

        if not tp_plan and tp_size > 1:  # 无TP计划但需要TP
            raise ValueError(  # 抛出异常
                f"{type(self.model)} does not support tensor parallel yet!"
            )

        # Prefix patterns to match from `self.model`
        prefixed_plan = {maybe_prefix("model", k): v for k, v in tp_plan.items()}  # 添加前缀

        def _recursive_replace(module: nn.Module, prefix: str):  # 递归替换函数
            for child_name, child_module in module.named_children():  # 遍历子模块
                qual_name = maybe_prefix(prefix, child_name)  # 获取限定名
                new_module = child_module  # 默认不替换

                if isinstance(child_module, nn.Linear):  # 线性层
                    pattern = next(  # 匹配模式
                        (p for p in prefixed_plan if re.match(p, qual_name)),
                        None,
                    )
                    style = prefixed_plan.get(pattern, "replicate")  # 获取风格
                    new_module = replace_linear_class(  # 替换线性层
                        child_module,
                        style,
                        self.quant_config,
                        prefix=qual_name,
                    )
                elif child_module.__class__.__name__.endswith("RMSNorm"):  # RMSNorm
                    new_module = replace_rms_norm_class(  # 替换RMSNorm
                        child_module,
                        self.text_config.hidden_size,
                    )
                else:  # 其他模块
                    _recursive_replace(child_module, prefix=qual_name)  # 递归替换

                if new_module is not child_module:  # 如果替换了
                    setattr(module, child_name, new_module)  # 设置新模块
                    log_replacement(qual_name, child_module, new_module)  # 记录替换

        _recursive_replace(self.model, prefix="model")  # 从模型根开始替换

    # -- Pipeline parallel --------------------------------------------------
    def _get_model_pp_plan(self) -> Mapping[str, object]:
        """获取模型的PP计划"""
        return (  # 尝试多种来源
            getattr(self.model, "_pp_plan", None)
            or getattr(self.model, "pp_plan", None)
            or getattr(self.model.config, "base_model_pp_plan", None)
            or getattr(self.text_config, "base_model_pp_plan", None)
            or {}
        )

    def _register_missing_prefix(self, prefix: str):
        """注册缺失前缀到跳过列表"""
        if not prefix.endswith("."):  # 如果不以点结尾
            prefix += "."  # 添加点
        if prefix not in self.skip_prefixes:  # 如果前缀不在列表中
            self.skip_prefixes.append(prefix)  # 添加到跳过列表

    @staticmethod
    def _make_pp_missing_layer(original: nn.Module) -> PPMissingLayer:
        """Create a PPMissingLayer that preserves plain attributes from
        *original* so that the HF forward loop can still access per-layer
        metadata (e.g. ``attention_type`` on Qwen2 decoder layers)."""  # 创建PP缺失层，保留原始模块的属性
        replacement = PPMissingLayer()  # 创建缺失层
        for key, value in original.__dict__.items():  # 遍历原始属性
            if key.startswith("_"):  # 跳过私有属性
                continue
            if isinstance(value, (nn.Module, nn.Parameter, torch.Tensor)):  # 跳过模块和参数
                continue
            setattr(replacement, key, value)  # 复制属性
        return replacement  # 返回缺失层

    def _get_submodule_or_none(self, name: str) -> Optional[nn.Module]:
        """获取子模块，不存在返回None"""
        try:
            return self.model.get_submodule(name)  # 获取子模块
        except AttributeError:
            return None  # 返回None

    def _set_submodule(self, name: str, module: nn.Module):
        """设置子模块"""
        if "." in name:  # 如果有嵌套
            parent_name, child_name = name.rsplit(".", 1)  # 分割父名和子名
            parent_module = self.model.get_submodule(parent_name)  # 获取父模块
        else:
            parent_module = self.model  # 父模块是根
            child_name = name  # 子名
        setattr(parent_module, child_name, module)  # 设置子模块

    def pipeline_parallel(self):
        """设置流水线并行，替换非本秩的模块为缺失层"""
        if self.pp_group.world_size <= 1:  # 不需要PP
            return

        pp_plan = self._get_model_pp_plan()  # 获取PP计划
        if not pp_plan:  # 无PP计划
            raise ValueError(  # 抛出异常
                f"{type(self.model)} does not support pipeline parallel yet!"
            )

        pp_keys = [re.sub(r"^model\.", "", name) for name in pp_plan.keys()]  # PP键列表
        module_list_idx = None  # ModuleList索引
        module_list_name = None  # ModuleList名称
        for idx, name in enumerate(pp_keys):  # 遍历PP键
            if isinstance(self._get_submodule_or_none(name), nn.ModuleList):  # 如果是ModuleList
                if module_list_idx is not None:  # 已有ModuleList
                    raise ValueError(  # 不支持多个ModuleList
                        "Pipeline parallel with multiple ModuleList blocks is not supported."
                    )
                module_list_idx = idx  # 记录索引
                module_list_name = name  # 记录名称

        if module_list_idx is None or module_list_name is None:  # 没有找到ModuleList
            raise ValueError(f"Could not find ModuleList in {type(self.model)}.")  # 抛出异常

        keep_prefix_modules = self.pp_group.is_first_rank or (  # 保留前缀模块的条件
            getattr(self.text_config, "tie_word_embeddings", False)
            and self.pp_group.is_last_rank
        )
        for name in pp_keys[:module_list_idx]:  # ModuleList之前的模块
            if keep_prefix_modules:  # 需要保留
                continue  # 跳过
            self._set_submodule(name, PPMissingLayer())  # 替换为缺失层
            self._register_missing_prefix(maybe_prefix("model", name))  # 注册缺失前缀

        layers = self.model.get_submodule(module_list_name)  # 获取层列表
        self.start_layer, self.end_layer = get_pp_indices(  # 获取PP索引
            len(layers),
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        for idx in range(len(layers)):  # 遍历层
            if self.start_layer <= idx < self.end_layer:  # 本秩负责的层
                continue  # 保留
            layers[idx] = self._make_pp_missing_layer(layers[idx])  # 替换为缺失层
            self._register_missing_prefix(  # 注册缺失前缀
                maybe_prefix("model", f"{module_list_name}.{idx}")
            )

        for name in pp_keys[module_list_idx + 1 :]:  # ModuleList之后的模块
            if self.pp_group.is_last_rank:  # 最后一个秩保留
                continue  # 跳过
            self._set_submodule(name, PPMissingLayer())  # 替换为缺失层
            self._register_missing_prefix(maybe_prefix("model", name))  # 注册缺失前缀

    # -- Attention instances ------------------------------------------------
    def _create_attention_instances(self, tp_size: int) -> dict[int, RadixAttention]:
        """为每层创建RadixAttention实例"""
        num_heads = self.text_config.num_attention_heads  # 总头数
        num_kv_heads = getattr(self.text_config, "num_key_value_heads", num_heads)  # KV头数
        hidden_size = self.text_config.hidden_size  # 隐藏大小
        head_dim = getattr(self.text_config, "head_dim", hidden_size // num_heads)  # 头维度

        layer_types = getattr(self.text_config, "layer_types", None) or getattr(  # 层类型
            self.config, "layer_types", None
        )
        global_sliding_window = getattr(  # 全局滑动窗口
            self.text_config, "sliding_window", None
        ) or getattr(self.config, "sliding_window", None)

        # Detect encoder-only models (non-causal attention everywhere)
        is_encoder_only = any(  # 检测纯编码器模型
            not getattr(m, "is_causal", True)
            for m in self.model.modules()
            if hasattr(m, "is_causal")
        )
        if is_encoder_only and self.config != self.text_config:  # 多模态模型非纯编码器
            is_encoder_only = False
        if is_encoder_only:  # 纯编码器
            logger.info(  # 记录信息
                "Detected encoder-only model (non-causal attention). "
                "Using RadixAttention with is_cross_attention=True."
            )

        instances = {}  # 注意力实例字典
        for idx in range(self.start_layer, self.end_layer):  # 遍历层
            # Per-layer sliding window (e.g. Gemma2, Cohere)
            per_layer_sliding_window = -1  # 默认无滑动窗口
            if (  # 如果该层使用滑动注意力
                layer_types is not None
                and idx < len(layer_types)
                and layer_types[idx] == "sliding_attention"
                and global_sliding_window is not None
            ):
                per_layer_sliding_window = global_sliding_window  # 设置滑动窗口大小

            instances[idx] = RadixAttention(  # 创建注意力实例
                num_heads=divide(num_heads, tp_size),  # TP后的头数
                head_dim=head_dim,  # 头维度
                scaling=head_dim**-0.5,  # 缩放因子
                num_kv_heads=divide(num_kv_heads, tp_size),  # TP后的KV头数
                layer_id=idx,  # 层ID
                quant_config=self.quant_config,  # 量化配置
                sliding_window_size=per_layer_sliding_window,  # 滑动窗口
                is_cross_attention=is_encoder_only,  # 是否交叉注意力
                prefix=f"{idx}.attn",  # 前缀
            )
        return instances  # 返回实例字典

    # -- Vocab embedding replacement ----------------------------------------
    def replace_vocab_embed_class(self, module: nn.Module):
        """替换词表嵌入类为SGLang并行嵌入"""
        old_module = self.model.get_input_embeddings()  # 获取原始嵌入
        if old_module is None or isinstance(old_module, PPMissingLayer):  # 不存在或PP缺失
            return
        embedding_dim = getattr(old_module, "embedding_dim", None)  # 获取嵌入维度
        if embedding_dim is None:  # 没有嵌入维度
            embedding_dim = _getattr_first(  # 从配置获取
                self.text_config,
                ("embedding_size", "hidden_size"),
                None,
            )
        assert embedding_dim is not None  # 断言嵌入维度存在
        new_module = VocabParallelEmbedding(  # 创建并行嵌入
            self.vocab_size,
            embedding_dim,
            org_num_embeddings=self.vocab_size,
            quant_config=None,
        )

        old_embed_scale = getattr(old_module, "embed_scale", None)  # 获取嵌入缩放
        if old_embed_scale is not None:  # 如果有缩放
            base_cls = new_module.__class__  # 获取基类

            class ScaledEmbedding(base_cls):  # 带缩放的嵌入
                def forward(self, input_):
                    return base_cls.forward(self, input_) * self.embed_scale  # 缩放

            new_module.__class__ = ScaledEmbedding  # 替换类
            new_module.embed_scale = old_embed_scale  # 设置缩放
            self.embed_scale = None  # 清除基类缩放

        self.log_replacement("input embedding", old_module, new_module)  # 记录替换
        self.model.set_input_embeddings(new_module)  # 设置新嵌入

    # -- Forward ------------------------------------------------------------
    def _format_position_ids(self, positions: torch.Tensor) -> torch.Tensor:
        """格式化位置ID，为HF模型添加batch维度"""
        if positions.ndim == 2 and positions.shape[0] == 3:  # M-RoPE (3D)
            return positions[:, None, ...]  # 添加head维度
        if positions.ndim == 1:  # 1D位置
            return positions[None, ...]  # 添加batch维度
        return positions  # 其他情况不变

    def _run_hf_backbone(
        self,
        input_ids: Optional[torch.Tensor],  # 输入ID
        input_embeds: Optional[torch.Tensor],  # 输入嵌入
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        **kwargs,
    ) -> torch.Tensor:
        """运行HF模型骨干"""
        hf_input_ids = None if input_ids is None else input_ids[None, ...]  # 添加batch维度
        hf_input_embeds = None  # HF输入嵌入
        if input_embeds is not None:  # 如果有输入嵌入
            hf_input_embeds = input_embeds[None, ...]  # 添加batch维度
            hf_input_ids = None  # 有嵌入时不用ID

        # Scale embeddings if needed
        if (  # 如果需要缩放嵌入
            self.embed_scale is not None
            and hf_input_ids is not None
            and hf_input_embeds is None
        ):
            hf_input_embeds = (  # 计算缩放后的嵌入
                self.model.get_input_embeddings()(hf_input_ids) * self.embed_scale
            )
            hf_input_ids = None  # 清除ID

        return self.model(  # 调用HF模型
            input_ids=hf_input_ids,  # 输入ID
            inputs_embeds=hf_input_embeds,  # 输入嵌入
            use_cache=False,  # 不使用缓存
            position_ids=self._format_position_ids(positions),  # 位置ID
            return_dict=False,  # 不返回字典
            forward_batch=forward_batch,  # 前向批次
            attention_instances=self.attention_instances,  # 注意力实例
            **kwargs,
        )[0][0, ...]  # 取第一个输出的第一个batch

    def _forward_hidden_states(
        self,
        input_ids: Optional[torch.Tensor],  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
    ) -> torch.Tensor:
        """前向传播获取隐藏状态"""
        return self._run_hf_backbone(  # 运行HF骨干
            input_ids=input_ids,
            input_embeds=input_embeds,
            positions=positions,
            forward_batch=forward_batch,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
        input_embeds: torch.Tensor = None,  # 输入嵌入
        get_embedding: bool = False,  # 是否获取嵌入
    ) -> Union[LogitsProcessorOutput, EmbeddingPoolerOutput, PPProxyTensors]:
        """模型前向传播，支持PP和嵌入获取"""
        runtime_input_ids: Optional[torch.Tensor] = input_ids  # 运行时输入ID
        runtime_input_embeds = input_embeds  # 运行时输入嵌入
        if not self.pp_group.is_first_rank:  # 非第一个PP秩
            assert pp_proxy_tensors is not None  # 断言有代理张量
            runtime_input_ids = None  # 不使用输入ID
            runtime_input_embeds = pp_proxy_tensors["hidden_states"]  # 使用代理隐藏状态

        hidden_states = self._forward_hidden_states(  # 获取隐藏状态
            input_ids=runtime_input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=runtime_input_embeds,
        )

        if not self.pp_group.is_last_rank:  # 非最后一个PP秩
            return PPProxyTensors(  # 返回代理张量
                {"hidden_states": hidden_states, "residual": hidden_states}
            )

        if get_embedding:  # 获取嵌入
            assert (
                self.pooler is not None
            ), "pooling is not enabled for this model class"
            return self.pooler(hidden_states, forward_batch)  # 返回池化结果

        assert self.logits_processor is not None and self.lm_head is not None  # 断言处理器存在
        return self.logits_processor(  # 返回logits处理结果
            input_ids, hidden_states, self.lm_head, forward_batch, None
        )

    # -- Weight loading -----------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载模型权重"""
        loader = AutoWeightsLoader(  # 创建权重加载器
            self,
            skip_prefixes=self.skip_prefixes,  # 跳过前缀
            skip_substrs=self.skip_substrs,  # 跳过子串
            ignore_unexpected_prefixes=self.ignore_unexpected_prefixes,  # 忽略前缀
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,  # 忽略后缀
        )
        return loader.load_weights(weights, mapper=self.weight_mapper)  # 加载权重


class CausalMixin:
    """因果语言模型混入类，添加LM头和logits处理器"""

    def __init__(self, *args, prefix: str = "", **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)  # 调用父类初始化

        tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)  # 是否绑定词嵌入
        if tie_word_embeddings:  # 绑定词嵌入
            self.skip_prefixes.append("lm_head.")  # 跳过LM头权重

        if not self.pp_group.is_last_rank:  # 非最后一个PP秩
            self._register_missing_prefix("lm_head")  # 注册缺失前缀
            return

        self.lm_head = ParallelLMHead(  # 创建并行LM头
            self.vocab_size,
            self.text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if tie_word_embeddings:  # 绑定词嵌入
            self.lm_head.weight = self.model.get_input_embeddings().weight  # 共享嵌入权重

        logit_scale = getattr(self.text_config, "logit_scale", 1.0)  # logit缩放
        self.logits_processor = LogitsProcessor(  # 创建logits处理器
            self.text_config, logit_scale=logit_scale
        )


class EmbeddingMixin:
    """嵌入模型混入类，添加池化层"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类初始化
        self.ignore_unexpected_prefixes.append("lm_head.")  # 忽略LM头权重
        if not self.pp_group.is_last_rank:  # 非最后一个PP秩
            return
        pooling_name = str(getattr(self.config, "pooling_type", "LAST")).upper()  # 池化类型
        pooling_type = PoolingType.CLS if pooling_name == "CLS" else PoolingType.LAST  # 池化方式
        normalize = bool(getattr(self.config, "normalize", True))  # 是否归一化
        self.pooler = Pooler(pooling_type=pooling_type, normalize=normalize)  # 创建池化层


class MoEMixin:
    """MoE混入类，替换专家模块为融合MoE"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类初始化

    @classmethod
    def get_model_config_for_expert_location(
        cls, config
    ) -> Optional[ModelConfigForExpertLocation]:
        """获取专家位置的模型配置"""
        text_config = getattr(config, "text_config", config)  # 获取文本配置
        num_experts = _getattr_first(  # 获取专家数
            text_config,
            ("num_local_experts", "num_experts", "n_routed_experts"),
        )
        if num_experts is None:  # 没有专家
            return None
        num_groups = getattr(text_config, "n_group", None)  # 获取组数
        return ModelConfigForExpertLocation(  # 返回配置
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=num_groups,
        )

    @property
    def routed_experts_weights_of_layer(self) -> dict[int, list[torch.Tensor]]:
        """获取每层路由专家的权重"""
        return {
            fused.experts.layer_id: fused.get_moe_weights() for fused in self.moe_layers  # 遍历MoE层
        }

    def _get_expert_mapping(self, num_experts: int) -> List[Tuple[str, str, int, str]]:
        """获取专家参数映射"""
        ckpt_names = [  # 检查点中的参数名称组合
            ("gate_proj", "down_proj", "up_proj"),  # 标准名称
            ("w1", "w2", "w3"),  # Mixtral风格
            ("linear", "linear_1", "linear_v"),  # 其他风格
        ]
        mapping: list = []  # 映射列表
        for gate, down, up in ckpt_names:  # 遍历名称组合
            mapping.extend(  # 添加映射
                FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name=gate,
                    ckpt_down_proj_name=down,
                    ckpt_up_proj_name=up,
                    num_experts=num_experts,
                )
            )
        # AutoWeightsLoader dispatches to TransformersFusedMoE (which IS the
        # ``experts`` module) so the incoming weight names have the "experts."
        # prefix already stripped.  Remove it from weight_name in the mapping.
        mapping = [  # 移除experts.前缀
            (pn, wn.removeprefix("experts."), eid, sid) for pn, wn, eid, sid in mapping
        ]
        return mapping  # 返回映射

    def recursive_replace(self):
        """Replace experts modules with TransformersFusedMoE, then call
        super().recursive_replace() for Linear/RMSNorm replacement."""  # 替换专家模块为融合MoE，然后调用父类替换
        text_config = self.text_config  # 获取文本配置

        num_experts = _getattr_first(  # 获取专家数
            text_config,
            ("num_local_experts", "num_experts", "n_routed_experts"),
        )
        assert num_experts is not None, "Cannot determine num_experts from config."  # 断言专家数存在

        top_k = _getattr_first(text_config, ("num_experts_per_tok", "top_k"))  # 获取Top-K
        assert top_k is not None, "Cannot determine top_k from config."  # 断言Top-K存在

        hidden_size = text_config.hidden_size  # 隐藏大小
        intermediate_size = _getattr_first(  # 中间层大小
            text_config,
            ("moe_intermediate_size", "intermediate_size"),
        )
        assert intermediate_size is not None, "Cannot determine intermediate_size."  # 断言存在

        num_shared_experts = _getattr_first(  # 共享专家数
            text_config,
            ("n_shared_experts", "moe_num_shared_experts"),
            0,
        )
        reduce_results = num_shared_experts == 0  # 无共享专家时归约

        renormalize = getattr(text_config, "norm_topk_prob", top_k > 1)  # 是否重归一化

        # Activation function
        activation = "silu"  # 默认SiLU
        wrapped_arch = self.config.architectures[0].lower()  # 获取架构名
        if "gptoss" in wrapped_arch:  # GPT-OSS
            activation = "swigluoai"
        elif "grok1" in wrapped_arch:  # Grok1
            activation = "gelu"

        # Expert mapping for AutoWeightsLoader
        expert_mapping = self._get_expert_mapping(num_experts)  # 获取专家映射

        # EPLB / EP tracking
        num_redundant = get_global_server_args().ep_num_redundant_experts  # 冗余专家数
        ep_size = get_moe_expert_parallel_world_size()  # EP大小

        self.mlp_moe_layers: list[nn.Module] = []  # MoE MLP层列表
        self.moe_layers: list[TransformersFusedMoE] = []  # MoE层列表
        self.num_moe_layers = 0  # MoE层数
        self.num_logical_experts = num_experts  # 逻辑专家数
        self.num_physical_experts = num_experts + num_redundant  # 物理专家数
        self.num_local_physical_experts = self.num_physical_experts // max(ep_size, 1)  # 本地物理专家数
        self.num_shared_experts = num_shared_experts  # 共享专家数
        self.num_redundant_experts = num_redundant  # 冗余专家数

        def _add_all_reduce(mlp: nn.Module):  # 添加全归约包装
            class MLPWithAllReduce(mlp.__class__):  # 带全归约的MLP
                def forward(self, *args, **kwargs):
                    output = super().forward(*args, **kwargs)  # 前向传播
                    return self.experts.maybe_all_reduce_tensor_model_parallel(output)  # 全归约

            mlp.__class__ = MLPWithAllReduce  # 替换类

        def _recursive_replace(module: nn.Module, prefix: str):  # 递归替换
            for child_name, child_module in module.named_children():  # 遍历子模块
                qual_name = maybe_prefix(prefix, child_name)  # 限定名

                is_modulelist = isinstance(child_module, nn.ModuleList)  # 是否ModuleList
                params = list(child_module.parameters())  # 参数列表
                is_3d = len(params) > 0 and all(p.ndim == 3 for p in params)  # 是否3D参数

                if child_name == "experts" and (is_modulelist or is_3d):  # 找到专家模块
                    mlp = module  # MLP模块
                    experts = child_module  # 专家列表

                    has_bias = any("bias" in n for n, _ in experts.named_parameters())  # 是否有偏置

                    nonlocal reduce_results
                    if reduce_results:  # 如果需要归约
                        if any("shared_expert" in n for n, _ in mlp.named_parameters()):  # 有共享专家
                            reduce_results = False  # 不归约
                            self.num_shared_experts = 1  # 设置共享专家数

                    layer_id = self.num_moe_layers  # 层ID

                    fused_experts = TransformersFusedMoE(  # 创建融合MoE
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        layer_id=layer_id,
                        reduce_results=reduce_results,
                        quant_config=self.quant_config,
                        prefix=qual_name,
                        activation=activation,
                        with_bias=has_bias,
                        expert_mapping=expert_mapping,
                    )
                    mlp.experts = fused_experts  # 替换专家
                    log_replacement(qual_name, experts, fused_experts)  # 记录替换

                    self.mlp_moe_layers.append(mlp)  # 添加到MoE MLP列表
                    self.moe_layers.append(fused_experts)  # 添加到MoE列表
                    self.num_moe_layers += 1  # 增加计数

                    if not reduce_results and (  # 不归约且有并行
                        fused_experts.tp_size > 1 or fused_experts.ep_size > 1
                    ):
                        _add_all_reduce(mlp)  # 添加全归约包装
                else:
                    _recursive_replace(child_module, prefix=qual_name)  # 递归替换

        _recursive_replace(self.model, prefix="model")  # 从模型根开始
        super().recursive_replace()  # 调用父类替换


class MultiModalMixin:
    """多模态混入类，处理图像/视频/音频特征编码"""

    torch_compile_dynamic_arg_dims: dict[str, int] = _MULTIMODAL_DYNAMIC_ARG_DIMS  # 动态维度

    # Older VL checkpoints (e.g. Qwen2.5-VL) store text weights as
    # "model.layers.*" but transformers >=5.0 nests the text model under
    # "model.language_model.*".  Map explicitly so these load correctly.
    hf_to_sglang_mapper = WeightsMapper(  # 多模态权重映射
        orig_to_new_prefix={
            "language_model.model.": "model.language_model.",  # 语言模型
            "text_model.model.": "model.text_model.",  # 文本模型
            "text_model.lm_head.": "lm_head.",  # 文本LM头
            "language_model.lm_head.": "lm_head.",  # 语言LM头
            "vision_tower.": "model.vision_tower.",  # 视觉塔
            "vision_model.": "model.vision_model.",  # 视觉模型
            "vision_embed_tokens.": "model.vision_embed_tokens.",  # 视觉嵌入
            "image_newline.": "model.image_newline.",  # 图像换行
            "vqmodel.": "model.vqmodel.",  # VQ模型
            "multi_modal_projector.": "model.multi_modal_projector.",  # 多模态投影器
            "visual.": "model.visual.",  # 视觉
            "model.layers.": "model.language_model.layers.",  # 层
            "model.embed_tokens.": "model.language_model.embed_tokens.",  # 嵌入
            "model.norm.": "model.language_model.norm.",  # 归一化
            "model.rotary_emb.": "model.language_model.rotary_emb.",  # 旋转编码
        }
    )

    _mm_feature_kwarg = {  # 多模态特征关键字参数
        "image": "pixel_values",  # 图像像素值
        "video": "pixel_values_videos",  # 视频像素值
        "audio": "input_features",  # 音频特征
    }
    _mm_encoder_candidates = {  # 多模态编码器候选方法
        "image": ("get_image_features", "get_image_feature"),  # 图像
        "video": ("get_video_features", "get_video_feature"),  # 视频
        "audio": ("get_audio_features", "get_audio_feature"),  # 音频
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类初始化
        self._mm_padding_pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 多模态填充模式

    def _uses_mrope_positions(self) -> bool:
        """检查是否使用M-RoPE位置编码"""
        rope_scaling = getattr(self.text_config, "rope_scaling", None)  # 获取RoPE缩放
        if isinstance(rope_scaling, Mapping) and "mrope_section" in rope_scaling:
            return True  # 使用M-RoPE
        rope_type = str(getattr(self.text_config, "rope_type", "")).lower()  # 获取RoPE类型
        return "mrope" in rope_type  # 类型包含mrope

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        """对输入ID进行多模态填充"""
        return input_ids  # 直接返回（子类可重写）

    def _get_modality_encoder(self, modality_name: str):
        """获取指定模态的编码器方法"""
        for name in self._mm_encoder_candidates[modality_name]:  # 遍历候选方法
            fn = getattr(self.model, name, None)  # 尝试获取方法
            if fn is not None:  # 方法存在
                return fn  # 返回方法
        raise AttributeError(f"No encoder method found for modality '{modality_name}'")  # 抛出异常

    def _get_modality_dtype_device(
        self, modality_name: str
    ) -> tuple[Optional[torch.dtype], Optional[torch.device]]:
        """获取指定模态编码器的数据类型和设备"""
        module_candidates = {  # 模态模块候选
            "image": ("vision_tower", "vision_model"),  # 图像
            "video": ("video_tower", "vision_tower", "vision_model"),  # 视频
            "audio": ("audio_tower", "audio_model", "audio_encoder"),  # 音频
        }
        modules = []  # 模块列表
        for name in module_candidates.get(modality_name, ()):  # 遍历候选
            module = getattr(self.model, name, None)  # 获取模块
            if module is not None:  # 模块存在
                modules.append(module)  # 添加到列表
        modules.append(self.model)  # 添加模型本身

        for module in modules:  # 遍历模块
            for param in module.parameters():  # 遍历参数
                if torch.is_floating_point(param):  # 浮点参数
                    return param.dtype, param.device  # 返回类型和设备
            for buf in module.buffers():  # 遍历缓冲区
                if torch.is_floating_point(buf):  # 浮点缓冲区
                    return buf.dtype, buf.device  # 返回类型和设备
        return None, None  # 未找到返回None

    def _cast_mm_value(self, value, dtype, device):
        """递归转换多模态值的数据类型和设备"""
        if torch.is_tensor(value):  # 张量
            if value.is_floating_point() and dtype is not None:  # 浮点且有目标类型
                return value.to(dtype=dtype, device=device)  # 转换
            return value  # 不转换
        if isinstance(value, dict):  # 字典
            return {k: self._cast_mm_value(v, dtype, device) for k, v in value.items()}  # 递归转换
        if isinstance(value, list):  # 列表
            return [self._cast_mm_value(v, dtype, device) for v in value]  # 递归转换
        if isinstance(value, tuple):  # 元组
            return tuple(self._cast_mm_value(v, dtype, device) for v in value)  # 递归转换
        return value  # 其他类型不转换

    def _to_tensor_output(self, output) -> torch.Tensor:
        """将编码器输出转换为张量"""
        if hasattr(output, "pooler_output") and output.pooler_output is not None:  # 有pooler输出
            output = output.pooler_output  # 使用pooler输出
        if isinstance(output, tuple):  # 元组
            output = output[0]  # 取第一个
        if isinstance(output, (list, tuple)):  # 列表或元组
            if len(output) == 0:  # 空列表
                raise ValueError("Empty multimodal encoder output.")  # 抛出异常
            if all(torch.is_tensor(x) for x in output):  # 全是张量
                output = torch.cat(  # 拼接
                    [x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x for x in output],
                    dim=0,
                )
            else:
                output = output[0]  # 取第一个
        elif hasattr(output, "last_hidden_state"):  # 有last_hidden_state
            output = output.last_hidden_state  # 使用
        elif isinstance(output, dict):  # 字典
            if output.get("pooler_output", None) is not None:  # 有pooler
                output = output["pooler_output"]  # 使用pooler
            else:
                output = next(v for v in output.values() if torch.is_tensor(v))  # 取第一个张量
            if isinstance(output, (list, tuple)):  # 输出是列表或元组
                if len(output) == 0:  # 空列表
                    raise ValueError("Empty multimodal encoder output.")  # 抛出异常
                if all(torch.is_tensor(x) for x in output):  # 全是张量
                    output = torch.cat(  # 拼接
                        [
                            x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
                            for x in output
                        ],
                        dim=0,
                    )
                else:
                    output = output[0]  # 取第一个

        if output.ndim > 2:  # 多维
            output = output.reshape(-1, output.shape[-1])  # 展平为2D
        return output  # 返回输出

    def _encode_modality_items(
        self, modality_name: str, items: list[MultimodalDataItem]
    ) -> torch.Tensor:
        """编码指定模态的数据项"""
        encoder = self._get_modality_encoder(modality_name)  # 获取编码器
        feature_kwarg = self._mm_feature_kwarg[modality_name]  # 获取特征关键字
        target_dtype, target_device = self._get_modality_dtype_device(modality_name)  # 获取目标类型和设备
        outputs = []  # 输出列表
        for item in items:  # 遍历数据项
            kwargs = self._cast_mm_value(  # 转换模型特定数据
                dict(item.model_specific_data),
                dtype=target_dtype,
                device=target_device,
            )
            feature = self._cast_mm_value(  # 转换特征
                item.feature,
                dtype=target_dtype,
                device=target_device,
            )
            if _encoder_accepts_feature_kwarg(encoder, feature_kwarg):  # 编码器接受特征关键字
                kwargs[feature_kwarg] = feature  # 设置特征
                result = encoder(**kwargs)  # 调用编码器
            else:
                result = encoder(feature, **kwargs)  # 特征作为位置参数
            outputs.append(self._to_tensor_output(result))  # 添加到输出列表
        return torch.cat(outputs, dim=0)  # 拼接并返回

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """获取图像特征"""
        return self._encode_modality_items("image", items)  # 编码图像

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """获取视频特征"""
        return self._encode_modality_items("video", items)  # 编码视频

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """获取音频特征"""
        return self._encode_modality_items("audio", items)  # 编码音频

    def _collect_mm_kwargs(self, forward_batch: ForwardBatch) -> dict:
        """Collect multimodal tensors from the forward batch and return them
        as kwargs suitable for the HF model's forward method."""  # 从前向批次收集多模态张量作为HF模型的kwargs
        kwargs = {}  # kwargs字典

        if getattr(forward_batch, "token_type_ids", None) is not None:  # 有token类型ID
            tti = forward_batch.token_type_ids  # 获取token类型ID
            if tti.ndim == 1:  # 1D
                tti = tti.unsqueeze(0)  # 添加batch维度
            token_type_key = (  # 确定关键字名
                "mm_token_type_ids"
                if "mm_token_type_ids"
                in inspect.signature(self.model.forward).parameters
                else "token_type_ids"
            )
            kwargs[token_type_key] = tti  # 设置token类型

        if (  # 预填充阶段且有多模态输入
            not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs = forward_batch.mm_inputs  # 多模态输入
            target_device = next(self.model.parameters()).device  # 目标设备

            for batch_idx in range(len(mm_inputs or [])):  # 遍历批次
                mm_input = mm_inputs[batch_idx]  # 获取输入
                if mm_input is None:  # 空输入
                    continue
                for item in mm_input.mm_items or []:  # 遍历多模态项
                    for key, value in (item.model_specific_data or {}).items():  # 遍历模型数据
                        if isinstance(value, torch.Tensor):  # 张量
                            value = value.to(device=target_device)  # 移到目标设备
                        if key not in kwargs:  # 新关键字
                            kwargs[key] = value  # 设置值
                        elif isinstance(value, torch.Tensor) and isinstance(  # 已有且都是张量
                            kwargs[key], torch.Tensor
                        ):
                            kwargs[key] = torch.cat([kwargs[key], value], dim=0)  # 拼接
                    if item.feature is not None:  # 有特征
                        feature_key = self._mm_feature_kwarg.get(  # 获取特征关键字
                            item.modality.name.lower(), "pixel_values"
                        )
                        feature = item.feature  # 获取特征
                        if isinstance(feature, torch.Tensor):  # 张量
                            feature = feature.to(device=target_device)  # 移到目标设备
                        if feature_key not in kwargs:  # 新关键字
                            kwargs[feature_key] = feature  # 设置值
                        elif isinstance(feature, torch.Tensor) and isinstance(  # 已有且都是张量
                            kwargs[feature_key], torch.Tensor
                        ):
                            kwargs[feature_key] = torch.cat(  # 拼接
                                [kwargs[feature_key], feature], dim=0
                            )

        return kwargs  # 返回kwargs

    def _forward_hidden_states(
        self,
        input_ids: Optional[torch.Tensor],  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
    ) -> torch.Tensor:
        """多模态模型的隐藏状态前向传播"""
        if input_embeds is not None:  # 如果有输入嵌入
            return super()._forward_hidden_states(  # 调用父类方法
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )

        if (  # 使用M-RoPE位置
            self._uses_mrope_positions()
            and getattr(forward_batch, "mrope_positions", None) is not None
        ):
            positions = forward_batch.mrope_positions  # 使用M-RoPE位置

        mm_kwargs = self._collect_mm_kwargs(forward_batch)  # 收集多模态kwargs

        return self._run_hf_backbone(  # 运行HF骨干
            input_ids=input_ids,
            input_embeds=None,
            positions=positions,
            forward_batch=forward_batch,
            **mm_kwargs,
        )


class TransformersForCausalLM(CausalMixin, TransformersBase):  # Transformers因果语言模型
    pass


class TransformersMoEForCausalLM(MoEMixin, CausalMixin, TransformersBase):  # Transformers MoE因果语言模型
    pass


class TransformersMultiModalForCausalLM(MultiModalMixin, CausalMixin, TransformersBase):  # Transformers多模态因果语言模型
    pass


class TransformersMultiModalMoEForCausalLM(  # Transformers多模态MoE因果语言模型
    MultiModalMixin, MoEMixin, CausalMixin, TransformersBase
):
    pass


class TransformersEmbeddingModel(EmbeddingMixin, TransformersBase):  # Transformers嵌入模型
    pass


class TransformersMoEEmbeddingModel(MoEMixin, EmbeddingMixin, TransformersBase):  # Transformers MoE嵌入模型
    pass


class TransformersMultiModalEmbeddingModel(  # Transformers多模态嵌入模型
    MultiModalMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEEmbeddingModel(  # Transformers多模态MoE嵌入模型
    MultiModalMixin, MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersForSequenceClassification(EmbeddingMixin, TransformersBase):  # Transformers序列分类模型
    pass


class TransformersMoEForSequenceClassification(  # Transformers MoE序列分类模型
    MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalForSequenceClassification(  # Transformers多模态序列分类模型
    MultiModalMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEForSequenceClassification(  # Transformers多模态MoE序列分类模型
    MultiModalMixin, MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


EntryClass = [  # 入口类列表
    TransformersForCausalLM,
    TransformersMoEForCausalLM,
    TransformersMultiModalForCausalLM,
    TransformersMultiModalMoEForCausalLM,
    TransformersEmbeddingModel,
    TransformersMoEEmbeddingModel,
    TransformersMultiModalEmbeddingModel,
    TransformersMultiModalMoEEmbeddingModel,
    TransformersForSequenceClassification,
    TransformersMoEForSequenceClassification,
    TransformersMultiModalForSequenceClassification,
    TransformersMultiModalMoEForSequenceClassification,
]
