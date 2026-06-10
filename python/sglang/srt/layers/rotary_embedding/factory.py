# 旋转位置编码（RoPE）工厂函数模块，根据配置创建不同类型的RoPE实例（含多种缩放策略）
"""Factory functions: get_rope, get_rope_cpu, get_rope_wrapper.  # 工厂函数：get_rope、get_rope_cpu、get_rope_wrapper"""

from __future__ import annotations  # 启用延迟注解求值 # 启用延迟注解

import logging  # 导入日志模块 # 导入日志模块
from typing import Any, Dict, Optional, Tuple  # 导入类型注解 # 导入类型提示

import torch  # 导入PyTorch # 导入PyTorch框架

from sglang.srt.layers.rotary_embedding.base import (  # 导入基础RoPE类 # 导入基础RoPE类
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.mrope import (  # 导入多模态RoPE类 # 导入多模态RoPE类
    MRotaryEmbedding,
    YaRNScalingMRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.rope_variant import (  # 导入RoPE变体类 # 导入RoPE变体类
    DeepseekScalingRotaryEmbedding,
    DualChunkRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    FourierRotaryEmbedding,
    Gemma4RotaryEmbedding,
    Llama3RotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding  # 导入YaRN缩放RoPE # 导入YaRN缩放RoPE
from sglang.srt.utils import get_bool_env_var, is_hip  # 导入工具函数 # 导入工具函数

logger = logging.getLogger(__name__)  # 获取日志器 # 获取日志器


def _get_rope_param(rope_scaling, key, default, scaling_type):  # 从rope_scaling字典获取参数，缺失时发出警告 # 从rope_scaling获取参数
    """Get a parameter from rope_scaling dict, warn if missing.  # 从rope_scaling字典获取参数，缺失时发出警告。

    In transformers v5, config.rope_scaling is an alias for rope_parameters  # 在transformers v5中，config.rope_scaling是rope_parameters的别名
    which may be non-None even for models with no actual scaling (rope_type=default).  # 即使对于没有实际缩放的模型（rope_type=default）也可能非None。
    When a required key is missing, this logs a warning instead of silently  # 当缺少必需的键时，此函数记录警告而不是静默
    defaulting, to make config mismatches easier to debug.  # 使用默认值，以使配置不匹配更容易调试。
    """
    if key in rope_scaling:  # 键存在则返回值 # 键存在返回值
        return rope_scaling[key]
    logger.warning(  # 键缺失时记录警告 # 键缺失时记录警告
        "rope_scaling (type=%s) missing key '%s', defaulting to %s. "
        "This may indicate a v5 config issue — check model accuracy.",
        scaling_type,
        key,
        default,
    )
    return default  # 返回默认值 # 返回默认值


_is_hip = is_hip()  # 是否为HIP平台 # 是否为HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER # 是否使用AITER

if _use_aiter:  # HIP平台使用AITER时导入aiter的RoPE # 使用AITER时导入aiter RoPE
    from aiter.rotary_embedding import get_rope as aiter_get_rope  # 导入aiter的get_rope # 导入aiter的get_rope

_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}  # RoPE实例缓存字典 # RoPE实例缓存


def get_rope(  # 创建或获取RoPE实例的工厂函数 # 创建或获取RoPE实例
    head_size: int,  # 注意力头大小 # 注意力头维度
    rotary_dim: int,  # 旋转维度 # 旋转编码维度
    max_position: int,  # 最大位置数 # 最大位置数
    base: int,  # 旋转基频 # 旋转基频
    is_neox_style: bool = True,  # 是否为NeoX风格 # 是否NeoX风格
    rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置 # RoPE缩放配置
    dtype: Optional[torch.dtype] = None,  # 数据类型 # 数据类型
    partial_rotary_factor: float = 1.0,  # 部分旋转因子 # 部分旋转因子
    dual_chunk_attention_config: Optional[Dict[str, Any]] = None,  # 双块注意力配置 # 双块注意力配置
) -> RotaryEmbedding:
    if dtype is None:  # 未指定数据类型则使用默认 # 未指定类型使用默认
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:  # 处理rope_scaling配置 # 处理缩放配置
        rope_scaling_tuple = {  # 将列表值转为元组以便哈希 # 将列表转为元组
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())  # 转为元组作为缓存键 # 转为元组作为缓存键
    else:
        rope_scaling_args = None  # 无缩放配置 # 无缩放配置

    if dual_chunk_attention_config is not None:  # 处理双块注意力配置 # 处理双块注意力配置
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v  # 将列表值转为元组 # 将列表转为元组
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"  # 排除稀疏注意力配置 # 排除稀疏注意力配置
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())  # 转为元组 # 转为元组
    else:
        dual_chunk_attention_args = None  # 无双块注意力配置 # 无双块注意力配置

    if partial_rotary_factor < 1.0:  # 部分旋转因子调整旋转维度 # 部分旋转因子调整维度
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (  # 构建缓存键 # 构建缓存键
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in _ROPE_DICT:  # 缓存命中则返回已有实例 # 缓存命中返回已有实例
        return _ROPE_DICT[key]

    if dual_chunk_attention_config is not None:  # 双块注意力配置 # 双块注意力配置
        extra_kwargs = {
            k: v
            for k, v in dual_chunk_attention_config.items()
            if k in ("chunk_size", "local_size")  # 提取chunk_size和local_size # 提取相关参数
        }
        rotary_emb = DualChunkRotaryEmbedding(  # 创建双块旋转编码 # 创建双块旋转编码
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            **extra_kwargs,
        )
    elif rope_scaling is None:  # 无缩放配置，使用基础RoPE # 无缩放配置使用基础RoPE
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:  # 有缩放配置 # 有缩放配置
        if "rope_type" in rope_scaling:  # 获取缩放类型 # 获取缩放类型
            scaling_type = rope_scaling["rope_type"]
        elif "type" in rope_scaling:  # 兼容旧版type字段 # 兼容旧版type字段
            scaling_type = rope_scaling["type"]
        else:
            raise ValueError(  # 未找到缩放类型则报错 # 未找到缩放类型报错
                f"Unknown RoPE scaling type, rope_scaling is {rope_scaling}"
            )

        if scaling_type == "llama3":  # Llama3缩放类型 # Llama3缩放类型
            scaling_factor = _get_rope_param(rope_scaling, "factor", 1.0, scaling_type)  # 获取缩放因子 # 获取缩放因子
            low_freq_factor = _get_rope_param(  # 获取低频因子 # 获取低频因子
                rope_scaling, "low_freq_factor", 1.0, scaling_type
            )
            high_freq_factor = _get_rope_param(  # 获取高频因子 # 获取高频因子
                rope_scaling, "high_freq_factor", 4.0, scaling_type
            )
            original_max_position = _get_rope_param(  # 获取原始最大位置数 # 获取原始最大位置数
                rope_scaling,
                "original_max_position_embeddings",
                max_position,
                scaling_type,
            )
            rotary_emb = Llama3RotaryEmbedding(  # 创建Llama3旋转编码 # 创建Llama3旋转编码
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            )
        elif scaling_type == "default":  # 默认缩放类型 # 默认缩放类型
            if "mrope_section" in rope_scaling:  # 多模态RoPE配置 # 多模态RoPE配置
                rotary_emb = MRotaryEmbedding(  # 创建多模态旋转编码 # 创建多模态旋转编码
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                    mrope_interleaved_glm=rope_scaling.get(
                        "mrope_interleaved_glm", False
                    ),
                )
            elif rope_scaling.get("use_fope", False):  # Fourier位置编码配置 # Fourier位置编码配置
                rotary_emb = FourierRotaryEmbedding(  # 创建Fourier旋转编码 # 创建Fourier旋转编码
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    num_kv_heads=rope_scaling["num_kv_heads"],
                    fope_init_factor=rope_scaling.get("fope_init_factor", 0.1),
                    fope_sep_head=rope_scaling.get("fope_sep_head", True),
                    num_inv_freq=rope_scaling.get("num_inv_freq", None),
                )
            else:  # 默认使用基础RoPE # 默认使用基础RoPE
                rotary_emb = RotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                )
        elif scaling_type == "linear":  # 线性缩放类型 # 线性缩放类型
            scaling_factor = _get_rope_param(rope_scaling, "factor", 1.0, scaling_type)  # 获取缩放因子 # 获取缩放因子
            rotary_emb = LinearScalingRotaryEmbedding(  # 创建线性缩放旋转编码 # 创建线性缩放旋转编码
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        elif scaling_type == "dynamic":  # 动态NTK缩放类型 # 动态NTK缩放类型
            scaling_factor = _get_rope_param(rope_scaling, "factor", 1.0, scaling_type)  # 获取缩放因子 # 获取缩放因子
            if "alpha" in rope_scaling:  # 有alpha参数使用带alpha的动态NTK # 有alpha参数
                rotary_emb = DynamicNTKAlphaRotaryEmbedding(  # 创建带alpha的动态NTK旋转编码 # 创建带alpha的动态NTK
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    rope_scaling["alpha"],
                    dtype,
                )
            else:  # 无alpha参数使用标准动态NTK # 无alpha参数
                rotary_emb = DynamicNTKScalingRotaryEmbedding(  # 创建动态NTK缩放旋转编码 # 创建动态NTK缩放旋转编码
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                )
        elif scaling_type == "yarn":  # YaRN缩放类型 # YaRN缩放类型
            scaling_factor = _get_rope_param(rope_scaling, "factor", 1.0, scaling_type)  # 获取缩放因子 # 获取缩放因子
            original_max_position = _get_rope_param(  # 获取原始最大位置数 # 获取原始最大位置数
                rope_scaling,
                "original_max_position_embeddings",
                max_position,
                scaling_type,
            )
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")  # 提取YaRN特定参数 # 提取YaRN参数
            }
            extra_kwargs["truncate"] = rope_scaling.get("truncate", True)  # 获取截断参数 # 获取截断参数
            if "mrope_section" in rope_scaling:  # YaRN+MRoPE组合 # YaRN+MRoPE组合
                rotary_emb = YaRNScalingMRotaryEmbedding(  # 创建YaRN缩放多模态旋转编码 # 创建YaRN多模态旋转编码
                    head_size,
                    rotary_dim,
                    original_max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                    **extra_kwargs,
                )
            else:  # 纯YaRN缩放 # 纯YaRN缩放
                rotary_emb = YaRNScalingRotaryEmbedding(  # 创建YaRN缩放旋转编码 # 创建YaRN缩放旋转编码
                    head_size,
                    rotary_dim,
                    original_max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                    **extra_kwargs,
                )
        elif scaling_type == "deepseek_yarn":  # DeepSeek YaRN缩放类型 # DeepSeek YaRN缩放类型
            scaling_factor = _get_rope_param(rope_scaling, "factor", 1.0, scaling_type)  # 获取缩放因子 # 获取缩放因子
            original_max_position = _get_rope_param(  # 获取原始最大位置数 # 获取原始最大位置数
                rope_scaling,
                "original_max_position_embeddings",
                max_position,
                scaling_type,
            )
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",  # 提取DeepSeek YaRN特定参数 # 提取DeepSeek YaRN参数
                )
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(  # 创建DeepSeek缩放旋转编码 # 创建DeepSeek缩放旋转编码
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
        elif scaling_type == "longrope":  # LongRoPE缩放类型 # LongRoPE缩放类型
            short_factor = rope_scaling["short_factor"]  # 短因子 # 短因子
            long_factor = rope_scaling["long_factor"]  # 长因子 # 长因子
            original_max_position = _get_rope_param(  # 获取原始最大位置数 # 获取原始最大位置数
                rope_scaling,
                "original_max_position_embeddings",
                max_position,
                scaling_type,
            )
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")  # 提取LongRoPE缩放参数 # 提取LongRoPE参数
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(  # 创建Phi3 LongRoPE缩放旋转编码 # 创建Phi3 LongRoPE旋转编码
                head_size,
                rotary_dim,
                max_position,
                original_max_position,
                base,
                is_neox_style,
                dtype,
                short_factor,
                long_factor,
                **extra_kwargs,
            )
        elif scaling_type == "proportional":  # 比例缩放类型 # 比例缩放类型
            rotary_emb = Gemma4RotaryEmbedding(  # 创建Gemma4旋转编码 # 创建Gemma4旋转编码
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")  # 未知缩放类型报错 # 未知缩放类型报错
    _ROPE_DICT[key] = rotary_emb  # 缓存实例 # 缓存实例
    return rotary_emb  # 返回旋转编码实例 # 返回实例


def get_rope_cpu(  # 创建CPU平台RoPE实例的工厂函数 # 创建CPU平台RoPE实例
    head_size: int,  # 注意力头大小 # 注意力头维度
    rotary_dim: int,  # 旋转维度 # 旋转编码维度
    max_position: int,  # 最大位置数 # 最大位置数
    base: int,  # 旋转基频 # 旋转基频
    is_neox_style: bool = True,  # 是否为NeoX风格 # 是否NeoX风格
    rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置 # RoPE缩放配置
    dtype: Optional[torch.dtype] = None,  # 数据类型 # 数据类型
    partial_rotary_factor: float = 1.0,  # 部分旋转因子 # 部分旋转因子
    device: Optional[str] = None,  # 设备 # 设备
) -> RotaryEmbedding:
    if dtype is None:  # 未指定数据类型则使用默认 # 未指定类型使用默认
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:  # 处理rope_scaling配置 # 处理缩放配置
        rope_scaling_tuple = {  # 将列表值转为元组以便哈希 # 将列表转为元组
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())  # 转为元组作为缓存键 # 转为元组作为缓存键
    else:
        rope_scaling_args = None  # 无缩放配置 # 无缩放配置
    if partial_rotary_factor < 1.0:  # 部分旋转因子调整旋转维度 # 部分旋转因子调整维度
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (  # 构建缓存键 # 构建缓存键
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:  # 缓存命中则返回已有实例 # 缓存命中返回已有实例
        return _ROPE_DICT[key]

    assert rope_scaling is not None  # CPU平台必须有缩放配置 # CPU平台必须有缩放配置
    scaling_type = rope_scaling["rope_type"]  # 获取缩放类型 # 获取缩放类型
    assert (  # 断言仅支持deepseek_yarn # 断言仅支持deepseek_yarn
        scaling_type == "deepseek_yarn"
    ), "Only deepseek_yarn is supported for CPU for now"

    scaling_factor = _get_rope_param(rope_scaling, "factor", 1.0, scaling_type)  # 获取缩放因子 # 获取缩放因子
    original_max_position = _get_rope_param(  # 获取原始最大位置数 # 获取原始最大位置数
        rope_scaling, "original_max_position_embeddings", max_position, scaling_type
    )
    extra_kwargs = {
        k: v
        for k, v in rope_scaling.items()
        if k
        in (
            "extrapolation_factor",
            "attn_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",  # 提取DeepSeek YaRN特定参数 # 提取DeepSeek YaRN参数
        )
    }
    extra_kwargs["device"] = device  # 添加设备参数 # 添加设备参数
    rotary_emb = DeepseekScalingRotaryEmbedding(  # 创建DeepSeek缩放旋转编码 # 创建DeepSeek缩放旋转编码
        head_size,
        rotary_dim,
        original_max_position,
        base,
        is_neox_style,
        scaling_factor,
        dtype,
        **extra_kwargs,
    )
    _ROPE_DICT[key] = rotary_emb  # 缓存实例 # 缓存实例
    return rotary_emb  # 返回旋转编码实例 # 返回实例


def get_rope_wrapper(  # RoPE工厂函数的包装器，根据设备选择不同的实现 # RoPE工厂函数包装器
    head_size: int,  # 注意力头大小 # 注意力头维度
    rotary_dim: int,  # 旋转维度 # 旋转编码维度
    max_position: int,  # 最大位置数 # 最大位置数
    base: int,  # 旋转基频 # 旋转基频
    is_neox_style: bool = True,  # 是否为NeoX风格 # 是否NeoX风格
    rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置 # RoPE缩放配置
    dtype: Optional[torch.dtype] = None,  # 数据类型 # 数据类型
    partial_rotary_factor: float = 1.0,  # 部分旋转因子 # 部分旋转因子
    device: Optional[str] = None,  # 设备 # 设备
):
    if device != "cpu":  # 非CPU设备 # 非CPU设备
        wrapper = aiter_get_rope if _use_aiter else get_rope  # HIP+AITER用aiter否则用get_rope # 选择实现
        return wrapper(  # 调用选定的工厂函数 # 调用工厂函数
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor,
        )

    return get_rope_cpu(  # CPU设备使用get_rope_cpu # CPU设备使用CPU工厂函数
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling,
        dtype,
        partial_rotary_factor,
        device,
    )
