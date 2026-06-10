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
# HF Transformers共享工具模块，提供配置注册、模型下载、RoPE配置提取、上下文长度计算等通用功能
# 被config、tokenizer和processor子模块共同使用
"""Shared helpers used by config, tokenizer, and processor modules."""  # 配置、分词器和处理器模块共享的辅助工具

import json  # 导入JSON解析模块
import os  # 导入操作系统接口模块
from pathlib import Path  # 导入路径处理模块
from typing import Any, Dict, Optional, Type, Union  # 导入类型注解

import torch  # 导入PyTorch张量库
from huggingface_hub import snapshot_download  # 导入HuggingFace Hub快照下载工具

from sglang.srt.configs import (  # 导入SGLang自定义模型配置类
    AfmoeConfig,
    BailingHybridConfig,
    ChatGLMConfig,
    DbrxConfig,
    DeepseekVL2Config,
    DotsOCRConfig,
    DotsVLMConfig,
    ExaoneConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiK25Config,
    KimiLinearConfig,
    KimiVLConfig,
    LagunaConfig,
    LongcatFlashConfig,
    MiniCPMV4_6Config,
    MiniCPMV4_6VisionConfig,
    MultiModalityConfig,
    NemotronH_Nano_Omni_Reasoning_V3_Config,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    NemotronHPuzzleConfig,
    Olmo3Config,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
    Step3p5Config,
    Step3p7Config,
    Step3VLConfig,
)
from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config  # 导入DeepSeek OCR V2配置
from sglang.srt.configs.internvl import InternVLChatConfig  # 导入InternVL聊天配置
from sglang.srt.utils import get_bool_env_var, logger, lru_cache_frozenset  # 导入通用工具
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri  # 导入RunAI对象存储工具

from ..hf_transformers_patches import normalize_rope_scaling_compat  # 导入RoPE缩放兼容性归一化

if get_bool_env_var("SGLANG_USE_MODELSCOPE"):  # 如果启用了ModelScope
    from modelscope import AutoConfig, GenerationConfig  # 从ModelScope导入
else:  # 否则使用HuggingFace
    from transformers import AutoConfig, GenerationConfig  # 从transformers导入

from transformers import PretrainedConfig  # 导入预训练配置基类

# ---------------------------------------------------------------------------
# Config registry  # 配置注册表
# ---------------------------------------------------------------------------

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {  # 模型类型到配置类的注册表
    cls.model_type: cls  # 使用类的model_type属性作为键
    for cls in [
        AfmoeConfig,
        BailingHybridConfig,
        ChatGLMConfig,
        DbrxConfig,
        ExaoneConfig,
        DeepseekVL2Config,
        MultiModalityConfig,
        KimiVLConfig,
        InternVLChatConfig,
        LagunaConfig,
        Step3VLConfig,
        LongcatFlashConfig,
        Olmo3Config,
        KimiLinearConfig,
        Qwen3NextConfig,
        FalconH1Config,
        GraniteMoeHybridConfig,
        DotsVLMConfig,
        DotsOCRConfig,
        NemotronH_Nano_VL_V2_Config,
        NemotronH_Nano_Omni_Reasoning_V3_Config,
        NemotronHConfig,
        NemotronHPuzzleConfig,
        DeepseekVLV2Config,
        Qwen3_5Config,
        Qwen3_5MoeConfig,
        InternS2PreviewConfig,
        JetNemotronConfig,
        JetVLMConfig,
        KimiK25Config,
        Step3p5Config,
        Step3p7Config,
        MiniCPMV4_6Config,
        MiniCPMV4_6VisionConfig,
    ]
}

# DeepSeek V3.2 / V4 reuse the V3 config schema. Subclass the upstream  # DeepSeek V3.2/V4复用V3配置模式。子类化上游
# transformers class with each model_type so AutoConfig.register passes its  # transformers类并设置各自的model_type，以便AutoConfig.register通过
# consistency check (which requires class.model_type == registered key).  # 一致性检查（要求class.model_type == 注册键）
# Default-value divergences (e.g. V4's topk_group) are handled in  # 默认值差异（如V4的topk_group）在
# model_config.py post-load.  # model_config.py的加载后处理中处理
try:  # 尝试导入DeepSeek V3配置
    from transformers import DeepseekV3Config as _HFDeepseekV3Config  # 导入HF的DeepSeek V3配置

    class _DeepseekV32ConfigAlias(_HFDeepseekV3Config):  # DeepSeek V3.2配置别名
        model_type = "deepseek_v32"  # 设置模型类型

    class _DeepseekV4ConfigAlias(_HFDeepseekV3Config):  # DeepSeek V4配置别名
        model_type = "deepseek_v4"  # 设置模型类型

    _CONFIG_REGISTRY["deepseek_v32"] = _DeepseekV32ConfigAlias  # 注册V3.2配置
    _CONFIG_REGISTRY["deepseek_v4"] = _DeepseekV4ConfigAlias  # 注册V4配置

    # For kimi_k25_eagle3  # 用于kimi_k25_eagle3
    class _KimiK2ConfigAlias(_HFDeepseekV3Config):  # Kimi K2配置别名
        model_type = "kimi_k2"  # 设置模型类型

    _CONFIG_REGISTRY["kimi_k2"] = _KimiK2ConfigAlias  # 注册Kimi K2配置
except ImportError:  # 如果导入失败
    pass  # 忽略

for name, cls in _CONFIG_REGISTRY.items():  # 遍历注册表，注册所有配置类
    try:  # 尝试注册
        AutoConfig.register(name, cls)  # 向AutoConfig注册配置类
    except ValueError as e:  # 如果注册失败
        err = str(e).lower()  # 获取错误消息
        if "already registered" not in err and "already used" not in err:  # 如果不是已注册错误
            logger.warning("Failed to register config %s: %s", name, e)  # 记录注册失败警告


# ---------------------------------------------------------------------------
# Download / path helpers  # 下载/路径辅助工具
# ---------------------------------------------------------------------------


def download_from_hf(  # 从HuggingFace Hub下载模型文件
    model_path: str,  # 模型路径
    allow_patterns: Optional[Union[str, list]] = None,  # 允许的文件模式
):
    if os.path.exists(model_path):  # 如果路径已存在本地
        return model_path  # 直接返回本地路径

    if not allow_patterns:  # 如果未指定允许模式
        allow_patterns = ["*.json", "*.bin", "*.model"]  # 默认允许JSON、二进制和模型文件

    return snapshot_download(model_path, allow_patterns=allow_patterns)  # 使用快照下载


def resolve_runai_obj_uri(model_name_or_path: str) -> str:  # 解析RunAI对象存储URI到本地路径
    if is_runai_obj_uri(model_name_or_path):  # 如果是RunAI对象存储URI
        return ObjectStorageModel.get_path(model_name_or_path)  # 返回本地缓存路径
    return model_name_or_path  # 非对象存储URI直接返回


def _resolve_local_or_cached_file(model_name_or_path, filename, revision=None):  # 从本地目录或HF Hub缓存解析文件（无需网络）
    """Resolve a file from a local directory or HF hub cache (no network)."""  # 从本地目录或HF Hub缓存解析文件（无需网络）
    local_path = Path(model_name_or_path) / filename  # 构造本地文件路径
    if local_path.is_file():  # 如果本地文件存在
        return str(local_path)  # 返回本地路径字符串
    from huggingface_hub import hf_hub_download  # 导入HF Hub下载工具

    return hf_hub_download(  # 从HF Hub缓存下载
        model_name_or_path, filename, revision=revision, local_files_only=True  # 仅使用本地文件
    )


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:  # 检查文件是否为GGUF格式
    model = Path(model)  # 转换为Path对象
    if not model.is_file():  # 如果不是文件
        return False  # 返回False
    elif model.suffix == ".gguf":  # 如果扩展名为.gguf
        return True  # 返回True

    with open(model, "rb") as f:  # 以二进制模式打开文件
        header = f.read(4)  # 读取前4字节
    return header == b"GGUF"  # 检查是否为GGUF魔数


# ---------------------------------------------------------------------------
# Rope / text config helpers  # RoPE/文本配置辅助工具
# ---------------------------------------------------------------------------


def get_rope_config(config):  # 从配置中获取RoPE theta和参数，支持v4和v5版本
    """Get (rope_theta, rope_params) from config, supporting both v4 and v5.  # 从配置获取(rope_theta, rope_params)，支持v4和v5

    Trust-remote-code configs or parent configs passed to sub-models may not  # 信任远程代码的配置或传递给子模型的父配置可能
    have the v5 ``rope_parameters`` property, so we fall back to the v4-style  # 没有v5的rope_parameters属性，因此回退到v4风格的
    ``config.rope_theta`` / ``config.rope_scaling`` attributes.  # config.rope_theta / config.rope_scaling属性

    Returns:  # 返回值
        (rope_theta, rope_params): In v5, rope_params is the full  # (rope_theta, rope_params)：v5中rope_params是完整的
        rope_parameters dict (which subsumes rope_scaling and includes  # rope_parameters字典（包含rope_scaling和
        rope_theta). In v4, rope_params is the rope_scaling dict or None.  # rope_theta）。v4中rope_params是rope_scaling字典或None
    """
    rope_params = getattr(config, "rope_parameters", None)  # 尝试获取v5风格参数
    if rope_params is not None:  # 如果有v5参数
        return rope_params["rope_theta"], rope_params  # 返回theta和完整参数
    return getattr(config, "rope_theta", 10000), getattr(config, "rope_scaling", None)  # 回退到v4风格


def _patch_text_config(parent_config: PretrainedConfig, text_config):  # 同步父配置和文本子配置之间的标准属性
    """Synchronize standard attributes between parent config and text sub-config.  # 同步父配置和文本子配置之间的标准属性

    In transformers v5, the "untangle config" refactor removed automatic  # 在transformers v5中，"解耦配置"重构移除了自动
    inheritance of top-level PretrainedConfig attributes (pad_token_id,  # 继承顶层PretrainedConfig属性（pad_token_id、
    tie_word_embeddings, etc.) from sub-configs. Downstream code expects  # tie_word_embeddings等）。下游代码期望
    these attributes to be present on both configs (some models pass the  # 这些属性在两个配置上都存在（某些模型传递
    parent directly to the language model, others pass the text sub-config),  # 父配置给语言模型，其他传递文本子配置）
    so we propagate in both directions when an attribute is missing.  # 因此当属性缺失时我们在两个方向传播
    (See https://github.com/huggingface/transformers/pull/41541)  # 参见HF PR #41541
    """
    _ATTRS_TO_PROPAGATE = [  # 需要传播的属性列表
        "pad_token_id",  # 填充token ID
        "bos_token_id",  # 起始token ID
        "eos_token_id",  # 结束token ID
        "tie_word_embeddings",  # 是否共享词嵌入
    ]
    for attr in _ATTRS_TO_PROPAGATE:  # 遍历需要传播的属性
        parent_has = hasattr(parent_config, attr)  # 父配置是否有该属性
        text_has = hasattr(text_config, attr)  # 文本配置是否有该属性
        if parent_has and not text_has:  # 父有而子没有
            setattr(text_config, attr, getattr(parent_config, attr))  # 从父传播到子
        elif text_has and not parent_has:  # 子有而父没有
            setattr(parent_config, attr, getattr(text_config, attr))  # 从子传播到父
    return text_config  # 返回文本配置


def get_hf_text_config(config: PretrainedConfig):  # 获取多模态模型中与LLM相关的"子"配置
    """Get the "sub" config relevant to llm for multi modal models.  # 获取多模态模型中与LLM相关的子配置
    No op for pure text models.  # 纯文本模型不做操作
    """
    if config.architectures is not None:  # 如果有架构信息
        class_name = config.architectures[0]  # 获取架构类名
        if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):  # 如果是LLaVA模型
            # We support non-hf version of llava models, so we do not want to  # 我们支持非HF版本的LLaVA模型，因此不希望
            # read the wrong values from the unused default text_config.  # 从未使用的默认text_config读取错误值
            # NOTE(HandH1998): We set `torch_dtype` of config to `torch.float16` for the weights, as  # 设置config的dtype为float16
            # `torch.float16` is default used for image features in `python/sglang/srt/models/llava.py`.  # 因为float16是LLaVA模型中图像特征的默认数据类型
            setattr(config, "dtype", torch.float16)  # 设置dtype为float16
            return config  # 直接返回配置

    text_config = None  # 初始化文本配置

    # Some models (e.g. DeepSeek-OCR) store sub-configs as plain dicts.  # 某些模型（如DeepSeek-OCR）将子配置存储为普通字典
    # Convert to PretrainedConfig early so hasattr() checks and asserts work.  # 尽早转换为PretrainedConfig以使hasattr()检查和断言正常工作
    parent_dtype = getattr(config, "dtype", None)  # 获取父配置的dtype
    for _attr in ("text_config", "llm_config", "language_config", "thinker_config"):  # 遍历可能的子配置属性名
        _sub = getattr(config, _attr, None)  # 获取子配置
        if isinstance(_sub, dict):  # 如果子配置是字典
            _converted = PretrainedConfig(**_sub)  # 转换为PretrainedConfig
            if getattr(_converted, "dtype", None) is None and parent_dtype is not None:  # 如果转换后无dtype
                _converted.dtype = parent_dtype  # 从父配置继承dtype
            setattr(config, _attr, _converted)  # 设置转换后的子配置
        elif _sub is not None and parent_dtype is not None:  # 如果子配置存在且父有dtype
            # transformers v5 multimodal configs (e.g. Mistral3Config) carry  # transformers v5多模态配置（如Mistral3Config）
            # `dtype` only on the top-level config, leaving the sub-configs at  # 仅在顶层配置携带dtype，子配置为
            # None. Without this, _get_and_verify_dtype falls back to float32  # None。没有此处理，会回退到float32
            # and then "auto" downcasts to float16, which overflows the Pixtral  # 然后"auto"降级为float16，导致Pixtral
            # vision tower on real images and produces NaN features.  # 视觉塔溢出并产生NaN特征
            if getattr(_sub, "dtype", None) is None:  # 如果子配置没有dtype
                _sub.dtype = parent_dtype  # 从父配置继承dtype

    # Priority: thinker_config > llm_config > language_config > text_config  # 优先级：thinker_config > llm_config > language_config > text_config
    if hasattr(config, "thinker_config"):  # 如果有thinker_config
        # qwen2.5 omni  # Qwen2.5全模态模型
        thinker_config = config.thinker_config  # 获取thinker配置
        if hasattr(thinker_config, "text_config"):  # 如果thinker有text_config
            setattr(  # 设置text_config的dtype
                thinker_config.text_config,
                "dtype",
                getattr(thinker_config, "dtype", None),  # 从thinker配置继承dtype
            )
            text_config = thinker_config.text_config  # 使用thinker的text_config
        else:  # thinker没有text_config
            text_config = thinker_config  # 直接使用thinker配置
    elif hasattr(config, "llm_config"):  # 如果有llm_config
        # PointsV1.5 Chat Model  # PointsV1.5聊天模型
        assert hasattr(config.llm_config, "num_attention_heads")  # 确保有attention_heads属性
        text_config = config.llm_config  # 使用llm_config
    elif hasattr(config, "language_config"):  # 如果有language_config
        text_config = config.language_config  # 使用language_config
    elif hasattr(config, "text_config"):  # 如果有text_config
        # The code operates under the assumption that text_config should have  # 代码假设text_config应该有
        # `num_attention_heads` (among others). Assert here to fail early  # num_attention_heads等属性。在此断言以尽早失败
        # if transformers config doesn't align with this assumption.  # 如果transformers配置与此假设不一致
        assert hasattr(config.text_config, "num_attention_heads")  # 确保有attention_heads属性
        text_config = config.text_config  # 使用text_config

    # Ensure rope_scaling dicts have "type" for remote-code compat (v5).  # 确保rope_scaling字典有"type"以兼容远程代码（v5）
    normalize_rope_scaling_compat(config)  # 归一化RoPE缩放兼容性

    if text_config is not None:  # 如果找到了文本配置
        return _patch_text_config(config, text_config)  # 修补并返回文本配置
    return config  # 纯文本模型直接返回配置


# ---------------------------------------------------------------------------
# Model-specific helpers  # 模型特定的辅助工具
# ---------------------------------------------------------------------------


def _ensure_sub_configs(config: PretrainedConfig, *attr_names: str) -> None:  # 将字典值的子配置就地转换为AutoConfig对象
    """Convert dict-valued sub-configs to proper AutoConfig objects in-place."""  # 将字典值的子配置就地转换为AutoConfig对象
    for attr in attr_names:  # 遍历属性名
        sub = getattr(config, attr, None)  # 获取子配置
        if sub is not None and isinstance(sub, dict):  # 如果子配置存在且为字典
            setattr(config, attr, AutoConfig.for_model(**sub))  # 转换为AutoConfig对象


def _is_deepseek_ocr_model(config: PretrainedConfig) -> bool:  # 判断配置是否为DeepSeek OCR模型
    # TODO: Remove this workaround once AutoConfig correctly identifies deepseek-ocr.  # 待办：AutoConfig正确识别deepseek-ocr后移除此变通
    # Hugging Face's AutoConfig currently misidentifies it as deepseekvl2.  # HF的AutoConfig目前将其误识别为deepseekvl2
    auto_map = getattr(config, "auto_map", None) or {}  # 获取auto_map
    return auto_map.get("AutoModel") == "modeling_deepseekocr.DeepseekOCRForCausalLM"  # 检查AutoModel映射


def _is_deepseek_ocr2_model(config: PretrainedConfig) -> bool:  # 判断配置是否为DeepSeek OCR2模型
    auto_map = getattr(config, "auto_map", None) or {}  # 获取auto_map
    return auto_map.get("AutoModel") == "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"  # 检查AutoModel映射


def _override_v_head_dim_if_zero(config: PretrainedConfig, patch: int = 128) -> None:  # 如果v_head_dim为0则覆盖为指定值
    patched = False  # 是否已修补标志
    for attr in ("text_config", "language_config"):  # 遍历可能的子配置属性
        sub = getattr(config, attr, None)  # 获取子配置
        if sub is None:  # 如果不存在
            continue  # 跳过
        if isinstance(sub, dict):  # 如果是字典类型
            if sub.get("v_head_dim") == 0:  # 如果v_head_dim为0
                sub["v_head_dim"] = patch  # 覆盖为指定值
                patched = True  # 标记已修补
        elif getattr(sub, "v_head_dim", None) == 0:  # 如果对象属性v_head_dim为0
            sub.v_head_dim = patch  # 覆盖为指定值
            patched = True  # 标记已修补
    if patched:  # 如果进行了修补
        logger.warning(  # 记录警告
            f"Overriding v_head_dim from 0 to {patch} to avoid potential issues."  # 显示覆盖信息
        )


# ---------------------------------------------------------------------------
# Context length / generation config / sparse attention  # 上下文长度/生成配置/稀疏注意力
# ---------------------------------------------------------------------------

# Models don't use the same configuration key for determining the maximum  # 模型不使用相同的配置键确定最大
# context length.  Store them here so we can sanely check them.  # 上下文长度。在此存储以便统一检查
# NOTE: The ordering here is important. Some models have two of these and we  # 注意：顺序很重要。某些模型有两个键，我们
# have a preference for which value gets used.  # 优先使用排在前面的值
CONTEXT_LENGTH_KEYS = [  # 上下文长度配置键列表（按优先级排序）
    "max_sequence_length",  # 最大序列长度
    "seq_length",  # 序列长度
    "max_seq_len",  # 最大序列长度
    "model_max_length",  # 模型最大长度
    "max_position_embeddings",  # 最大位置嵌入数
]


def get_context_length(config):  # 从HuggingFace模型配置获取上下文长度
    """Get the context length of a model from a huggingface model configs."""  # 从HuggingFace模型配置获取上下文长度
    text_config = config  # 使用配置作为文本配置
    rope_scaling = getattr(text_config, "rope_scaling", None)  # 获取RoPE缩放配置
    if rope_scaling:  # 如果有RoPE缩放
        rope_scaling_factor = rope_scaling.get("factor", 1)  # 获取缩放因子
        if "original_max_position_embeddings" in rope_scaling:  # 如果有原始最大位置嵌入
            rope_scaling_factor = 1  # 不使用缩放因子
        if rope_scaling.get("rope_type", None) == "llama3":  # 如果是LLaMA3类型
            rope_scaling_factor = 1  # 不使用缩放因子
    else:  # 没有RoPE缩放
        rope_scaling_factor = 1  # 缩放因子为1

    for key in CONTEXT_LENGTH_KEYS:  # 按优先级遍历上下文长度键
        val = getattr(text_config, key, None)  # 获取键值
        if val is not None:  # 如果找到值
            return int(rope_scaling_factor * val)  # 返回缩放后的上下文长度
    return 2048  # 默认上下文长度


@lru_cache_frozenset(maxsize=32)  # LRU缓存，最多32个条目
def get_generation_config(  # 获取模型的生成配置
    model: str,  # 模型名称或路径
    trust_remote_code: bool,  # 是否信任远程代码
    revision: Optional[str] = None,  # 模型版本
    **kwargs,  # 其他参数
):
    try:  # 尝试加载生成配置
        return GenerationConfig.from_pretrained(  # 从预训练模型加载生成配置
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except FileNotFoundError:  # 如果文件未找到
        return None  # 返回None
    except OSError as e:  # 如果操作系统错误
        logger.warning(  # 记录警告
            "Failed to load generation config for %s: %s. "
            "Proceeding without generation config.",  # 未找到生成配置，继续运行
            model,
            e,
        )
        return None  # 返回None


# Qwen-1M related  # Qwen-1M相关
def get_sparse_attention_config(  # 获取稀疏注意力配置
    model: str,  # 模型名称或路径
    sparse_attention_config_filename: str = "sparse_attention_config.json",  # 配置文件名
) -> Dict[str, Any]:
    is_local = os.path.isdir(model)  # 检查是否为本地目录
    if not is_local:  # 如果不是本地
        model = download_from_hf(model, allow_patterns=["*.json"])  # 下载JSON配置文件

    config_file = os.path.join(model, sparse_attention_config_filename)  # 构造配置文件路径
    if not os.path.exists(config_file):  # 如果配置文件不存在
        return {}  # 返回空字典

    with open(config_file) as f:  # 打开配置文件
        config = json.load(f)  # 解析JSON
    return config  # 返回配置字典


# ---------------------------------------------------------------------------
# Tokenizer / processor helpers  # 分词器/处理器辅助工具
# ---------------------------------------------------------------------------


# Some models don't have an available processor, e.g.: InternVL  # 某些模型没有可用的处理器，如InternVL
def get_tokenizer_from_processor(processor):  # 从处理器中提取分词器
    from transformers import PreTrainedTokenizerBase  # 导入分词器基类

    if isinstance(processor, PreTrainedTokenizerBase):  # 如果处理器本身就是分词器
        return processor  # 直接返回
    return processor.tokenizer  # 返回处理器的分词器属性


def attach_additional_stop_token_ids(tokenizer):  # 附加额外的停止token ID
    added = tokenizer.get_added_vocab()  # 获取已添加的词汇表
    if "<|eom_id|>" in added:  # 如果有eom_id token
        tokenizer.additional_stop_token_ids = {added["<|eom_id|>"]}  # 设置为停止token
    else:  # 否则
        tokenizer.additional_stop_token_ids = None  # 不设置额外停止token
