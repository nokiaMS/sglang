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
# HuggingFace模型配置加载工具，支持HF和Mistral两种配置解析器
# 处理各种模型类型的配置覆盖和特殊化，包括DeepSeek OCR、Gemma4、Longcat等
"""Config loading utilities."""  # 配置加载工具

from pathlib import Path  # 导入路径处理模块
from typing import Optional  # 导入可选类型注解

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # 导入因果LM模型映射

from sglang.srt.configs.model_config_parser_registry import (  # 导入模型配置解析器注册表
    ModelConfigParserBase,  # 模型配置解析器基类
    get_model_config_parser,  # 获取解析器函数
    register_model_config_parser,  # 注册解析器装饰器
)
from sglang.srt.connector import create_remote_connector  # 导入远程连接器创建工具
from sglang.srt.utils import is_remote_url, lru_cache_frozenset  # 导入URL判断和LRU缓存工具

from ..hf_transformers_patches import _ensure_gguf_version  # 导入GGUF版本确保工具
from .common import (  # 从通用模块导入
    _CONFIG_REGISTRY,  # 配置注册表
    AutoConfig,  # 自动配置类
    DeepseekVLV2Config,  # DeepSeek VL V2配置
    _is_deepseek_ocr2_model,  # DeepSeek OCR2判断
    _is_deepseek_ocr_model,  # DeepSeek OCR判断
    _override_v_head_dim_if_zero,  # v_head_dim覆盖工具
    check_gguf_file,  # GGUF文件检查
    get_hf_text_config,  # 获取HF文本配置
    resolve_runai_obj_uri,  # 解析RunAI对象URI
)
from .mistral_utils import is_mistral_model, load_mistral_config  # 导入Mistral模型判断和配置加载


def _set_architectures(config, arch_name):  # 设置配置的架构字段
    config.update({"architectures": [arch_name]})  # 更新架构列表


def _apply_deepseek_ocr_overrides(config, model):  # 应用DeepSeek OCR配置覆盖
    _override_v_head_dim_if_zero(config)  # 覆盖v_head_dim
    _set_architectures(config, "DeepseekOCRForCausalLM")  # 设置架构为DeepseekOCR
    config._name_or_path = model  # 设置模型名称路径


@register_model_config_parser("hf")  # 注册为HF配置解析器
class HfModelConfigParser(ModelConfigParserBase):  # HuggingFace模型配置解析器
    def parse(  # 解析HF模型配置
        self,
        model,  # 模型名称或路径
        trust_remote_code: bool,  # 是否信任远程代码
        revision: Optional[str] = None,  # 模型版本
        **kwargs,  # 其他参数
    ):
        config = AutoConfig.from_pretrained(  # 从预训练模型加载自动配置
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

        if (  # 如果是Phi4MM模型
            config.architectures is not None
            and config.architectures[0] == "Phi4MMForCausalLM"  # Phi4MM多模态模型
        ):
            from transformers import SiglipVisionConfig  # 导入Siglip视觉配置

            config.vision_config = SiglipVisionConfig(  # 设置视觉配置
                hidden_size=1152,  # 隐藏层大小
                image_size=448,  # 图像大小
                intermediate_size=4304,  # 中间层大小
                model_type="siglip_vision_model",  # 模型类型
                num_attention_heads=16,  # 注意力头数
                num_hidden_layers=26,  # 隐藏层数
                patch_size=14,  # 补丁大小
            )

        if config.architectures in [  # 如果是Longcat系列架构
            ["LongcatCausalLM"],
            ["LongcatFlashForCausalLM"],
            ["LongcatFlashNgramForCausalLM"],
        ]:
            config.model_type = "longcat_flash"  # 设置模型类型为longcat_flash

        text_config = get_hf_text_config(config=config)  # 获取文本配置

        if isinstance(model, str) and text_config is not None:  # 如果模型是字符串且有文本配置
            items = (  # 获取配置项
                text_config.items()
                if hasattr(text_config, "items")  # 如果有items方法
                else vars(text_config).items()  # 否则使用__dict__
            )
            for key, val in items:  # 遍历配置项
                if not hasattr(config, key) and val is not None:  # 如果顶层配置缺少该属性
                    setattr(config, key, val)  # 从文本配置传播到顶层

        is_ocr = _is_deepseek_ocr_model(config)  # 检查是否为DeepSeek OCR模型
        is_ocr2 = _is_deepseek_ocr2_model(config)  # 检查是否为DeepSeek OCR2模型

        if is_ocr2:  # 如果是OCR2模型
            _override_v_head_dim_if_zero(config)  # 覆盖v_head_dim
            config.model_type = "deepseek-ocr"  # 设置模型类型
            _set_architectures(config, "DeepseekOCRForCausalLM")  # 设置架构
            config = DeepseekVLV2Config.from_pretrained(model, revision=revision)  # 从DeepseekVLV2配置加载
            _apply_deepseek_ocr_overrides(config, model)  # 应用OCR覆盖
        elif config.model_type in _CONFIG_REGISTRY:  # 如果模型类型在注册表中
            model_type = config.model_type  # 获取模型类型
            if model_type == "deepseek_vl_v2" and is_ocr:  # 如果是DeepSeek VL V2但实际是OCR
                model_type = "deepseek-ocr"  # 更正模型类型
            config = _CONFIG_REGISTRY[model_type].from_pretrained(  # 从注册的配置类加载
                model, revision=revision
            )

            # Re-check after reloading config from registry  # 从注册表重新加载配置后再次检查
            if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):  # 如果是OCR模型
                _apply_deepseek_ocr_overrides(config, model)  # 应用OCR覆盖
            else:  # 非OCR模型
                config._name_or_path = model  # 设置模型名称路径

        if isinstance(model, str) and config.model_type == "internvl_chat":  # 如果是InternVL聊天模型
            for key, val in config.llm_config.__dict__.items():  # 遍历LLM配置属性
                if not hasattr(config, key):  # 如果顶层缺少该属性
                    setattr(config, key, val)  # 从LLM配置传播到顶层

        if config.model_type == "multi_modality":  # 如果是多模态模型
            _set_architectures(config, "MultiModalityCausalLM")  # 设置架构

        if config.model_type in ("gemma4", "gemma4_assistant"):  # 如果是Gemma4模型
            # Gemma4 configs use base attributes for SWA layers and `global_*`  # Gemma4配置中基础属性用于SWA层，global_*变体
            # variants for full-attention layers.  SGLang expects the opposite:  # 用于全注意力层。SGLang期望相反：
            # base = full-attention, `swa_*` = sliding-window overrides.  # 基础=全注意力，swa_*=滑动窗口覆盖
            text_config = config.text_config  # 获取文本配置
            global_head_dim = getattr(text_config, "global_head_dim", None)  # 获取全局头维度
            global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)  # 获取全局KV头数

            swa_head_dim = text_config.head_dim  # 保存当前head_dim作为SWA头维度
            swa_kv_heads = text_config.num_key_value_heads  # 保存当前KV头数作为SWA KV头数

            text_config.swa_head_dim = swa_head_dim  # 设置SWA头维度
            text_config.swa_v_head_dim = swa_head_dim  # 设置SWA V头维度
            text_config.swa_num_key_value_heads = swa_kv_heads  # 设置SWA KV头数

            if global_head_dim is not None:  # 如果有全局头维度
                text_config.head_dim = global_head_dim  # 用全局头维度覆盖基础头维度
            if global_kv_heads is not None:  # 如果有全局KV头数
                text_config.num_key_value_heads = global_kv_heads  # 用全局KV头数覆盖基础KV头数

            if not hasattr(text_config, "v_head_dim"):  # 如果没有v_head_dim
                text_config.v_head_dim = text_config.head_dim  # 设置为head_dim
            if not hasattr(text_config, "swa_v_head_dim"):  # 如果没有swa_v_head_dim
                text_config.swa_v_head_dim = text_config.swa_head_dim  # 设置为swa_head_dim

        if config.model_type == "longcat_flash":  # 如果是Longcat Flash模型
            _set_architectures(config, "LongcatFlashForCausalLM")  # 设置架构

        return config  # 返回处理后的配置


@register_model_config_parser("mistral")  # 注册为Mistral配置解析器
class MistralModelConfigParser(ModelConfigParserBase):  # Mistral模型配置解析器
    def parse(  # 解析Mistral模型配置
        self,
        model,  # 模型名称或路径
        trust_remote_code: bool,  # 是否信任远程代码
        revision: Optional[str] = None,  # 模型版本
        **kwargs,  # 其他参数
    ):
        del kwargs  # 删除未使用的参数
        return load_mistral_config(  # 加载Mistral配置
            model, trust_remote_code=trust_remote_code, revision=revision  # 传入模型和版本
        )


@lru_cache_frozenset(maxsize=32)  # LRU缓存，最多32个条目
def get_config(  # 获取模型配置的统一入口
    model: str,  # 模型名称或路径
    trust_remote_code: bool,  # 是否信任远程代码
    revision: Optional[str] = None,  # 模型版本
    model_override_args: Optional[dict] = None,  # 模型覆盖参数
    model_config_parser: str = "auto",  # 配置解析器类型（auto/hf/mistral）
    **kwargs,  # 其他参数
):
    is_gguf = check_gguf_file(model)  # 检查是否为GGUF文件
    if is_gguf:  # 如果是GGUF文件
        if model_config_parser not in ("auto", "hf"):  # 如果解析器不兼容GGUF
            raise ValueError(  # 抛出异常
                f"model_config_parser={model_config_parser!r} is incompatible "
                "with GGUF inputs; only 'hf' (or 'auto') is supported."  # GGUF仅支持hf或auto解析器
            )
        _ensure_gguf_version()  # 确保GGUF版本支持
        kwargs["gguf_file"] = model  # 设置GGUF文件参数
        model = Path(model).parent  # 使用GGUF文件所在目录
        # Skip auto-resolution for GGUF: the name-based Mistral heuristic  # GGUF跳过自动解析：基于名称的Mistral启发式
        # would misfire on the rewritten parent dir.  # 会在重写的父目录上误判
        model_config_parser = "hf"  # 强制使用HF解析器

    model = resolve_runai_obj_uri(model)  # 解析RunAI对象存储URI

    if is_remote_url(model):  # 如果是远程URL
        client = create_remote_connector(model)  # 创建远程连接器
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])  # 拉取非权重文件
        model = client.get_local_dir()  # 获取本地目录

    if model_config_parser == "auto":  # 如果是自动选择解析器
        # `model` is post-rewrite (gguf parent / runai uri / remote pull).  # model已重写（gguf父目录/runai URI/远程拉取）
        model_config_parser = "mistral" if is_mistral_model(model) else "hf"  # Mistral模型用mistral解析器，其他用hf

    parser = get_model_config_parser(model_config_parser)  # 获取配置解析器
    config = parser.parse(  # 解析配置
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )

    if model_override_args:  # 如果有模型覆盖参数
        config.update(model_override_args)  # 更新配置

    if is_gguf:  # 如果是GGUF格式
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:  # 如果模型类型不在映射中
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")  # 抛出运行时错误
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])  # 设置GGUF架构

    return config  # 返回配置
