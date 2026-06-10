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
# 处理器加载工具，负责从HuggingFace加载多模态处理器
# 处理AutoProcessor失败时的手动构建、Mistral模型特殊处理和后加载修补
"""Processor loading utilities."""  # 处理器加载工具

import json  # 导入JSON解析模块
from pathlib import Path  # 导入路径处理模块
from typing import Optional  # 导入可选类型注解

from transformers import (  # 导入transformers处理器类
    AutoProcessor,  # 自动处理器
    AutoTokenizer,  # 自动分词器
    PreTrainedTokenizerBase,  # 预训练分词器基类
)

from sglang.srt.multimodal.customized_mm_processor_utils import _CUSTOMIZED_MM_PROCESSOR  # 导入自定义多模态处理器
from sglang.srt.utils import logger  # 导入日志记录器

from .common import (  # 从通用模块导入
    AutoConfig,  # 自动配置类
    _is_deepseek_ocr2_model,  # DeepSeek OCR2判断
    _is_deepseek_ocr_model,  # DeepSeek OCR判断
    _override_v_head_dim_if_zero,  # v_head_dim覆盖工具
    _resolve_local_or_cached_file,  # 本地/缓存文件解析
    attach_additional_stop_token_ids,  # 附加停止token ID
    download_from_hf,  # 从HF下载
    get_tokenizer_from_processor,  # 从处理器获取分词器
    resolve_runai_obj_uri,  # 解析RunAI对象URI
)
from .mistral_utils import (  # 从Mistral工具模块导入
    is_mistral_model,  # Mistral模型判断
    load_mistral_config,  # 加载Mistral配置
    patch_mistral_common_tokenizer,  # 修补MistralCommon分词器
    wrap_as_pixtral,  # 包装为Pixtral处理器
)
from .tokenizer import (  # 从分词器模块导入
    _TOKENIZERS_BACKEND,  # TokenizersBackend类名
    _fix_added_tokens_encoding,  # 修复添加的token编码
    _fix_special_tokens_pattern,  # 修复特殊token模式
)


def _build_processor_manually(  # 当AutoProcessor无法解析feature_extractor_type时手动构建处理器
    model_path, config, trust_remote_code, revision, **kwargs  # 模型路径、配置、信任远程代码、版本、其他参数
):
    """Build processor when AutoProcessor fails to resolve feature_extractor_type.  # AutoProcessor无法解析feature_extractor_type时手动构建处理器

    In transformers v5, AutoProcessor.from_pretrained calls  # 在transformers v5中，AutoProcessor.from_pretrained调用
    AutoFeatureExtractor.from_pretrained which fails if  # AutoFeatureExtractor.from_pretrained，如果
    preprocessor_config.json lacks 'feature_extractor_type'. This resolves  # preprocessor_config.json缺少feature_extractor_type则失败。此方法
    the processor class via dynamic module resolution and constructs it with  # 通过动态模块解析处理器类，并使用
    individually-loaded components.  # 单独加载的组件构造
    """
    import transformers  # 导入transformers模块
    from transformers import AutoImageProcessor, AutoTokenizer  # 导入图像处理器和分词器
    from transformers.dynamic_module_utils import get_class_from_dynamic_module  # 导入动态模块类获取工具

    # Resolve processor class from auto_map -- check both the model config  # 从auto_map解析处理器类——同时检查模型配置
    # and the preprocessor_config.json (some models like MiniCPM-o only  # 和preprocessor_config.json（某些模型如MiniCPM-o
    # declare AutoProcessor in the latter).  # 仅在后者中声明AutoProcessor）
    auto_map = getattr(config, "auto_map", None) or {}  # 获取auto_map
    proc_ref = auto_map.get("AutoProcessor")  # 从模型配置获取处理器引用
    if not proc_ref:  # 如果模型配置中没有
        try:  # 尝试从preprocessor_config.json获取
            pp_file = _resolve_local_or_cached_file(  # 解析preprocessor_config.json
                model_path, "preprocessor_config.json", revision
            )
            with open(pp_file) as f:  # 打开文件
                pp_auto_map = json.load(f).get("auto_map", {})  # 读取auto_map
            proc_ref = pp_auto_map.get("AutoProcessor")  # 获取处理器引用
        except (OSError, json.JSONDecodeError, ValueError) as e:  # 捕获异常
            logger.warning(  # 记录警告
                "_build_processor_manually: could not read preprocessor_config.json "
                "for %s: %s",  # 无法读取preprocessor_config.json
                model_path,
                e,
            )
    if not proc_ref:  # 如果仍未找到处理器引用
        raise ValueError(f"Cannot determine processor class for {model_path}")  # 抛出异常

    proc_cls = get_class_from_dynamic_module(  # 通过动态模块获取处理器类
        proc_ref, model_path, code_revision=revision
    )

    # Load sub-components individually (these succeed)  # 单独加载子组件（这些会成功）
    tokenizer = AutoTokenizer.from_pretrained(  # 加载分词器
        model_path, trust_remote_code=trust_remote_code, revision=revision
    )
    init_kwargs = {"tokenizer": tokenizer}  # 初始化参数：分词器

    if "image_processor" in getattr(proc_cls, "attributes", []):  # 如果处理器需要图像处理器
        try:
            init_kwargs["image_processor"] = AutoImageProcessor.from_pretrained(  # 加载图像处理器
                model_path, trust_remote_code=trust_remote_code, revision=revision
            )
        except (ImportError, OSError, ValueError) as e:  # 如果加载失败
            raise RuntimeError(  # 抛出运行时错误
                f"Failed to load image_processor for {model_path}: {e}. "
                f"This model requires an image processor for multimodal features. "
                f"Check that the model files are complete and accessible."  # 检查模型文件完整性和可访问性
            ) from e

    # Instantiate feature extractor from its declared class  # 从声明的类实例化特征提取器
    fe_class_name = getattr(proc_cls, "feature_extractor_class", None)  # 获取特征提取器类名
    if fe_class_name:  # 如果有声明
        fe_class = getattr(transformers, fe_class_name, None)  # 从transformers获取类
        if fe_class is not None:  # 如果找到类
            try:
                init_kwargs["feature_extractor"] = fe_class()  # 实例化特征提取器
            except TypeError as e:  # 如果实例化失败
                logger.warning(  # 记录警告
                    "Cannot instantiate feature extractor %s with no arguments "
                    "for %s: %s",  # 无法用无参数实例化特征提取器
                    fe_class_name,
                    model_path,
                    e,
                )
        else:  # 类未在transformers中找到
            logger.warning(  # 记录警告
                "Feature extractor class %s not found in transformers for %s",  # 特征提取器类在transformers中未找到
                fe_class_name,
                model_path,
            )

    return proc_cls(**init_kwargs)  # 使用初始化参数创建处理器实例


def get_processor(  # 获取多模态处理器
    tokenizer_name: str,  # 分词器名称
    *args,  # 位置参数
    tokenizer_mode: str = "auto",  # 分词器模式
    trust_remote_code: bool = False,  # 是否信任远程代码
    tokenizer_revision: Optional[str] = None,  # 分词器版本
    use_fast: Optional[bool] = True,  # 是否使用快速分词器
    tokenizer_backend: str = "huggingface",  # 分词器后端
    **kwargs,  # 其他参数
):
    if tokenizer_backend == "fastokens":  # 如果使用fastokens后端
        from .tokenizer import _ensure_fastokens_patched  # 导入fastokens补丁

        _ensure_fastokens_patched()  # 确保已应用fastokens补丁

    revision = kwargs.pop("revision", tokenizer_revision)  # 获取版本号
    tokenizer_name = resolve_runai_obj_uri(tokenizer_name)  # 解析RunAI对象URI

    if is_mistral_model(tokenizer_name):  # 如果是Mistral模型
        config = load_mistral_config(  # 加载Mistral配置
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    else:  # 非Mistral模型
        config = AutoConfig.from_pretrained(  # 从HF加载自动配置
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
    is_ocr2 = _is_deepseek_ocr2_model(config)  # 检查是否为DeepSeek OCR2
    if _is_deepseek_ocr_model(config) or is_ocr2:  # 如果是OCR模型
        config.model_type = "deepseek-ocr"  # 设置模型类型
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})  # 设置架构
        if is_ocr2:  # 如果是OCR2
            _override_v_head_dim_if_zero(config)  # 覆盖v_head_dim

    if config.model_type in {"qwen2_vl", "sarashina2_vision"}:  # 如果是Qwen2-VL或Sarashina2视觉模型
        if "size" not in kwargs:  # 如果未指定size参数
            kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}  # 设置默认size

    if config.model_type not in {"llava", "clip"}:  # 如果不是LLaVA或CLIP模型
        kwargs["use_fast"] = use_fast  # 设置use_fast参数
    try:  # 尝试加载处理器
        if "InternVL3_5" in tokenizer_name:  # 如果是InternVL3.5
            processor = AutoTokenizer.from_pretrained(  # 使用AutoTokenizer加载
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        else:  # 非InternVL3.5
            if config.model_type in _CUSTOMIZED_MM_PROCESSOR:  # 如果有自定义多模态处理器
                processor = _CUSTOMIZED_MM_PROCESSOR[config.model_type].from_pretrained(  # 使用自定义处理器
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            else:  # 使用标准AutoProcessor
                processor = AutoProcessor.from_pretrained(  # 使用AutoProcessor加载
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )

    except ValueError as e:  # 如果加载失败
        error_message = str(e)  # 获取错误消息
        if "does not have a slow version" in error_message:  # 如果没有慢速版本
            logger.info(  # 记录信息
                "Processor %s does not have a slow version. Automatically use fast version",  # 自动使用快速版本
                tokenizer_name,
            )
            kwargs["use_fast"] = True  # 强制使用快速版本
            processor = AutoProcessor.from_pretrained(  # 重新加载
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        elif "Unrecognized feature extractor" in error_message:  # 如果特征提取器无法识别
            logger.info(  # 记录信息
                "AutoProcessor failed on feature extractor for %s, "
                "constructing processor manually",  # 手动构建处理器
                tokenizer_name,
            )
            processor = _build_processor_manually(  # 手动构建处理器
                tokenizer_name,
                config,
                trust_remote_code,
                revision,
                **kwargs,
            )
        elif (  # 如果MistralCommon拒绝标准参数
            "are not supported by" in error_message and "MistralCommon" in error_message
        ):
            logger.info(  # 记录信息
                "AutoProcessor for %s rejected standard kwargs, "
                "retrying without trust_remote_code/use_fast",  # 去除不支持的参数重试
                tokenizer_name,
            )
            kwargs.pop("use_fast", None)  # 移除use_fast
            kwargs.pop("_from_auto", None)  # 移除_from_auto
            processor = AutoProcessor.from_pretrained(  # 重新加载
                tokenizer_name,
                *args,
                revision=revision,
                **kwargs,
            )
        else:  # 其他错误
            raise  # 重新抛出
    if (  # 如果处理器是分词器且配置为Pixtral
        isinstance(processor, PreTrainedTokenizerBase)
        and getattr(config, "model_type", None) == "pixtral"
    ):
        processor = wrap_as_pixtral(processor, config)  # 包装为Pixtral处理器

    tokenizer = get_tokenizer_from_processor(processor)  # 从处理器获取分词器

    # AutoProcessor may internally create a TokenizersBackend tokenizer  # AutoProcessor可能内部创建TokenizersBackend分词器
    # (same issue as get_tokenizer). Replace it with a properly loaded one.  # 与get_tokenizer相同的问题。替换为正确加载的分词器
    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:  # 如果分词器是TokenizersBackend
        from .tokenizer import get_tokenizer  # 导入get_tokenizer

        logger.warning(  # 记录警告
            "Processor tokenizer for %s is TokenizersBackend, "
            "reloading via get_tokenizer",  # 通过get_tokenizer重新加载
            tokenizer_name,
        )
        tokenizer = get_tokenizer(  # 使用get_tokenizer重新加载
            tokenizer_name,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=revision,
            tokenizer_backend=tokenizer_backend,
        )
        if isinstance(processor, PreTrainedTokenizerBase):  # 如果处理器本身就是分词器
            processor = tokenizer  # 替换为正确加载的分词器
        else:  # 处理器不是分词器
            processor.tokenizer = tokenizer  # 替换处理器的分词器属性

    if tokenizer.chat_template is None:  # 如果分词器缺少聊天模板
        local_path = download_from_hf(  # 下载模板文件
            tokenizer_name, allow_patterns=["*.json", "*.jinja", "*.model"]
        )
        jinja_path = Path(local_path) / "chat_template.jinja"  # 构造Jinja模板路径
        if jinja_path.is_file():  # 如果模板文件存在
            tokenizer.chat_template = jinja_path.read_text()  # 读取并设置模板
            logger.info("Loaded chat_template from %s", jinja_path)  # 记录模板加载

    patch_mistral_common_tokenizer(tokenizer)  # 修补MistralCommon分词器
    _fix_special_tokens_pattern(tokenizer)  # 修复特殊token模式
    _fix_added_tokens_encoding(tokenizer)  # 修复添加的token编码
    attach_additional_stop_token_ids(tokenizer)  # 附加停止token ID
    return processor  # 返回处理器
