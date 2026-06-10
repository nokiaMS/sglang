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
# 分词器加载工具，负责从HuggingFace加载分词器并应用各种兼容性修补
# 处理transformers v5的TokenizersBackend回退、BOS/EOS token修复、特殊token编码修复等
"""Tokenizer loading utilities."""  # 分词器加载工具

import json  # 导入JSON解析模块
import logging  # 导入日志记录模块
import warnings  # 导入警告模块
from pathlib import Path  # 导入路径处理模块
from typing import Optional, Union  # 导入类型注解

from transformers import (  # 导入transformers分词器类
    AutoTokenizer,  # 自动分词器
    PreTrainedTokenizer,  # 预训练分词器
    PreTrainedTokenizerFast,  # 快速预训练分词器
)

from sglang.srt.connector import create_remote_connector  # 导入远程连接器创建工具
from sglang.srt.utils import is_remote_url, logger  # 导入URL判断和日志工具
from sglang.srt.utils.patch_tokenizer import patch_tokenizer  # 导入分词器补丁工具

from ..hf_transformers_patches import _ensure_gguf_version  # 导入GGUF版本确保工具
from .common import (  # 从通用模块导入
    _resolve_local_or_cached_file,  # 本地/缓存文件解析
    attach_additional_stop_token_ids,  # 附加停止token ID
    check_gguf_file,  # GGUF文件检查
    resolve_runai_obj_uri,  # 解析RunAI对象URI
)
from .mistral_utils import (  # 从Mistral工具模块导入
    _MISTRAL_TOKENIZER_REDIRECTS,  # Mistral分词器重定向
    patch_mistral_common_tokenizer,  # 修补MistralCommon分词器
    retry_without_mistral_common_kwargs,  # 去除MistralCommon拒绝参数重试
)

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.  # 预处理tokenizer.json的快速LLaMA分词器
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"  # 快速LLaMA分词器标识

# Class name used by transformers v5 when no tokenizer mapping exists for a model_type.  # transformers v5中当model_type无分词器映射时使用的类名
_TOKENIZERS_BACKEND = "TokenizersBackend"  # TokenizersBackend类名


def _load_tokenizer_by_declared_class(tokenizer_name, *args, **kwargs):  # 通过tokenizer_config.json中声明的类加载分词器
    """Load tokenizer by the class declared in tokenizer_config.json.  # 通过tokenizer_config.json中声明的类加载分词器

    AutoTokenizer resolves to TokenizersBackend when the model's config  # 当模型配置的model_type没有分词器类映射时，
    model_type has no tokenizer class mapping (e.g. deepseek_vl_v2), even  # AutoTokenizer会解析为TokenizersBackend，
    though tokenizer_config.json declares a standard class like  # 即使tokenizer_config.json声明了标准类如
    LlamaTokenizerFast.  Returns None if it cannot improve on AutoTokenizer.  # LlamaTokenizerFast。如果无法改进则返回None
    """
    import transformers  # 导入transformers模块

    try:  # 尝试读取tokenizer_config.json
        revision = kwargs.get("revision") or kwargs.get("tokenizer_revision")  # 获取版本号
        config_file = _resolve_local_or_cached_file(  # 解析tokenizer_config.json路径
            tokenizer_name, "tokenizer_config.json", revision
        )
        with open(config_file) as f:  # 打开配置文件
            tok_config = json.load(f)  # 解析JSON
        tok_class_name = tok_config.get("tokenizer_class")  # 获取声明的分词器类名
    except FileNotFoundError:  # 如果文件未找到
        return None  # 返回None
    except (OSError, json.JSONDecodeError) as e:  # 如果读取或解析失败
        logger.debug(  # 记录调试信息
            "Failed to read tokenizer_config.json for %s: %s", tokenizer_name, e  # 读取失败
        )
        return None  # 返回None

    if not tok_class_name:  # 如果没有声明分词器类
        return None  # 返回None

    # Skip base classes that don't implement required methods (e.g. get_vocab)  # 跳过未实现必需方法的基类
    if tok_class_name in ("PreTrainedTokenizer", "PreTrainedTokenizerBase"):  # 如果是基类
        return None  # 返回None

    tok_cls = getattr(transformers, tok_class_name, None)  # 从transformers获取分词器类
    if tok_cls is None and kwargs.get("trust_remote_code"):  # 如果类不在transformers中且启用远程代码
        # Class not in transformers — try loading via auto_map.  # 类不在transformers中——尝试通过auto_map加载
        try:  # 尝试动态加载
            auto_map = tok_config.get("auto_map", {})  # 获取auto_map
            auto_tok_ref = auto_map.get("AutoTokenizer")  # 获取AutoTokenizer引用
            if isinstance(auto_tok_ref, (list, tuple)):  # 如果引用是列表
                auto_tok_ref = auto_tok_ref[0]  # 取第一个元素
            if auto_tok_ref:  # 如果有引用
                from transformers.dynamic_module_utils import (  # 导入动态模块工具
                    get_class_from_dynamic_module,
                )

                tok_cls = get_class_from_dynamic_module(  # 从动态模块获取类
                    auto_tok_ref,
                    tokenizer_name,
                    code_revision=revision,
                )
        except (OSError, ImportError, ValueError, RuntimeError) as e:  # 捕获异常
            logger.debug("Dynamic module lookup for %s failed: %s", tok_class_name, e)  # 记录调试信息
    if tok_cls is None:  # 如果仍未找到类
        return None  # 返回None

    logger.debug(  # 记录调试信息
        "Loading tokenizer for %s directly as %s (bypassing AutoTokenizer)",  # 直接加载分词器（绕过AutoTokenizer）
        tokenizer_name,
        tok_class_name,
    )
    try:  # 尝试加载分词器
        return tok_cls.from_pretrained(tokenizer_name, *args, **kwargs)  # 使用声明的类加载
    except (OSError, ValueError, TypeError, ImportError) as e:  # 如果加载失败
        logger.warning(  # 记录警告
            "Direct load as %s failed for %s: %s. "
            "Falling back to AutoTokenizer result.",  # 直接加载失败，回退到AutoTokenizer
            tok_class_name,
            tokenizer_name,
            e,
        )
        return None  # 返回None


# Filter warnings like: https://github.com/sgl-project/sglang/issues/8082  # 过滤此类警告：sglang问题#8082
class TokenizerWarningsFilter(logging.Filter):  # 分词器警告过滤器
    def filter(self, record: logging.LogRecord) -> bool:  # 过滤日志记录
        return "Calling super().encode with" not in record.getMessage()  # 过滤特定警告消息


# ---------------------------------------------------------------------------
# Helpers for get_tokenizer  # get_tokenizer的辅助函数
# ---------------------------------------------------------------------------


def _resolve_tokenizer_name(tokenizer_name, kwargs):  # 解析特殊名称格式（GGUF、远程URL等）到本地路径
    """Resolve special name formats (GGUF, remote URLs, etc.) to a local path.  # 解析特殊名称格式（GGUF、远程URL等）到本地路径

    May mutate *kwargs* (e.g. to add ``gguf_file``).  # 可能修改kwargs（如添加gguf_file）
    """
    tokenizer_name = _MISTRAL_TOKENIZER_REDIRECTS.get(tokenizer_name, tokenizer_name)  # 应用Mistral分词器重定向

    if check_gguf_file(tokenizer_name):  # 如果是GGUF文件
        _ensure_gguf_version()  # 确保GGUF版本支持
        kwargs["gguf_file"] = tokenizer_name  # 设置gguf_file参数
        tokenizer_name = Path(tokenizer_name).parent  # 使用GGUF文件所在目录

    tokenizer_name = resolve_runai_obj_uri(tokenizer_name)  # 解析RunAI对象URI

    if is_remote_url(tokenizer_name):  # 如果是远程URL
        # BaseConnector implements __del__() to clean up the local dir.  # BaseConnector实现__del__()清理本地目录
        # Since config files need to exist all the time, so we DO NOT use  # 由于配置文件需要一直存在，我们不使用
        # with statement to avoid closing the client.  # with语句以避免关闭客户端
        client = create_remote_connector(tokenizer_name)  # 创建远程连接器
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])  # 拉取非权重文件
        tokenizer_name = client.get_local_dir()  # 获取本地目录

    return tokenizer_name  # 返回解析后的名称


def _auto_tokenizer_from_pretrained(tokenizer_name, *args, **common_kwargs):  # 调用AutoTokenizer.from_pretrained并处理错误
    """Call ``AutoTokenizer.from_pretrained`` with error handling."""  # 带错误处理的AutoTokenizer.from_pretrained调用
    try:  # 尝试加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
        logging.getLogger(tokenizer.__class__.__module__).addFilter(  # 添加警告过滤器
            TokenizerWarningsFilter()
        )
        return tokenizer  # 返回加载的分词器
    except TypeError as e:  # 如果类型错误
        err_msg = (  # 构造错误消息
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."  # 建议使用快速LLaMA分词器
        )
        raise RuntimeError(err_msg) from e  # 抛出运行时错误
    except ValueError as e:  # 如果值错误
        # MistralCommon tokenizers reject standard HF kwargs like  # MistralCommon分词器拒绝标准HF参数如
        # trust_remote_code, use_fast etc. Retry without them.  # trust_remote_code、use_fast等。去除后重试
        if "are not supported by" in str(e) and "MistralCommon" in str(e):  # 如果是MistralCommon参数拒绝
            return retry_without_mistral_common_kwargs(  # 去除拒绝的参数重试
                tokenizer_name, *args, **common_kwargs
            )
        # If the error pertains to the tokenizer class not existing or not  # 如果错误与分词器类不存在或
        # currently being imported, suggest using the --trust-remote-code flag.  # 未导入有关，建议使用--trust-remote-code
        if not common_kwargs.get("trust_remote_code") and (  # 如果未启用信任远程代码
            "does not exist or is not currently imported." in str(e)  # 类不存在
            or "requires you to execute the tokenizer file" in str(e)  # 需要执行分词器文件
        ):
            err_msg = (  # 构造错误消息
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."  # 建议启用信任远程代码
            )
            raise RuntimeError(err_msg) from e  # 抛出运行时错误
        raise  # 重新抛出其他值错误


def _resolve_tokenizers_backend(tokenizer_name, *args, **common_kwargs):  # 将通用的TokenizersBackend解析为正确的分词器类
    """Resolve generic ``TokenizersBackend`` to a proper tokenizer class.  # 将通用TokenizersBackend解析为正确的分词器类

    In transformers v5, ``AutoTokenizer`` falls back to ``TokenizersBackend``  # 在transformers v5中，AutoTokenizer回退到TokenizersBackend
    when the model_type has no tokenizer mapping.  This retries with  # 当model_type没有分词器映射时。此方法先用
    ``use_fast=False``, then attempts loading by the class declared in  # use_fast=False重试，然后尝试通过
    ``tokenizer_config.json``.  May still return a ``TokenizersBackend``  # tokenizer_config.json声明的类加载。可能仍返回
    if all retries fail (with a warning).  # TokenizersBackend（带警告）
    """
    logger.debug(  # 记录调试信息
        "Tokenizer loaded as generic TokenizersBackend for %s, "
        "retrying with use_fast=False",  # TokenizersBackend，用use_fast=False重试
        tokenizer_name,
    )
    common_kwargs = {**common_kwargs, "use_fast": False}  # 设置use_fast为False
    try:  # 尝试用慢速分词器加载
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as e:  # 捕获异常
        raise RuntimeError(  # 抛出运行时错误
            f"Retry with use_fast=False for {tokenizer_name} also failed "
            f"(initial load returned TokenizersBackend): {e}"  # 慢速分词器也加载失败
        ) from e

    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:  # 如果仍然是TokenizersBackend
        tokenizer = (  # 尝试通过声明的类加载
            _load_tokenizer_by_declared_class(tokenizer_name, *args, **common_kwargs)
            or tokenizer  # 如果失败则保留TokenizersBackend
        )

    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:  # 如果仍然是TokenizersBackend
        if common_kwargs.get("trust_remote_code"):  # 如果启用了信任远程代码
            logger.warning(  # 记录警告
                "Tokenizer for %s is still TokenizersBackend after retries "
                "with --trust-remote-code. Model-specific tokenizer attributes "
                "may be missing.",  # 信任远程代码后仍为TokenizersBackend，模型特定属性可能缺失
                tokenizer_name,
            )
        else:  # 未启用信任远程代码
            logger.debug(  # 记录调试信息
                "Tokenizer for %s loaded as generic TokenizersBackend. "
                "Set --trust-remote-code to load the model-specific tokenizer.",  # 设置--trust-remote-code以加载模型特定分词器
                tokenizer_name,
            )

    return tokenizer  # 返回分词器


# ---------------------------------------------------------------------------
# Post-load fixups  # 加载后修补
# ---------------------------------------------------------------------------


def _fix_v5_tokenizer_components(tokenizer, model_name_or_path, revision=None):  # 修复v5分词器类覆盖pre_tokenizer/decoder的问题
    """Fix pre_tokenizer/decoder when a v5 tokenizer class overwrites them.  # 修复v5分词器类覆盖pre_tokenizer/decoder的问题

    In transformers v5, some tokenizer classes (e.g. LlamaTokenizer) have a  # 在transformers v5中，某些分词器类（如LlamaTokenizer）有
    custom __init__ that rebuilds the pre_tokenizer and decoder from scratch  # 自定义__init__从头重建pre_tokenizer和decoder
    with class-specific components, discarding the originals from tokenizer.json.  # 使用类特定组件，丢弃tokenizer.json中的原始值
    This breaks models that specify LlamaTokenizerFast but actually use a  # 这破坏了指定LlamaTokenizerFast但实际使用
    different tokenizer architecture (e.g. DeepSeek-V3.2 uses ByteLevel).  # 不同分词器架构的模型（如DeepSeek-V3.2使用ByteLevel）

    Detects the mismatch by comparing against the raw tokenizer.json and  # 通过与原始tokenizer.json比较检测不匹配
    restores the original components when they differ.  # 在不同时恢复原始组件
    """
    backend = getattr(tokenizer, "_tokenizer", None)  # 获取底层tokenizer后端
    if backend is None:  # 如果没有后端
        return  # 直接返回

    try:  # 尝试加载原始tokenizer.json
        from tokenizers import Tokenizer as RawTokenizer  # 导入原始Tokenizer

        tok_file = _resolve_local_or_cached_file(  # 解析tokenizer.json路径
            model_name_or_path, "tokenizer.json", revision
        )
        raw = RawTokenizer.from_file(tok_file)  # 从文件加载
    except FileNotFoundError:  # 如果文件未找到
        return  # 直接返回
    except (OSError, ValueError, RuntimeError) as e:  # 如果加载出错
        logger.warning(  # 记录警告
            "_fix_v5_tokenizer_components: unexpected error loading tokenizer.json "
            "for %s, v5 component fix will not be applied: %s",  # v5组件修复不会应用
            model_name_or_path,
            e,
        )
        return  # 直接返回

    raw_pre = type(raw.pre_tokenizer).__name__ if raw.pre_tokenizer else None  # 获取原始pre_tokenizer类型
    loaded_pre = type(backend.pre_tokenizer).__name__ if backend.pre_tokenizer else None  # 获取加载的pre_tokenizer类型

    if raw_pre and loaded_pre and raw_pre != loaded_pre:  # 如果类型不匹配
        logger.info(  # 记录信息
            "Fixing v5 tokenizer component mismatch for %s: "
            "pre_tokenizer %s -> %s, decoder %s -> %s",  # 修复v5分词器组件不匹配
            model_name_or_path,
            loaded_pre,  # 加载的类型
            raw_pre,  # 原始类型
            type(backend.decoder).__name__ if backend.decoder else None,  # 加载的decoder类型
            type(raw.decoder).__name__ if raw.decoder else None,  # 原始decoder类型
        )
        backend.pre_tokenizer = raw.pre_tokenizer  # 恢复原始pre_tokenizer
        backend.decoder = raw.decoder  # 恢复原始decoder


def _fix_v5_add_bos_eos_token(tokenizer, model_name_or_path, revision=None):  # 恢复被transformers v5剥离的add_bos_token/add_eos_token
    """Restore add_bos_token/add_eos_token stripped by transformers v5.  # 恢复被transformers v5剥离的add_bos_token/add_eos_token

    In transformers v5, _from_pretrained() strips add_bos_token and  # 在transformers v5中，_from_pretrained()剥离add_bos_token和
    add_eos_token from init kwargs when a tokenizer.json file is present,  # add_eos_token当tokenizer.json文件存在时，
    assuming the tokenizer.json post-processor handles BOS/EOS addition.  # 假设tokenizer.json的post-processor处理BOS/EOS添加
    However, many models (e.g. DeepSeek-V3) have a tokenizer.json whose  # 然而，许多模型（如DeepSeek-V3）的tokenizer.json
    post-processor does NOT add BOS/EOS, and rely on the add_bos_token flag  # 的post-processor不添加BOS/EOS，而是依赖
    from tokenizer_config.json instead. This causes silent accuracy regressions.  # tokenizer_config.json的add_bos_token标志。这导致静默的精度回退

    This function reads the tokenizer_config.json and restores the values,  # 此函数读取tokenizer_config.json并恢复值
    but only for tokenizer classes that actually supported these flags in v4.  # 但仅对v4中实际支持这些标志的分词器类
    Classes like Qwen2Tokenizer did not support add_bos_token/add_eos_token  # Qwen2Tokenizer等类在v4中不支持add_bos_token/add_eos_token
    in v4, so restoring them would change behavior.  # 因此恢复它们会改变行为
    """
    # In transformers v4, only certain tokenizer classes supported  # 在transformers v4中，仅特定分词器类支持
    # add_bos_token / add_eos_token as init parameters.  Restoring these  # add_bos_token / add_eos_token作为初始化参数。恢复这些
    # flags for classes that never supported them (e.g. Qwen2Tokenizer)  # 标志给从未支持它们的类（如Qwen2Tokenizer）
    # would incorrectly change tokenization behavior.  # 会错误地改变分词行为
    _V4_CLASSES_WITH_BOS_EOS_FLAGS = frozenset(  # v4中支持BOS/EOS标志的类
        {
            "LlamaTokenizer",  # LLaMA分词器
            "LlamaTokenizerFast",  # LLaMA快速分词器
            "CodeLlamaTokenizer",  # CodeLLaMA分词器
            "CodeLlamaTokenizerFast",  # CodeLLaMA快速分词器
            "GemmaTokenizer",  # Gemma分词器
            "GemmaTokenizerFast",  # Gemma快速分词器
            "CohereTokenizerFast",  # Cohere快速分词器
        }
    )

    try:  # 尝试读取tokenizer_config.json
        config_file = _resolve_local_or_cached_file(  # 解析配置文件路径
            model_name_or_path, "tokenizer_config.json", revision
        )
        with open(config_file) as f:  # 打开配置文件
            config = json.load(f)  # 解析JSON
    except FileNotFoundError:  # 如果文件未找到
        return  # 直接返回
    except (OSError, json.JSONDecodeError, ValueError) as e:  # 如果读取失败
        logger.warning(  # 记录警告
            "_fix_v5_add_bos_eos_token: failed to read tokenizer_config.json "
            "for %s, BOS/EOS token restoration will not be applied: %s",  # BOS/EOS token恢复不会应用
            model_name_or_path,
            e,
        )
        return  # 直接返回

    tokenizer_class = config.get("tokenizer_class", "")  # 获取分词器类名
    if tokenizer_class not in _V4_CLASSES_WITH_BOS_EOS_FLAGS:  # 如果不是支持BOS/EOS标志的类
        logger.debug(  # 记录调试信息
            "_fix_v5_add_bos_eos_token: skipping %s (tokenizer_class=%s "
            "did not support add_bos/eos_token in v4)",  # 跳过（v4不支持add_bos/eos_token）
            model_name_or_path,
            tokenizer_class,
        )
        return  # 直接返回

    # In v4, Llama/Gemma tokenizers defaulted add_bos_token=True.  # 在v4中，Llama/Gemma分词器默认add_bos_token=True
    # When the config omits the key or has null, use the v4 default so that  # 当配置省略键或为null时，使用v4默认值以便
    # update_post_processor() doesn't drop BOS/EOS that was there before.  # update_post_processor()不会删除之前的BOS/EOS
    _V4_DEFAULTS = {"add_bos_token": True, "add_eos_token": False}  # v4默认值

    changed = False  # 是否有变更标志
    for attr in ("add_bos_token", "add_eos_token"):  # 遍历BOS/EOS属性
        config_val = config.get(attr)  # 从配置获取值
        if config_val is None:  # 如果键缺失或为null
            # Key missing or null -> use v4 default for this tokenizer class  # 键缺失或为null -> 使用v4默认值
            config_val = _V4_DEFAULTS.get(attr, False)  # 获取默认值
        # Fast tokenizers in v4 used tokenizer.json post-processor for EOS —  # v4中快速分词器使用tokenizer.json post-processor处理EOS——
        # the add_eos_token Python attribute was set but the post-processor  # add_eos_token Python属性已设置但post-processor
        # came from tokenizer.json, not from the attribute.  In v5, the flag is  # 来自tokenizer.json而非属性。在v5中，标志被
        # stripped and both sglang and HF reference end up with add_eos_token=False.  # 剥离，sglang和HF参考都以add_eos_token=False结束
        # Restoring add_eos_token for fast tokenizers makes sglang diverge from  # 为快速分词器恢复add_eos_token使sglang偏离
        # the HF reference, breaking embedding models like e5-mistral-7b-instruct.  # HF参考，破坏e5-mistral-7b-instruct等嵌入模型
        if attr == "add_eos_token" and isinstance(tokenizer, PreTrainedTokenizerFast):  # 如果是快速分词器的EOS
            config_val = _V4_DEFAULTS["add_eos_token"]  # False  # 使用v4默认值False
        current_val = getattr(tokenizer, attr, None)  # 获取当前值
        if current_val != config_val:  # 如果值不同
            logger.info(  # 记录信息
                "Restoring %s=%s for %s (was %s after v5 loading)",  # 恢复v5加载后的值
                attr,
                config_val,  # 要恢复的值
                model_name_or_path,
                current_val,  # 加载后的值
            )
            # Set the private backing attribute (not the property) because  # 设置私有支持属性（而非属性）因为
            # transformers tokenizers expose add_bos/eos_token as properties  # transformers分词器将add_bos/eos_token暴露为属性
            # that read from the underscore-prefixed attribute.  # 从下划线前缀属性读取
            setattr(tokenizer, f"_{attr}", config_val)  # 设置私有属性
            changed = True  # 标记已变更

    # Rebuild the post-processor so it respects the restored flags  # 重建post-processor以遵守恢复的标志
    if changed and hasattr(tokenizer, "update_post_processor"):  # 如果有变更且有update_post_processor
        tokenizer.update_post_processor()  # 更新post-processor


def _fix_special_tokens_pattern(tokenizer):  # 修复特殊token模式问题
    """Fix https://github.com/huggingface/transformers/pull/42563 which defaults  # 修复transformers PR #42563默认
    special_tokens_pattern to "cls_sep", inserting None into token IDs when  # special_tokens_pattern为"cls_sep"，当
    cls_token/sep_token are undefined (e.g. Kimi-VL's TikTokenTokenizer).  # cls_token/sep_token未定义时插入None到token ID
    """
    pattern = getattr(tokenizer, "special_tokens_pattern", None)  # 获取特殊token模式
    if pattern == "cls_sep" and (  # 如果模式为cls_sep且
        tokenizer.cls_token_id is None or tokenizer.sep_token_id is None  # cls或sep token ID未定义
    ):
        tokenizer.special_tokens_pattern = "none"  # 设置为none模式


def _apply_post_load_fixes(tokenizer, tokenizer_name, revision):  # 应用所有加载后修补并返回最终分词器
    """Apply all post-load patches and return the final tokenizer."""  # 应用所有加载后修补并返回最终分词器
    _fix_v5_tokenizer_components(tokenizer, tokenizer_name, revision)  # 修复v5分词器组件
    _fix_v5_add_bos_eos_token(tokenizer, tokenizer_name, revision)  # 修复v5 BOS/EOS token

    if not isinstance(tokenizer, PreTrainedTokenizerFast):  # 如果不是快速分词器
        warnings.warn(  # 发出警告
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."  # 使用慢速分词器可能显著降低性能
        )

    patch_mistral_common_tokenizer(tokenizer)  # 修补MistralCommon分词器
    _fix_special_tokens_pattern(tokenizer)  # 修复特殊token模式
    attach_additional_stop_token_ids(tokenizer)  # 附加停止token ID
    return patch_tokenizer(tokenizer)  # 应用分词器补丁


# ---------------------------------------------------------------------------
# Public entry point  # 公共入口点
# ---------------------------------------------------------------------------


_fastokens_patched = False  # fastokens补丁状态


def _ensure_fastokens_patched():  # 确保fastokens后端补丁已应用（仅一次）
    """Monkey-patch transformers to use the fastokens backend (once)."""  # 猴子补丁transformers以使用fastokens后端（仅一次）
    global _fastokens_patched  # 声明使用全局变量
    if _fastokens_patched:  # 如果已修补
        return  # 直接返回
    try:  # 尝试导入fastokens
        import fastokens  # 导入fastokens
    except ImportError:  # 如果导入失败
        raise ImportError(  # 抛出导入错误
            "The fastokens package is required when --tokenizer-backend=fastokens. "
            "Install it with: pip install 'sglang[fastokens]'"  # 安装fastokens
        ) from None

    fastokens.patch_transformers()  # 应用fastokens补丁
    _fastokens_patched = True  # 标记已修补
    logger.info("fastokens backend enabled - transformers patched successfully")  # 记录fastokens已启用


def get_tokenizer(  # 获取分词器的公共入口函数
    tokenizer_name: str,  # 分词器名称
    *args,  # 位置参数
    tokenizer_mode: str = "auto",  # 分词器模式
    trust_remote_code: bool = False,  # 是否信任远程代码
    tokenizer_revision: Optional[str] = None,  # 分词器版本
    tokenizer_backend: str = "huggingface",  # 分词器后端
    **kwargs,  # 其他参数
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""  # 通过HuggingFace获取指定模型名称的分词器
    # Tiktoken format has its own backend — no fastokens patching needed.  # Tiktoken格式有自己的后端——不需要fastokens补丁
    if tokenizer_name.endswith(".json"):  # 如果是Tiktoken格式
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer  # 导入Tiktoken分词器

        return TiktokenTokenizer(tokenizer_name)  # 返回Tiktoken分词器

    if tokenizer_backend == "fastokens":  # 如果使用fastokens后端
        _ensure_fastokens_patched()  # 确保已应用fastokens补丁

    if tokenizer_mode == "slow":  # 如果使用慢速模式
        if kwargs.get("use_fast", False):  # 如果同时请求快速分词器
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")  # 抛出冲突错误
        kwargs["use_fast"] = False  # 强制使用慢速分词器
    elif tokenizer_mode == "auto":  # 自动模式
        # Transformers v5 AutoTokenizer ignores use_fast (always fast), but  # Transformers v5 AutoTokenizer忽略use_fast（始终快速），但
        # some code paths pass kwargs to non-AutoTokenizer loaders where  # 某些代码路径将kwargs传递给非AutoTokenizer加载器
        # use_fast still matters. Set explicitly for those fallback paths.  # use_fast仍然重要。为这些回退路径显式设置
        if "use_fast" not in kwargs:  # 如果未指定use_fast
            kwargs["use_fast"] = True  # 默认使用快速分词器

    tokenizer_name = _resolve_tokenizer_name(tokenizer_name, kwargs)  # 解析分词器名称

    common_kwargs = dict(  # 构造通用参数字典
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        clean_up_tokenization_spaces=False,  # 不清理分词空格
        **kwargs,
    )

    try:  # 尝试加载分词器
        tokenizer = _auto_tokenizer_from_pretrained(  # 通过AutoTokenizer加载
            tokenizer_name, *args, **common_kwargs
        )

        # With fastokens, the patched TokenizersBackend.from_pretrained already  # 使用fastokens时，修补后的TokenizersBackend.from_pretrained已经
        # returned a tokenizer whose backend is a fastokens shim. Re-resolving via  # 返回了后端为fastokens垫片的分词器。通过
        # the declared class (e.g. Qwen2Tokenizer) would discard that work.  # 声明的类（如Qwen2Tokenizer）重新解析会丢弃这些工作
        if (  # 如果需要解析TokenizersBackend
            type(tokenizer).__name__ == _TOKENIZERS_BACKEND
            and tokenizer_backend != "fastokens"  # 非fastokens后端才解析
        ):
            tokenizer = _resolve_tokenizers_backend(  # 解析TokenizersBackend
                tokenizer_name, *args, **common_kwargs
            )

        return _apply_post_load_fixes(tokenizer, tokenizer_name, tokenizer_revision)  # 应用加载后修补
    except Exception as e:  # 捕获异常
        if tokenizer_backend == "fastokens":  # 如果是fastokens后端
            raise RuntimeError(  # 抛出运行时错误
                f"fastokens failed to load tokenizer for {tokenizer_name!r}. "
                f"This model's tokenizer may not be supported by fastokens — "
                f"see https://github.com/crusoecloud/fastokens. "
                f"Re-run without --tokenizer-backend=fastokens to use the default backend."  # 不使用fastokens重新运行
            ) from e
        raise  # 重新抛出其他异常


# ---------------------------------------------------------------------------
# Exported helpers (used by processor.py, etc.)  # 导出的辅助函数（被processor.py等使用）
# ---------------------------------------------------------------------------


def _fix_added_tokens_encoding(tokenizer):  # 确保特殊token在transformers v5中编码为单个token
    """Ensure special tokens encode as single tokens in transformers v5.  # 确保特殊token在transformers v5中编码为单个token

    Some model tokenizers (e.g. MiniCPM-V-4) define special tokens like <image>,  # 某些模型分词器（如MiniCPM-V-4）定义了<image>、
    <slice> as attributes on the tokenizer class with corresponding IDs in the  # <slice>等特殊token作为分词器类属性，在词汇表中有对应ID
    vocabulary (via tokenizer.json's added_tokens). In transformers v5, these  # （通过tokenizer.json的added_tokens）。在v5中这些
    tokens may not appear in get_added_vocab() and encode() splits them into  # token可能不出现在get_added_vocab()中，encode()将它们拆分为
    subwords, breaking multimodal pipelines that rely on finding them in input_ids.  # 子词，破坏依赖在input_ids中找到它们的多模态管道

    This function discovers such tokens by scanning tokenizer attributes, checks  # 此函数通过扫描分词器属性发现这些token，检查
    if they encode correctly, and re-registers any that don't.  # 它们是否正确编码，并重新注册不正确的
    """

    # Discover special token strings from tokenizer attributes.  # 从分词器属性发现特殊token字符串
    # Model tokenizers (e.g. MiniCPMVTokenizerFast) store them as attributes  # 模型分词器（如MiniCPMVTokenizerFast）将它们存储为属性
    # like im_start="<image>", slice_start="<slice>", etc.  # 如im_start="<image>"、slice_start="<slice>"等
    def _is_special_token_attr(val):  # 判断值是否为特殊token属性
        return (  # 返回判断结果
            isinstance(val, str)  # 值是字符串
            and val.startswith("<")  # 以<开头
            and val.endswith(">")  # 以>结尾
            and len(val) <= 20  # 长度不超过20
        )

    candidates = {}  # 候选token字典
    for attr in dir(tokenizer):  # 遍历分词器属性
        if attr.startswith("_"):  # 跳过私有属性
            continue  # 继续
        try:  # 尝试获取属性值
            val = getattr(tokenizer, attr)  # 获取属性值
        except (AttributeError, TypeError, ValueError):  # 如果获取失败
            continue  # 继续
        if not _is_special_token_attr(val):  # 如果不是特殊token
            continue  # 继续
        token_id = tokenizer.convert_tokens_to_ids(val)  # 获取token ID
        if token_id is not None and token_id != tokenizer.unk_token_id:  # 如果有有效ID
            candidates[val] = token_id  # 添加到候选

    if not candidates:  # 如果没有候选token
        return  # 直接返回

    def _encodes_correctly(token_str, expected_id):  # 检查token是否正确编码为单个ID
        try:  # 尝试编码
            ids = tokenizer.encode(token_str, add_special_tokens=False)  # 编码token
            return len(ids) == 1 and ids[0] == expected_id  # 检查是否为单个ID且匹配
        except (ValueError, OverflowError, RuntimeError) as e:  # 如果编码失败
            logger.debug("Token %s encode check failed: %s", token_str, e)  # 记录调试信息
            return False  # 返回False

    broken = [  # 找出编码不正确的token
        tok for tok, eid in candidates.items() if not _encodes_correctly(tok, eid)
    ]

    if not broken:  # 如果没有损坏的token
        return  # 直接返回

    from transformers import AddedToken  # 导入AddedToken类

    tokens_to_add = [AddedToken(tok, special=True, normalized=False) for tok in broken]  # 创建AddedToken列表
    tokenizer.add_tokens(tokens_to_add, special_tokens=True)  # 重新注册为特殊token
    logger.info(  # 记录信息
        "Re-registered %d special tokens for correct v5 encoding: %s",  # 重新注册特殊token以正确编码
        len(broken),
        broken[:10],  # 最多显示10个
    )
