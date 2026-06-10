# 对transformers内部的猴子补丁（monkey-patches）模块
# 包含向后兼容垫片（重新添加v5中移除的符号）、transformers v5错误的变通方案、
# 尚未为v5更新的远程模型代码(trust_remote_code)修复，以及仅CI的补丁
# （例如中和HF API调用以避免速率限制）
# 在任何from_pretrained调用之前尽早导入此模块以激活所有补丁
# 多次导入是安全的——补丁是幂等的
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
"""Monkey-patches on transformers internals.

Mix of backward-compat shims (re-add symbols removed in v5), workarounds
for transformers v5 bugs, fixes for remote-model-code (trust_remote_code)
that hasn't been updated for v5 yet, and CI-only patches (e.g. neutralize
HF API calls to avoid rate limits).

Import this module early (before any ``from_pretrained`` call) to activate
all patches.  It is safe to import multiple times -- patches are idempotent.
"""

import inspect  # 导入检查模块，用于函数签名检查

from sglang.srt.utils import logger  # 导入日志记录器

_applied = False  # 标记补丁是否已应用的全局标志


# ---------------------------------------------------------------------------
# Public API: apply_all() -- import-time patches (idempotent)
# ---------------------------------------------------------------------------


def apply_all():  # 应用所有transformers兼容性补丁（幂等操作）
    """Apply all transformers compatibility patches (idempotent).

    Call this once at import time.  It is safe to call multiple times.

    No-op when the ``transformers`` package is not installed -- frontend-only
    sglang users should not be forced to install transformers just to import
    the top-level ``sglang`` package.
    """
    global _applied  # 声明使用全局变量
    if _applied:  # 如果补丁已应用，直接返回
        return
    try:
        import transformers  # noqa: F401  # 尝试导入transformers
    except ImportError:  # 如果未安装transformers
        _applied = True  # 标记为已应用
        return  # 直接返回
    _applied = True  # 标记补丁为已应用

    # v5.4 patches  # v5.4版本的补丁
    _patch_flash_attn_availability()  # 修补flash-attn可用性检测
    _patch_rope_parameters_validation()  # 修补rope参数验证
    _patch_removed_symbols()  # 修补已移除的符号
    _patch_image_processor_kwargs()  # 修补图像处理器关键字参数
    _patch_image_process_cuda_tensor()  # 修补图像处理CUDA张量问题
    _patch_nemotron_h_pattern()  # 修补Nemotron-H模式问题

    # v5 general patches  # v5通用补丁
    _ensure_clean_up_tokenization_compat()  # 确保clean_up_tokenization兼容性
    _ensure_is_torch_fx_available_compat()  # 确保is_torch_fx_available兼容性

    # CI-only: neutralize HF API calls inside tokenizer from_pretrained  # 仅CI：中和tokenizer from_pretrained中的HF API调用
    patch_is_base_mistral_in_ci()  # 在CI中修补is_base_mistral

    logger.debug("transformers compatibility patches applied")  # 记录调试日志


# ---------------------------------------------------------------------------
# Public API: on-demand helpers (called explicitly by other modules)
# ---------------------------------------------------------------------------


def normalize_rope_scaling_compat(config) -> None:  # 确保rope_scaling字典中同时有"type"和"rope_type"键
    """Ensure rope_scaling dicts have ``"type"`` alongside ``"rope_type"``.

    Transformers v5 standardises rope_scaling to use ``"rope_type"`` and may
    omit the legacy ``"type"`` key.  Remote-code models (e.g. Kimi-VL) still
    read ``rope_scaling["type"]``, causing a ``KeyError``.  This helper adds
    ``"type"`` from ``"rope_type"`` whenever it is missing, recursively across
    the config and all its sub-configs.
    """

    def _patch(cfg):  # 递归修补配置对象
        rs = getattr(cfg, "rope_scaling", None)  # 获取rope_scaling属性
        if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:  # 如果rope_scaling是字典且包含rope_type但不包含type
            rs["type"] = rs["rope_type"]  # 将rope_type的值复制到type键
        # Recurse into sub-configs  # 递归处理子配置
        for attr in (  # 遍历可能的子配置属性名
            "text_config",
            "llm_config",
            "language_config",
            "vision_config",
            "thinker_config",
        ):
            sub = getattr(cfg, attr, None)  # 获取子配置属性
            if sub is not None:  # 如果子配置存在
                _patch(sub)  # 递归修补子配置

    _patch(config)  # 从顶层配置开始修补


def _ensure_gguf_version():  # 确保gguf包有__version__属性的变通方案
    """Workaround for transformers v5 bug where is_gguf_available() fails
    when the gguf package lacks __version__ and metadata lookup also fails,
    resulting in packaging.version.InvalidVersion: Invalid version: 'N/A'."""
    try:
        import gguf  # 尝试导入gguf包

        if not hasattr(gguf, "__version__"):  # 如果gguf没有__version__属性
            import importlib.metadata  # 导入元数据模块

            try:
                gguf.__version__ = importlib.metadata.version("gguf")  # 从元数据获取版本号
            except importlib.metadata.PackageNotFoundError:  # 如果包未找到
                gguf.__version__ = "0.0.0"  # 设置默认版本号
            except (ValueError, OSError, TypeError) as e:  # 如果出现其他错误
                logger.warning(  # 记录警告日志
                    "Failed to determine gguf package version: %s. "
                    "Falling back to '0.0.0'.",
                    e,
                )
                gguf.__version__ = "0.0.0"  # 回退到默认版本号
    except ImportError:  # 如果gguf未安装
        pass  # 忽略


# ---------------------------------------------------------------------------
# v5.4 patches (merged from transformers_v54_compat.py)
# ---------------------------------------------------------------------------


def _patch_rope_parameters_validation():  # 修补rope_parameters验证，处理未注册模型类型
    """Fix rope_parameters validation for unregistered model types.

    For unregistered model types (e.g. ``deepseek_v32``), the generic
    ``PretrainedConfig`` lacks a ``rope_parameters`` field so the conversion
    that injects ``rope_theta`` from the top-level config is skipped.
    Additionally, ``standardize_rope_params()`` accesses
    ``self.max_position_embeddings`` during ``__post_init__`` before extra
    kwargs are set as attributes, causing ``AttributeError``.

    Fix: (1) patch ``from_dict`` to inject ``rope_theta`` into
    ``rope_scaling``, (2) guard ``standardize_rope_params`` against missing
    ``max_position_embeddings``.

    TODO(upstream): remove once unregistered model types handle rope
    standardization correctly in transformers.
    """
    from transformers import PretrainedConfig  # 导入PretrainedConfig类

    original = PretrainedConfig.from_dict.__func__  # 保存原始的from_dict方法

    @classmethod  # type: ignore[misc]
    def patched(cls, config_dict, **kwargs):  # 修补后的from_dict方法
        rope_scaling = config_dict.get("rope_scaling")  # 获取rope_scaling配置
        rope_theta = config_dict.get("rope_theta")  # 获取rope_theta配置
        if (  # 如果rope_scaling是字典，且rope_theta存在，且rope_scaling中没有rope_theta
            isinstance(rope_scaling, dict)
            and rope_theta is not None
            and "rope_theta" not in rope_scaling
        ):
            config_dict = config_dict.copy()  # 复制配置字典以避免修改原始数据
            config_dict["rope_scaling"] = {**rope_scaling, "rope_theta": rope_theta}  # 将rope_theta注入rope_scaling
        return original(cls, config_dict, **kwargs)  # 调用原始方法

    PretrainedConfig.from_dict = patched  # 替换from_dict方法

    # standardize_rope_params accesses self.max_position_embeddings before
    # __post_init__ sets extra kwargs — skip when the attribute is absent.
    if hasattr(PretrainedConfig, "standardize_rope_params"):  # 如果存在standardize_rope_params方法
        _orig_standardize = PretrainedConfig.standardize_rope_params  # 保存原始方法

        def _safe_standardize(self):  # 安全版本的standardize_rope_params
            if not hasattr(self, "max_position_embeddings"):  # 如果缺少max_position_embeddings属性
                return  # 直接返回，跳过标准化
            return _orig_standardize(self)  # 否则调用原始方法

        PretrainedConfig.standardize_rope_params = _safe_standardize  # 替换为安全版本


def _patch_flash_attn_availability():  # 防止flash-attn-4伪装成flash-attn-2
    """Prevent flash-attn-4 from masquerading as flash-attn-2.

    flash-attn-4 registers a bare ``flash_attn`` namespace that makes
    ``is_flash_attn_2_available()`` return True, but lacks the v2 API.
    Remote model code (e.g. Kimi-VL) guarded by that check will crash.

    TODO(upstream): model authors should check for specific API symbols.
    """
    try:
        import flash_attn as _fa  # 尝试导入flash_attn

        if not hasattr(_fa, "flash_attn_func"):  # 如果没有flash_attn_func（说明不是v2 API）
            import transformers.utils as _u  # 导入transformers.utils
            import transformers.utils.import_utils as _ui  # 导入import_utils

            _ui.is_flash_attn_2_available = lambda: False  # 强制返回False
            _u.is_flash_attn_2_available = lambda: False  # 强制返回False
    except ImportError:  # 如果flash_attn未安装
        pass  # 忽略


def _patch_removed_symbols():  # 重新导出transformers v5.4.0中移除的符号
    """Re-export symbols removed in transformers v5.4.0.

    Remote model code (e.g. DeepSeek-OCR) still imports these.
    ``check_imports`` in ``dynamic_module_utils.py`` validates imports at
    config-load time, so these must exist before any ``from_pretrained``.

    Removed symbols:
    - ``LlamaFlashAttention2`` -- replaced by unified ``LlamaAttention``
    - ``is_flash_attn_greater_or_equal_2_10`` -- replaced by
      ``is_flash_attn_greater_or_equal("2.10.0")``

    TODO(upstream): DeepSeek-OCR / deepseek_vl_v2 remote code needs update.
    """
    # LlamaFlashAttention2  # 修补LlamaFlashAttention2
    try:
        import logging  # 导入日志模块

        # Importing modeling_llama triggers a deep import chain:
        #   modeling_llama -> modeling_utils -> quantizers -> torchao
        # torchao emits a noisy warning about incompatible torch versions
        # that is irrelevant here — suppress it during this import.
        _torchao_logger = logging.getLogger("torchao")  # 获取torchao日志记录器
        _prev_level = _torchao_logger.level  # 保存之前的日志级别
        _torchao_logger.setLevel(logging.ERROR)  # 临时设置为ERROR级别以抑制警告
        try:
            from transformers.models.llama import modeling_llama  # 导入llama建模模块
        finally:
            _torchao_logger.setLevel(_prev_level)  # 恢复原始日志级别

        if not hasattr(modeling_llama, "LlamaFlashAttention2"):  # 如果LlamaFlashAttention2不存在
            if hasattr(modeling_llama, "LlamaAttention"):  # 如果LlamaAttention存在
                modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention  # 用LlamaAttention替代
    except ImportError:  # 如果导入失败
        logger.warning(  # 记录警告日志
            "Could not import transformers.models.llama.modeling_llama; "
            "LlamaFlashAttention2 compat patch not applied."
        )

    # is_flash_attn_greater_or_equal_2_10  # 修补is_flash_attn_greater_or_equal_2_10
    try:
        import transformers.utils as _u  # 导入transformers.utils

        if not hasattr(_u, "is_flash_attn_greater_or_equal_2_10"):  # 如果符号不存在
            if hasattr(_u, "is_flash_attn_greater_or_equal"):  # 如果新版本函数存在
                _u.is_flash_attn_greater_or_equal_2_10 = (  # 创建兼容函数
                    lambda: _u.is_flash_attn_greater_or_equal("2.10.0")
                )
            else:  # 如果新版本函数也不存在
                _u.is_flash_attn_greater_or_equal_2_10 = lambda: False  # 默认返回False
    except ImportError:  # 如果导入失败
        logger.warning(  # 记录警告日志
            "Could not import transformers.utils; "
            "is_flash_attn_greater_or_equal_2_10 compat patch not applied."
        )


def _patch_image_processor_kwargs():  # 允许缺少**kwargs的远程图像处理器正常工作
    """Allow remote image processors that lack ``**kwargs`` in preprocess().

    Transformers v5.4 passes new kwargs (e.g. ``device``) through
    ``BaseImageProcessor.__call__`` -> ``preprocess()``.  Remote model code
    (e.g. KimiVL) that defines ``preprocess()`` without ``**kwargs`` will
    crash with ``TypeError``.

    Fix: wrap ``__call__`` to catch ``TypeError`` and retry with only the
    kwargs that ``preprocess()`` actually accepts.

    TODO(upstream): KimiVL image_processing_kimi_vl.py needs ``**kwargs``.
    """
    try:
        from transformers.image_processing_utils import BaseImageProcessor  # 导入BaseImageProcessor

        original = BaseImageProcessor.__call__  # 保存原始__call__方法

        def safe_call(self, images, *args, **kwargs):  # 安全版本的__call__方法
            try:
                return original(self, images, *args, **kwargs)  # 尝试调用原始方法
            except TypeError as e:  # 如果出现类型错误
                if "unexpected keyword argument" not in str(e):  # 如果不是意外关键字参数错误
                    raise  # 重新抛出异常
                sig = inspect.signature(self.preprocess)  # 获取preprocess方法的签名
                params = sig.parameters  # 获取参数信息
                if any(  # 如果preprocess接受**kwargs
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    raise  # 说明错误不是缺少**kwargs，重新抛出
                dropped = {k for k in kwargs if k not in params}  # 找出preprocess不接受的参数
                if dropped:  # 如果有不接受的参数
                    logger.warning(  # 记录警告日志
                        "Image processor %s.preprocess() does not accept %s; "
                        "retrying without them. Update the model's image processor "
                        "to accept **kwargs.",
                        type(self).__name__,
                        dropped,
                    )
                valid = {k: v for k, v in kwargs.items() if k in params}  # 仅保留有效参数
                return original(self, images, *args, **valid)  # 用有效参数重试

        BaseImageProcessor.__call__ = safe_call  # 替换为安全版本
    except ImportError:  # 如果导入失败
        logger.debug(  # 记录调试日志
            "_patch_image_processor_kwargs: BaseImageProcessor not importable, patch skipped"
        )


def _patch_image_process_cuda_tensor():  # 修复process_image()在CUDA张量上崩溃的问题
    """Fix ``process_image()`` crashing on CUDA tensors.

    Transformers v5.4's PIL image processing backend calls
    ``image.numpy()`` on torch tensors, which fails for CUDA tensors.
    Patch to call ``.cpu().numpy()`` instead.

    TODO(upstream): report to HF transformers.
    """
    try:
        import torch  # 导入torch
        import transformers.image_processing_backends as ipb  # 导入图像处理后端

        for cls_name in ("PilBackend", "PilImageProcessingMixin"):  # 遍历可能的类名
            cls = getattr(ipb, cls_name, None)  # 获取类对象
            if cls is None or not hasattr(cls, "process_image"):  # 如果类不存在或没有process_image方法
                continue  # 跳过
            original = cls.process_image  # 保存原始process_image方法

            def patched_process_image(  # 修补后的process_image方法
                self, image, *args, _orig=original, _Tensor=torch.Tensor, **kwargs
            ):
                if isinstance(image, _Tensor) and image.is_cuda:  # 如果图像是CUDA张量
                    image = image.cpu()  # 将其移动到CPU
                return _orig(self, image, *args, **kwargs)  # 调用原始方法

            cls.process_image = patched_process_image  # 替换为修补版本
    except ImportError:  # 如果导入失败
        logger.debug(  # 记录调试日志
            "_patch_image_process_cuda_tensor: required modules not importable, patch skipped"
        )


def _patch_nemotron_h_pattern():  # 修复_pattern_to_list()在hybrid_override_pattern中遇到"-"时崩溃的问题
    """Fix ``_pattern_to_list()`` crashing on ``-`` in hybrid_override_pattern.

    Nemotron-H models (e.g. NVIDIA-Nemotron-Nano-9B-v2) use patterns like
    ``M-M-M-MM-M-*-...`` where ``-`` denotes an MLP layer.  The upstream
    ``_pattern_to_list`` tries to map every character and crashes with
    ``KeyError: '-'``.  We skip ``-`` (and any other unmapped chars)
    since ``layers_block_type`` only tracks mamba/moe/attention layers.
    SGLang reads MLP positions from ``hybrid_override_pattern`` directly.

    TODO(upstream): report to HF transformers.
    """
    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import (  # 导入NemotronHConfig
            NemotronHConfig,
        )

        @staticmethod
        def _pattern_to_list(pattern: str) -> list:  # 修补后的_pattern_to_list方法
            pattern_mapping = {  # 模式字符映射表
                "M": "mamba",
                "E": "moe",
                "*": "attention",
            }
            return [  # 仅返回映射表中存在的字符对应的值
                pattern_mapping[char] for char in pattern if char in pattern_mapping
            ]

        NemotronHConfig._pattern_to_list = _pattern_to_list  # 替换为修补版本
    except ImportError:  # 如果导入失败
        logger.debug(  # 记录调试日志
            "_patch_nemotron_h_pattern: NemotronHConfig not importable, patch skipped"
        )


# ---------------------------------------------------------------------------
# v5 general patches
# ---------------------------------------------------------------------------


def _ensure_clean_up_tokenization_compat() -> None:  # 重新添加transformers v5中移除的clean_up_tokenization方法
    """Re-add ``clean_up_tokenization`` removed in transformers v5.

    Remote-code tokenizers (e.g. InternLM2Tokenizer) call
    ``self.clean_up_tokenization()`` which was a static method on
    ``PreTrainedTokenizerBase`` in v4 but removed in v5. Patch it back
    so existing HuggingFace Hub tokenizer code keeps working.
    """
    from transformers import PreTrainedTokenizerBase  # 导入PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "clean_up_tokenization"):  # 如果方法已存在
        return  # 无需修补

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:  # clean_up_tokenization的实现
        out_string = (  # 替换分词产生的多余空格
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string  # 返回清理后的字符串

    PreTrainedTokenizerBase.clean_up_tokenization = clean_up_tokenization  # 将方法添加回类


def _ensure_is_torch_fx_available_compat() -> None:  # 重新添加transformers v5中移除的is_torch_fx_available函数
    """Re-add ``is_torch_fx_available`` removed in transformers v5.

    Remote-code models (e.g. MiniCPM-V) import ``is_torch_fx_available``
    from ``transformers.utils.import_utils``.  The function was removed
    in v5.  Patch it back so existing HuggingFace Hub model code keeps
    working.  torch.fx is always available in PyTorch >= 2.0.
    """
    import transformers.utils.import_utils as _import_utils  # 导入import_utils模块

    if hasattr(_import_utils, "is_torch_fx_available"):  # 如果函数已存在
        return  # 无需修补

    _import_utils.is_torch_fx_available = lambda: True  # 添加函数，PyTorch>=2.0始终可用


# ---------------------------------------------------------------------------
# CI-only patches
# ---------------------------------------------------------------------------

_is_base_mistral_patched = False  # 标记is_base_mistral补丁是否已应用


def patch_is_base_mistral_in_ci():  # 在CI中修补transformers的_patch_mistral_regex以避免HF API调用
    """Patch transformers' _patch_mistral_regex to avoid HF API calls in CI.

    transformers defines is_base_mistral as a local function inside
    _patch_mistral_regex, so it cannot be patched via module attribute.
    Instead we replace the entire _patch_mistral_regex classmethod with a
    version that simply returns the tokenizer unchanged.

    In CI this prevents exhausting the 3000 req/5min HF API rate limit.

    TODO(upstream): remove once transformers stops calling model_info()
    inside _patch_mistral_regex (or removes the method entirely).
    """
    global _is_base_mistral_patched  # 声明使用全局变量
    if _is_base_mistral_patched:  # 如果已修补
        return  # 直接返回

    from sglang.srt.environ import envs  # 导入环境变量

    if not envs.SGLANG_IS_IN_CI.get():  # 如果不在CI环境中
        return  # 直接返回

    from transformers import PreTrainedTokenizerFast  # 导入PreTrainedTokenizerFast

    if hasattr(PreTrainedTokenizerFast, "_patch_mistral_regex"):  # 如果存在_patch_mistral_regex方法

        @classmethod
        def _noop_patch_mistral_regex(cls, tokenizer, *args, **kwargs):  # 空操作版本，直接返回tokenizer
            return tokenizer

        PreTrainedTokenizerFast._patch_mistral_regex = _noop_patch_mistral_regex  # 替换为空操作版本
        logger.info("CI: patched _patch_mistral_regex to skip HF API calls")  # 记录信息日志

    _is_base_mistral_patched = True  # 标记为已修补
