# 模型选择与加载工具模块
# 提供模型架构解析、实现方式选择、权重加载辅助等功能，支持SGLang原生实现和Transformers后端回退

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py
# 改编自 https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""  # 选择和加载模型的工具函数

import concurrent.futures  # 导入并发未来模块
import contextlib  # 导入上下文管理工具模块
import logging  # 导入日志模块
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type  # 导入类型提示

import torch  # 导入PyTorch
import transformers  # 导入transformers库
from torch import nn  # 从torch导入神经网络模块
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # 从transformers导入动态模块类获取函数

from sglang.srt.configs.model_config import ModelConfig, ModelImpl  # 导入模型配置类
from sglang.srt.layers import deep_gemm_wrapper  # 导入DeepGEMM包装器

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


@contextlib.contextmanager  # 上下文管理器装饰器
def set_default_torch_dtype(dtype: torch.dtype):  # 设置默认的PyTorch数据类型
    """Sets the default torch dtype to the given dtype."""  # 将默认torch数据类型设置为给定类型
    old_dtype = torch.get_default_dtype()  # 获取旧的默认数据类型
    torch.set_default_dtype(dtype)  # 设置新的默认数据类型
    yield  # 生成器暂停点
    torch.set_default_dtype(old_dtype)  # 恢复旧的默认数据类型


def _is_moe_model(model_config: ModelConfig, architectures: list[str]) -> bool:  # 判断模型是否为MoE（混合专家）模型
    lowered_arches = [arch.lower() for arch in architectures]  # 将架构名称转为小写
    if any("moe" in arch or "mixtral" in arch for arch in lowered_arches):  # 如果架构名中包含moe或mixtral
        return True  # 是MoE模型

    text_config = model_config.hf_text_config  # 获取HuggingFace文本配置
    expert_attrs = (  # 专家相关属性列表
        "num_local_experts",  # 本地专家数量
        "num_experts",  # 专家数量
        "num_experts_per_tok",  # 每个token的专家数量
        "moe_intermediate_size",  # MoE中间层大小
        "n_routed_experts",  # 路由专家数量
    )
    for attr in expert_attrs:  # 遍历专家属性
        value = getattr(text_config, attr, None)  # 获取属性值
        if value is None:  # 如果属性不存在
            continue  # 跳过
        if isinstance(value, bool):  # 如果属性值为布尔类型
            if value:  # 如果为True
                return True  # 是MoE模型
            continue  # 继续检查下一个属性
        if isinstance(value, (int, float)):  # 如果属性值为整数或浮点数
            threshold = 0 if attr == "moe_intermediate_size" else 1  # 设置阈值：moe_intermediate_size阈值为0，其他为1
            if value > threshold:  # 如果值超过阈值
                return True  # 是MoE模型
            continue  # 继续检查下一个属性
        if isinstance(value, (list, tuple, set, dict)):  # 如果属性值为集合类型
            if len(value) > 0:  # 如果集合非空
                return True  # 是MoE模型
            continue  # 继续检查下一个属性
        if isinstance(value, str) and value == "":  # 如果属性值为空字符串
            continue  # 跳过
        if value is not None:  # 如果属性值不为None
            return True  # 是MoE模型
    return False  # 不是MoE模型


def _is_sequence_classification_model(architectures: list[str]) -> bool:  # 判断模型是否为序列分类模型
    return any(  # 返回是否存在匹配
        "sequenceclassification" in lowered or "rewardmodel" in lowered  # 架构名中包含sequenceclassification或rewardmodel
        for lowered in (arch.lower() for arch in architectures)  # 将架构名转为小写
    )


def _get_transformers_backend_arch(  # 获取Transformers后端架构名称
    model_config: ModelConfig, architectures: list[str]  # 模型配置和架构列表
) -> str:
    is_pooling = not model_config.is_generation  # 是否为池化模型（非生成模型）
    is_multimodal = model_config.is_multimodal or (  # 是否为多模态模型
        model_config.hf_config is not model_config.hf_text_config  # 或者HF配置与文本配置不同
    )
    is_moe = _is_moe_model(model_config, architectures)  # 是否为MoE模型
    base_arch = "ForCausalLM"  # 基础架构后缀为因果语言模型
    if is_pooling:  # 如果是池化模型
        base_arch = (  # 设置基础架构后缀
            "ForSequenceClassification"  # 序列分类模型
            if _is_sequence_classification_model(architectures)  # 如果是序列分类模型
            else "EmbeddingModel"  # 否则为嵌入模型
        )

    arch = "Transformers"  # 架构前缀为Transformers
    if is_multimodal:  # 如果是多模态模型
        arch += "MultiModal"  # 追加MultiModal
    if is_moe:  # 如果是MoE模型
        arch += "MoE"  # 追加MoE
    return arch + base_arch  # 返回完整架构名称


def _model_impl_from_architecture(architecture: str) -> ModelImpl:  # 根据架构名称获取模型实现方式
    if architecture.startswith("Transformers"):  # 如果架构以Transformers开头
        return ModelImpl.TRANSFORMERS  # 返回Transformers实现
    if architecture.startswith("MindSpore"):  # 如果架构以MindSpore开头
        return ModelImpl.MINDSPORE  # 返回MindSpore实现
    return ModelImpl.SGLANG  # 返回SGLang实现


def resolve_transformers_arch(model_config: ModelConfig, architectures: list[str]):  # 解析Transformers架构，检查兼容性
    backend_arch = _get_transformers_backend_arch(model_config, architectures)  # 获取后端架构名称

    for arch in architectures:  # 遍历所有架构
        if arch.startswith("Transformers"):  # 如果架构以Transformers开头
            continue  # 跳过
        auto_map: dict[str, str] = (  # 获取auto_map配置
            getattr(model_config.hf_config, "auto_map", None) or dict()  # 获取auto_map或空字典
        )
        # Make sure that config class is always initialized before model class,
        # 确保配置类总是在模型类之前初始化，
        # otherwise the model class won't be able to access the config class,
        # 否则模型类将无法访问配置类，
        # the expected auto_map should have correct order like:
        # 预期的auto_map应具有正确的顺序，如：
        # "auto_map": {
        # "auto_map": {
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        # },
        # },
        auto_modules = {}  # 自动模块字典
        try:  # 尝试加载动态模块
            auto_modules = {  # 构建自动模块字典
                name: get_class_from_dynamic_module(  # 从动态模块获取类
                    module, model_config.model_path, revision=model_config.revision  # 模块路径和版本
                )
                for name, module in sorted(auto_map.items(), key=lambda x: x[0])  # 按名称排序
            }
        except Exception as e:  # 捕获异常
            logger.warning(  # 记录警告日志
                "Failed to load dynamic modules from auto_map for '%s': %s. "
                "Skipping remote model compatibility checks.",  # 加载动态模块失败，跳过兼容性检查
                arch,  # 架构名称
                e,  # 异常信息
            )
        model_module = getattr(transformers, arch, None)  # 从transformers获取模型模块
        if model_module is None:  # 如果模块不存在
            has_auto_model = "AutoModel" in auto_modules  # 检查auto_modules中是否有AutoModel
            if not has_auto_model and model_config.model_impl == ModelImpl.TRANSFORMERS:  # 如果没有AutoModel但显式请求Transformers
                logger.warning(  # 记录警告日志
                    "Cannot resolve model class for '%s' and no auto_map.AutoModel "
                    "is present. Skipping compatibility gate because "
                    "--model-impl=transformers is explicitly requested.",  # 无法解析模型类，跳过兼容性检查
                    arch,  # 架构名称
                )
                continue  # 跳过
            if not has_auto_model and "AutoModel" not in auto_map:  # 如果auto_modules和auto_map中都没有AutoModel
                raise ValueError(  # 抛出值错误
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library (only relevant if the "
                    "model is meant to be in Transformers) and 'AutoModel' is "
                    "not present in the model config's 'auto_map' (relevant "
                    "if the model is custom)."  # 无法找到模型模块
                )
            if not has_auto_model:  # 如果auto_modules中没有AutoModel（但auto_map中有）
                raise ValueError(  # 抛出值错误
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library and loading the custom "
                    f"model from auto_map failed. The remote model code may be "
                    f"incompatible with the installed transformers version."  # 加载自定义模型失败
                )
            model_module = auto_modules["AutoModel"]  # 使用auto_modules中的AutoModel
        if model_config.model_impl == ModelImpl.TRANSFORMERS:  # 如果显式请求Transformers实现
            if hasattr(model_module, "is_backend_compatible") and (  # 如果模型模块有兼容性检查方法
                not model_module.is_backend_compatible()  # 且不兼容
            ):
                logger.warning(  # 记录警告日志
                    "The Transformers implementation of %s reports it is not "
                    "backend-compatible (_supports_attention_backend=False). "
                    "Proceeding anyway because --model-impl=transformers was "
                    "explicitly requested. The model may not work correctly.",  # Transformers实现不兼容但仍继续
                    arch,  # 架构名称
                )
        if model_config.model_impl == ModelImpl.AUTO:  # 如果为自动选择实现模式
            if hasattr(model_module, "is_backend_compatible") and (  # 如果模型模块有兼容性检查方法
                not model_module.is_backend_compatible()  # 且不兼容
            ):
                raise ValueError(  # 抛出值错误
                    f"{arch} has no SGlang implementation and the Transformers "
                    "implementation is not compatible with SGLang."  # 无SGLang实现且Transformers不兼容
                )
            logger.warning(  # 记录警告日志
                "%s has no SGLang implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",  # 回退到Transformers实现
                arch,  # 架构名称
            )
    return [backend_arch]  # 返回后端架构列表


def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:  # 获取模型架构类和名称
    from sglang.srt.models.registry import ModelRegistry  # 导入模型注册表

    architectures = getattr(model_config.hf_config, "architectures", [])  # 获取模型架构列表
    # Special handling for quantized Mixtral.
    # 量化Mixtral模型的特殊处理。
    # FIXME(woosuk): This is a temporary hack.
    # FIXME(woosuk): 这是一个临时方案。
    mixtral_supported = [  # Mixtral支持的量化方式列表
        "fp8",  # FP8量化
        "compressed-tensors",  # 压缩张量量化
        "gptq_marlin",  # GPTQ Marlin量化
        "awq_marlin",  # AWQ Marlin量化
        "quark_int4fp8_moe",  # Quark INT4 FP8 MoE量化
    ]

    if (  # 如果满足以下条件
        model_config.quantization is not None  # 模型配置有量化设置
        and model_config.quantization not in mixtral_supported  # 量化方式不在支持列表中
        and "MixtralForCausalLM" in architectures  # 且架构为MixtralForCausalLM
    ):
        architectures = ["QuantMixtralForCausalLM"]  # 使用量化Mixtral架构

    supported_archs = ModelRegistry.get_supported_archs()  # 获取注册表支持的架构列表
    is_native_supported = any(arch in supported_archs for arch in architectures)  # 检查是否有原生支持的架构

    if model_config.model_impl == ModelImpl.MINDSPORE:  # 如果使用MindSpore实现
        architectures = ["MindSporeForCausalLM"]  # 使用MindSpore架构
    elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:  # 如果没有原生支持或显式请求Transformers
        architectures = resolve_transformers_arch(model_config, architectures)  # 解析Transformers架构
    model_cls, resolved_arch = ModelRegistry.resolve_model_cls(architectures)  # 从注册表解析模型类
    setattr(model_config, "_resolved_model_arch", resolved_arch)  # 设置已解析的模型架构属性
    setattr(  # 设置已解析的模型实现属性
        model_config,
        "_resolved_model_impl",
        _model_impl_from_architecture(resolved_arch),  # 根据架构获取实现方式
    )
    return model_cls, resolved_arch  # 返回模型类和已解析的架构名


def get_resolved_model_impl(model_config: ModelConfig) -> ModelImpl:  # 获取已解析的模型实现方式
    resolved_model_impl = getattr(model_config, "_resolved_model_impl", None)  # 尝试获取缓存的实现方式
    if resolved_model_impl is not None:  # 如果已缓存
        return resolved_model_impl  # 直接返回

    resolved_arch = getattr(model_config, "_resolved_model_arch", None)  # 尝试获取已解析的架构
    if resolved_arch is None:  # 如果未缓存架构
        _, resolved_arch = get_model_architecture(model_config)  # 获取模型架构

    resolved_model_impl = _model_impl_from_architecture(resolved_arch)  # 根据架构获取实现方式
    setattr(model_config, "_resolved_model_arch", resolved_arch)  # 缓存已解析的架构
    setattr(model_config, "_resolved_model_impl", resolved_model_impl)  # 缓存已解析的实现方式
    return resolved_model_impl  # 返回模型实现方式


def get_architecture_class_name(model_config: ModelConfig) -> str:  # 获取模型架构类名
    return get_model_architecture(model_config)[1]  # 返回架构名称字符串


def should_deepgemm_weight_requant_ue8m0(weight_block_size):  # 判断是否应将FP8权重量化为UE8M0格式
    """Should we requant fp8 weights into UE8M0 format when loading the model"""  # 加载模型时是否应将FP8权重量化为UE8M0格式
    return (  # 返回条件判断结果
        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM  # 是否启用JIT DeepGEMM
        and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0  # DeepGEMM是否使用UE8M0缩放
        and weight_block_size is not None  # 权重块大小不为None
    )


def should_async_load(weight: torch.Tensor) -> bool:  # 判断是否应异步加载给定权重
    """Return True if we should load the given weight asynchronously.
    """返回True如果我们应异步加载给定权重。

    For host (CPU) tensors, using a threadpool can overlap H2D copies
    对于主机（CPU）张量，使用线程池可以重叠H2D拷贝
    and improve throughput. For device tensors, threading often adds overhead
    并提高吞吐量。对于设备张量，线程化通常增加开销
    (e.g., GIL contention) without benefit, so we do it synchronously.
    （例如GIL竞争）而没有收益，因此我们同步执行。
    """
    device = getattr(weight, "device", None)  # 获取权重张量的设备属性
    if device is None:  # 如果没有设备属性
        return False  # 不异步加载
    return device.type == "cpu"  # CPU张量返回True（异步加载），设备张量返回False（同步加载）


def maybe_executor_submit(  # 可选地提交任务到线程池执行器
    *,
    executor: concurrent.futures.ThreadPoolExecutor,  # 线程池执行器
    futures: List[concurrent.futures.Future],  # 用于收集已提交Future对象的列表
    use_async: bool,  # 是否提交到执行器或同步运行
    func: Callable[..., Any],  # 要运行的 callable
    func_args: Iterable[Any] = (),  # callable 的位置参数（默认为空元组）
    func_kwargs: Optional[Dict[str, Any]] = None,  # callable 的关键字参数（默认为空字典）
) -> None:
    """Submit a task to the executor if async loading is enabled.
    """如果启用了异步加载，则提交任务到执行器。

    Parameters (keyword-only):
    参数（仅关键字）：
    - executor: ThreadPoolExecutor used to submit background tasks
    - executor: 用于提交后台任务的ThreadPoolExecutor
    - futures: a list collecting the submitted Future objects
    - futures: 收集已提交Future对象的列表
    - use_async: whether to submit to executor or run inline
    - use_async: 是否提交到执行器或同步运行
    - func: the callable to run
    - func: 要运行的 callable
    - func_args: positional args for the callable (defaults to empty tuple)
    - func_args: callable 的位置参数（默认为空元组）
    - func_kwargs: keyword args for the callable (defaults to empty dict)
    - func_kwargs: callable 的关键字参数（默认为空字典）
    """
    if func_kwargs is None:  # 如果关键字参数为None
        func_kwargs = {}  # 设置为空字典
    if use_async:  # 如果启用异步加载
        futures.append(executor.submit(func, *func_args, **func_kwargs))  # 提交任务到执行器并收集Future
    else:  # 否则同步执行
        func(*func_args, **func_kwargs)  # 直接调用函数
