# LoRA后端注册表实现
# 提供LoRA后端的注册和查找机制，支持triton、csgmv、ascend、
# torch_native等后端，flashinfer后端已弃用
import logging  # 导入日志模块
from typing import Type  # 导入类型提示

from sglang.srt.lora.backend.base_backend import BaseLoRABackend  # 导入LoRA后端基类

logger = logging.getLogger(__name__)  # 创建模块级日志器

LORA_SUPPORTED_BACKENDS = {}  # 存储已注册LoRA后端的字典


def register_lora_backend(name):  # LoRA后端注册装饰器，按名称注册后端类
    def decorator(fn):  # 装饰器内部函数
        LORA_SUPPORTED_BACKENDS[name] = fn  # 将后端创建函数注册到字典中
        return fn  # 返回原函数

    return decorator  # 返回装饰器


@register_lora_backend("triton")  # 注册triton后端
def create_triton_backend():  # 创建Triton LoRA后端的工厂函数
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend  # 延迟导入Triton后端类

    return TritonLoRABackend  # 返回Triton后端类


@register_lora_backend("csgmv")  # 注册csgmv（分段式SGMV）后端
def create_triton_csgmv_backend():  # 创建分段式SGMV LoRA后端的工厂函数
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend  # 延迟导入分段式后端类

    return ChunkedSgmvLoRABackend  # 返回分段式后端类


@register_lora_backend("ascend")  # 注册ascend（昇腾NPU）后端
def create_ascend_backend():  # 创建昇腾LoRA后端的工厂函数
    from sglang.srt.lora.backend.ascend_backend import AscendLoRABackend  # 延迟导入昇腾后端类

    return AscendLoRABackend  # 返回昇腾后端类


@register_lora_backend("torch_native")  # 注册torch_native后端
def create_torch_native_backend():  # 创建PyTorch原生LoRA后端的工厂函数
    from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend  # 延迟导入PyTorch原生后端类

    return TorchNativeLoRABackend  # 返回PyTorch原生后端类


@register_lora_backend("flashinfer")  # 注册flashinfer后端（已弃用）
def create_flashinfer_backend():  # 创建FlashInfer LoRA后端的工厂函数（已弃用）
    raise ValueError(  # 抛出值错误
        "FlashInfer LoRA backend has been deprecated, please use `triton` instead."  # FlashInfer LoRA后端已弃用，请改用`triton`
    )


def get_backend_from_name(name: str) -> Type[BaseLoRABackend]:  # 根据后端名称获取对应的后端类
    """
    Get corresponding backend class from backend's name
    根据后端名称获取对应的后端类
    """
    if name not in LORA_SUPPORTED_BACKENDS:  # 如果名称不在已注册后端中
        raise ValueError(f"Invalid backend: {name}")  # 抛出无效后端错误
    lora_backend = LORA_SUPPORTED_BACKENDS[name]()  # 调用注册的工厂函数获取后端类
    return lora_backend  # 返回后端类
