# 多模态处理器注册与查找模块
# 负责自动发现、注册和获取多模态数据处理器

# TODO: also move pad_input_ids into this module  TODO: 也将pad_input_ids移入此模块
import importlib  # 动态导入模块
import inspect  # 检查模块成员
import logging  # 日志库
import pkgutil  # 包工具，用于迭代包中的模块

from sglang.srt.configs.model_config import ModelImpl  # 模型实现枚举
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor  # 多模态处理器基类
from sglang.srt.server_args import ServerArgs  # 服务器参数

logger = logging.getLogger(__name__)  # 获取日志记录器

PROCESSOR_MAPPING = {}  # 模型架构类到处理器类的映射字典


def import_processors(package_name: str, overwrite: bool = False):  # 导入指定包中的所有多模态处理器并注册到映射表
    """导入指定包中的所有多模态处理器并注册到映射表"""
    package = importlib.import_module(package_name)  # 导入指定包
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):  # 遍历包中的所有模块
        if not ispkg:  # 如果不是包
            try:  # 尝试导入模块
                module = importlib.import_module(name)  # 导入模块
            except Exception as e:  # 捕获导入异常
                logger.warning(f"Ignore import error when loading {name}: {e}")  # 忽略导入错误
                continue  # 跳过当前模块
            all_members = inspect.getmembers(module, inspect.isclass)  # 获取模块中所有类成员
            classes = [  # 筛选定义在当前模块中的类
                member
                for name, member in all_members
                if member.__module__ == module.__name__  # 只选择定义在当前模块中的类
            ]
            for cls in (  # 遍历BaseMultimodalProcessor的子类
                cls for cls in classes if issubclass(cls, BaseMultimodalProcessor)
            ):
                assert hasattr(cls, "models")  # 断言处理器类必须有models属性
                for arch in getattr(cls, "models"):  # 遍历处理器类支持的模型架构
                    if overwrite:  # 如果允许覆盖
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():  # 遍历已有映射
                            if model_cls.__name__ == arch.__name__:  # 如果名称匹配
                                del PROCESSOR_MAPPING[model_cls]  # 删除旧映射
                                break  # 跳出内层循环
                    PROCESSOR_MAPPING[arch] = cls  # 注册模型架构到处理器的映射


def get_mm_processor(  # 根据模型配置获取对应的多模态处理器实例
    hf_config,  # HuggingFace模型配置
    server_args: ServerArgs,  # 服务器参数
    processor,  # HuggingFace处理器
    transport_mode,  # 传输模式
    model_config=None,  # 模型配置（可选）
    **kwargs,  # 其他关键字参数
) -> BaseMultimodalProcessor:  # 返回多模态处理器实例
    """根据模型架构和实现类型获取对应的多模态处理器实例"""
    model_impl = str(getattr(server_args, "model_impl", "auto")).lower()  # 获取模型实现类型，默认auto
    uses_transformers_backend = model_impl == "transformers"  # 是否使用transformers后端
    if model_impl == "auto" and model_config is not None:  # 如果是自动模式且提供了模型配置
        from sglang.srt.model_loader.utils import get_resolved_model_impl  # 导入获取解析模型实现的函数

        uses_transformers_backend = (  # 判断是否使用transformers后端
            get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
        )

    for model_cls, processor_cls in PROCESSOR_MAPPING.items():  # 遍历所有已注册的处理器映射
        if model_cls.__name__ not in hf_config.architectures:  # 如果模型架构不匹配
            continue  # 跳过
        if not uses_transformers_backend or getattr(  # 如果不使用transformers后端，或处理器支持transformers后端
            processor_cls, "supports_transformers_backend", False
        ):
            return processor_cls(  # 返回处理器实例
                hf_config, server_args, processor, transport_mode, **kwargs
            )

    if uses_transformers_backend:  # 如果使用transformers后端但没有找到专用处理器
        from sglang.srt.multimodal.processors.transformers_auto import (  # 导入transformers自动处理器
            TransformersAutoMultimodalProcessor,
        )

        return TransformersAutoMultimodalProcessor(  # 返回transformers自动处理器实例
            hf_config, server_args, processor, transport_mode, **kwargs
        )

    raise ValueError(  # 找不到处理器则抛出值错误
        f"No processor registered for architecture: {hf_config.architectures}.\n"
        f"Registered architectures: {[model_cls.__name__ for model_cls in PROCESSOR_MAPPING.keys()]}"
    )
