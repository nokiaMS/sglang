# 本文件提供自定义多模态处理器注册工具，用于将模型配置类的 model_type 映射到自定义的处理器类，替代 Hugging Face 默认处理器
from typing import Dict, Type  # 导入类型提示工具

from transformers import PretrainedConfig, ProcessorMixin  # 导入 Hugging Face 的配置基类和处理器混合基类

# Useful for registering a custom processor different from Hugging Face's default.
_CUSTOMIZED_MM_PROCESSOR: Dict[str, Type[ProcessorMixin]] = dict()  # 自定义多模态处理器映射字典，键为 model_type，值为处理器类


def register_customized_processor(
    processor_class: Type[ProcessorMixin],  # 要注册的自定义处理器类
):
    """Class decorator that maps a config class's model_type field to a customized processor class.
    类装饰器，将配置类的 model_type 字段映射到自定义处理器类。

    Args:
        processor_class: A processor class that inherits from ProcessorMixin

    Example:
        ```python
        @register_customized_processor(MyCustomProcessor)
        class MyModelConfig(PretrainedConfig):
            model_type = "my_model"

        ```
    """

    def decorator(config_class: PretrainedConfig):  # 内部装饰器函数，接收配置类
        if not hasattr(config_class, "model_type"):  # 检查配置类是否具有 model_type 属性
            raise ValueError(  # 若无则抛出异常
                f"Class {config_class.__name__} with register_customized_processor should "
                f"have a 'model_type' class attribute."
            )
        _CUSTOMIZED_MM_PROCESSOR[config_class.model_type] = processor_class  # 将 model_type 映射到处理器类
        return config_class  # 返回原配置类

    return decorator  # 返回装饰器函数
