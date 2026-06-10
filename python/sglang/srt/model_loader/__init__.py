# 模型加载器模块初始化文件，提供模型加载的入口函数和公共API导出
# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明：vLLM项目的贡献者
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/__init__.py  # 改编自vLLM项目的模型加载器初始化模块

from __future__ import annotations  # 启用延迟注解评估

from typing import TYPE_CHECKING  # 导入类型检查常量

from torch import nn  # 导入PyTorch神经网络模块

from sglang.srt.model_loader.loader import BaseModelLoader, get_model_loader  # 导入基础模型加载器类和获取模型加载器函数
from sglang.srt.model_loader.utils import (  # 导入模型加载工具函数
    get_architecture_class_name,  # 获取架构类名
    get_model_architecture,  # 获取模型架构
)

if TYPE_CHECKING:  # 仅在类型检查时执行以下导入
    from sglang.srt.configs.device_config import DeviceConfig  # 导入设备配置类
    from sglang.srt.configs.load_config import LoadConfig  # 导入加载配置类
    from sglang.srt.configs.model_config import ModelConfig  # 导入模型配置类


def get_model(  # 获取并加载模型实例，返回nn.Module对象
    *,
    model_config: ModelConfig,  # 模型配置参数
    load_config: LoadConfig,  # 加载配置参数
    device_config: DeviceConfig,  # 设备配置参数
) -> nn.Module:  # 返回PyTorch模块
    loader = get_model_loader(load_config, model_config)  # 根据加载配置和模型配置获取对应的模型加载器
    return loader.load_model(  # 使用加载器加载模型并返回
        model_config=model_config,  # 传入模型配置
        device_config=device_config,  # 传入设备配置
    )


__all__ = [  # 模块公共API导出列表
    "get_model",  # 获取模型函数
    "get_model_loader",  # 获取模型加载器函数
    "BaseModelLoader",  # 基础模型加载器类
    "get_architecture_class_name",  # 获取架构类名函数
    "get_model_architecture",  # 获取模型架构函数
]
