# SGLang HiCache 存储后端模块的初始化文件
# 该模块提供了存储后端的工厂类，用于创建不同类型的存储后端实例

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project  # SPDX版权声明：SGLang项目的贡献者

"""Storage backend module for SGLang HiCache."""  # SGLang HiCache的存储后端模块

from .backend_factory import StorageBackendFactory  # 从backend_factory模块导入存储后端工厂类

__all__ = [  # 模块公开导出的符号列表
    "StorageBackendFactory",  # 存储后端工厂类
]
