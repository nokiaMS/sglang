# GPTQ量化方案子包初始化文件：导出各种GPTQ线性方案和MoE方案
# SPDX-License-Identifier: Apache-2.0  # Apache 2.0许可证声明

from .gptq_linear import GPTQAscendLinearScheme, GPTQLinearScheme  # 从gptq_linear模块导入GPTQ线性方案和Ascend线性方案
from .gptq_marlin import GPTQMarlinLinearScheme  # 从gptq_marlin模块导入GPTQ Marlin线性方案
from .gptq_moe import GPTQMarlinMoEScheme, GPTQMoEAscendScheme  # 从gptq_moe模块导入GPTQ Marlin MoE方案和Ascend MoE方案
from .gptq_scheme import GPTQLinearSchemeBase, GPTQMoESchemeBase  # 从gptq_scheme模块导入GPTQ线性方案基类和MoE方案基类

__all__ = [  # 定义模块公开导出的符号列表
    "GPTQLinearSchemeBase",  # GPTQ线性方案基类
    "GPTQMoESchemeBase",  # GPTQ MoE方案基类
    "GPTQLinearScheme",  # GPTQ线性方案
    "GPTQAscendLinearScheme",  # GPTQ Ascend NPU线性方案
    "GPTQMarlinLinearScheme",  # GPTQ Marlin线性方案
    "GPTQMoEAscendScheme",  # GPTQ Ascend NPU MoE方案
    "GPTQMarlinMoEScheme",  # GPTQ Marlin MoE方案
]
