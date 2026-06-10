# AWQ（Activation-aware Weight Quantization）量化模块的初始化文件
# 本模块导出AWQ量化相关的配置类、线性层方法、MoE方法、反量化函数及各种量化方案

# SPDX-License-Identifier: Apache-2.0

from .awq import (  # 从awq子模块导入AWQ核心配置和方法类
    AWQConfig,  # AWQ基本配置类
    AWQCPUConfig,  # AWQ CPU配置类
    AWQLinearMethod,  # AWQ线性层量化方法
    AWQMarlinConfig,  # AWQ Marlin配置类
    AWQMoEMethod,  # AWQ MoE（混合专家）量化方法
)
from .awq_triton import awq_dequantize_decomposition, awq_dequantize_triton  # 从triton子模块导入AWQ反量化函数
from .schemes import (  # 从schemes子模块导入各种AWQ量化方案
    AWQAscendLinearScheme,  # 昇腾平台AWQ线性层方案
    AWQAscendMoEScheme,  # 昇腾平台AWQ MoE方案
    AWQLinearScheme,  # 通用AWQ线性层方案
    AWQMarlinLinearScheme,  # Marlin后端AWQ线性层方案
    AWQMoEScheme,  # 通用AWQ MoE方案
)

__all__ = [  # 模块公开导出的符号列表
    "AWQConfig",  # AWQ基本配置类
    "AWQCPUConfig",  # AWQ CPU配置类
    "AWQMarlinConfig",  # AWQ Marlin配置类
    "AWQLinearMethod",  # AWQ线性层量化方法
    "AWQMoEMethod",  # AWQ MoE量化方法
    "AWQLinearScheme",  # 通用AWQ线性层方案
    "AWQMarlinLinearScheme",  # Marlin后端AWQ线性层方案
    "AWQAscendLinearScheme",  # 昇腾平台AWQ线性层方案
    "AWQMoEScheme",  # 通用AWQ MoE方案
    "AWQAscendMoEScheme",  # 昇腾平台AWQ MoE方案
    "awq_dequantize_triton",  # Triton实现的AWQ反量化函数
    "awq_dequantize_decomposition",  # 基于分解的AWQ反量化函数
]
