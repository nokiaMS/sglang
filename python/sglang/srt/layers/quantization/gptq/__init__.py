# GPTQ量化模块初始化文件：导出GPTQ量化配置、线性方法、MoE方法和校验函数
# SPDX-License-Identifier: Apache-2.0  # Apache 2.0许可证声明

from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (  # 从GPU硬件后端导入GPTQ MoE重打包函数
    gptq_marlin_moe_repack,  # GPTQ Marlin MoE权重重打包函数
)

from .gptq import (  # 从当前包的gptq模块导入各种配置和方法类
    GPTQAscendConfig,  # GPTQ Ascend NPU配置类
    GPTQConfig,  # GPTQ基础配置类
    GPTQLinearAscendMethod,  # GPTQ Ascend NPU线性方法类
    GPTQLinearMethod,  # GPTQ线性方法类
    GPTQMarlinConfig,  # GPTQ Marlin配置类
    GPTQMarlinLinearMethod,  # GPTQ Marlin线性方法类
    GPTQMarlinMoEMethod,  # GPTQ Marlin MoE方法类
    GPTQMoEAscendMethod,  # GPTQ Ascend NPU MoE方法类
    check_marlin_format,  # 检查是否为Marlin格式的函数
)
from .schemes import (  # 从schemes子包导入各种量化方案类
    GPTQAscendLinearScheme,  # GPTQ Ascend NPU线性方案
    GPTQLinearScheme,  # GPTQ线性方案
    GPTQMarlinLinearScheme,  # GPTQ Marlin线性方案
    GPTQMarlinMoEScheme,  # GPTQ Marlin MoE方案
    GPTQMoEAscendScheme,  # GPTQ Ascend NPU MoE方案
)

__all__ = [  # 定义模块公开导出的符号列表
    "GPTQConfig",  # GPTQ基础配置类
    "GPTQAscendConfig",  # GPTQ Ascend NPU配置类
    "GPTQMarlinConfig",  # GPTQ Marlin配置类
    "GPTQLinearMethod",  # GPTQ线性方法类
    "GPTQMoEAscendMethod",  # GPTQ Ascend NPU MoE方法类
    "GPTQMarlinLinearMethod",  # GPTQ Marlin线性方法类
    "GPTQLinearAscendMethod",  # GPTQ Ascend NPU线性方法类
    "GPTQMarlinMoEMethod",  # GPTQ Marlin MoE方法类
    "GPTQLinearScheme",  # GPTQ线性方案
    "GPTQAscendLinearScheme",  # GPTQ Ascend NPU线性方案
    "GPTQMarlinLinearScheme",  # GPTQ Marlin线性方案
    "GPTQMoEAscendScheme",  # GPTQ Ascend NPU MoE方案
    "GPTQMarlinMoEScheme",  # GPTQ Marlin MoE方案
    "check_marlin_format",  # 检查Marlin格式函数
    "gptq_marlin_moe_repack",  # GPTQ Marlin MoE重打包函数
]
