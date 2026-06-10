# Quark 量化方案子包初始化模块
# 本文件导出 Quark 量化框架支持的所有量化方案，
# 包括线性层方案（QuarkW8A8Fp8、QuarkW4A4MXFP4）
# 和 MoE 方案（QuarkW8A8FP8MoE、QuarkW4A4MXFp4MoE）。

# SPDX-License-Identifier: Apache-2.0

from .quark_scheme import QuarkLinearScheme, QuarkMoEScheme  # 导入 Quark 线性和 MoE 量化方案基类 # import Quark linear and MoE scheme base classes
from .quark_w4a4_mxfp4 import QuarkW4A4MXFP4  # 导入 W4A4 MX-FP4 量化方案 # import W4A4 MX-FP4 quant scheme
from .quark_w4a4_mxfp4_moe import QuarkW4A4MXFp4MoE  # 导入 W4A4 MX-FP4 MoE 量化方案 # import W4A4 MX-FP4 MoE quant scheme
from .quark_w8a8_fp8 import QuarkW8A8Fp8  # 导入 W8A8 FP8 量化方案 # import W8A8 FP8 quant scheme
from .quark_w8a8_fp8_moe import QuarkW8A8FP8MoE  # 导入 W8A8 FP8 MoE 量化方案 # import W8A8 FP8 MoE quant scheme

__all__ = [  # 模块公开接口列表 # module public interface list
    "QuarkLinearScheme",  # Quark 线性层量化方案基类 # Quark linear scheme base class
    "QuarkMoEScheme",  # Quark MoE 量化方案基类 # Quark MoE scheme base class
    "QuarkW4A4MXFP4",  # W4A4 MX-FP4 量化方案 # W4A4 MX-FP4 quant scheme
    "QuarkW8A8Fp8",  # W8A8 FP8 量化方案 # W8A8 FP8 quant scheme
    "QuarkW4A4MXFp4MoE",  # W4A4 MX-FP4 MoE 量化方案 # W4A4 MX-FP4 MoE quant scheme
    "QuarkW8A8FP8MoE",  # W8A8 FP8 MoE 量化方案 # W8A8 FP8 MoE quant scheme
]
