# 压缩张量量化方案的包初始化模块
# 导出所有支持的压缩张量量化方案类和常量
# SPDX-License-Identifier: Apache-2.0

from .compressed_tensors_scheme import (  # 从方案基类模块导入线性方案和MoE方案基类
    CompressedTensorsLinearScheme,  # 压缩张量线性层量化方案基类
    CompressedTensorsMoEScheme,  # 压缩张量MoE层量化方案基类
)
from .compressed_tensors_w4a4_mxint4_moe import CompressedTensorsMxInt4MoE  # 导入MX INT4量化MoE方案
from .compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4  # 导入NV FP4 W4A4量化方案
from .compressed_tensors_w4a4_nvfp4_moe import CompressedTensorsW4A4Nvfp4MoE  # 导入NV FP4 MoE量化方案
from .compressed_tensors_w4a8_int8_moe import NPUCompressedTensorsW4A8Int8DynamicMoE  # 导入NPU W4A8 INT8动态MoE方案
from .compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8  # 导入W8A8 FP8量化方案
from .compressed_tensors_w8a8_fp8_moe import CompressedTensorsW8A8Fp8MoE  # 导入W8A8 FP8 MoE量化方案
from .compressed_tensors_w8a8_int8 import (  # 从W8A8 INT8模块导入量化方案
    CompressedTensorsW8A8Int8,  # W8A8 INT8量化方案
    NPUCompressedTensorsW8A8Int8,  # NPU版本的W8A8 INT8量化方案
)
from .compressed_tensors_w8a8_int8_moe import NPUCompressedTensorsW8A8Int8DynamicMoE  # 导入NPU W8A8 INT8动态MoE方案
from .compressed_tensors_w8a16_fp8 import CompressedTensorsW8A16Fp8  # 导入W8A16 FP8量化方案
from .compressed_tensors_wNa16 import WNA16_SUPPORTED_BITS, CompressedTensorsWNA16  # 导入WNA16量化方案及支持的位数常量
from .compressed_tensors_wNa16_moe import (  # 从WNA16 MoE模块导入量化方案
    CompressedTensorsWNA16MoE,  # WNA16 MoE量化方案
    CompressedTensorsWNA16TritonMoE,  # WNA16 Triton MoE量化方案
    NPUCompressedTensorsW4A16Int4DynamicMoE,  # NPU W4A16 INT4动态MoE量化方案
)

__all__ = [  # 模块公开导出的符号列表
    "CompressedTensorsLinearScheme",  # 线性层方案基类
    "CompressedTensorsMoEScheme",  # MoE层方案基类
    "CompressedTensorsW8A8Fp8",  # W8A8 FP8方案
    "CompressedTensorsW8A8Fp8MoE",  # W8A8 FP8 MoE方案
    "CompressedTensorsW8A16Fp8",  # W8A16 FP8方案
    "CompressedTensorsW8A8Int8",  # W8A8 INT8方案
    "NPUCompressedTensorsW8A8Int8",  # NPU W8A8 INT8方案
    "NPUCompressedTensorsW8A8Int8DynamicMoE",  # NPU W8A8 INT8动态MoE方案
    "CompressedTensorsWNA16",  # WNA16方案
    "CompressedTensorsWNA16MoE",  # WNA16 MoE方案
    "CompressedTensorsWNA16TritonMoE",  # WNA16 Triton MoE方案
    "NPUCompressedTensorsW4A16Int4DynamicMoE",  # NPU W4A16 INT4动态MoE方案
    "WNA16_SUPPORTED_BITS",  # WNA16支持的位宽集合
    "CompressedTensorsW4A4Fp4",  # W4A4 FP4方案
    "CompressedTensorsW4A4Nvfp4MoE",  # W4A4 NVFP4 MoE方案
    "NPUCompressedTensorsW4A8Int8DynamicMoE",  # NPU W4A8 INT8动态MoE方案
    "CompressedTensorsMxInt4MoE",  # MX INT4 MoE方案
]
