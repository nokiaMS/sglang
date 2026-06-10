# ROCm MX-FP4 量化工具函数模块，提供融合 RMS MX-FP4 量化、扁平化 MX-FP4 量化和批量 GEMM 预量化功能
from aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant import (  # 导入批量 AFP4-WFP4 GEMM 预量化函数
    batched_gemm_afp4wfp4_pre_quant,
)
from aiter.ops.triton.fused_mxfp4_quant import (  # 导入融合 MX-FP4 量化函数
    fused_flatten_mxfp4_quant,  # 扁平化 MX-FP4 量化
    fused_rms_mxfp4_quant,  # 融合 RMS MX-FP4 量化
)

__all__ = [  # 导出列表
    "fused_rms_mxfp4_quant",  # 融合 RMS MX-FP4 量化函数
    "fused_flatten_mxfp4_quant",  # 扁平化 MX-FP4 量化函数
    "batched_gemm_afp4wfp4_pre_quant",  # 批量 AFP4-WFP4 GEMM 预量化函数
]
