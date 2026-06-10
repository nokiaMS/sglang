# ModelOpt 相关常量定义模块
# 本文件定义了 NVIDIA ModelOpt 工具支持的量化配置选项映射
"""
ModelOpt related constants
"""  # ModelOpt 相关常量

QUANT_CFG_CHOICES = {  # 量化配置选项字典，键为量化类型简称，值为 ModelOpt 中的配置名
    "fp8": "FP8_DEFAULT_CFG",  # FP8 量化默认配置
    "int4_awq": "INT4_AWQ_CFG",  # INT4 AWQ 量化配置 # TODO: add support for int4_awq # TODO: 添加 int4_awq 支持
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",  # W4A8 AWQ 量化配置 # TODO: add support for w4a8_awq # TODO: 添加 w4a8_awq 支持
    "nvfp4": "NVFP4_DEFAULT_CFG",  # NVFP4 量化默认配置
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",  # NVFP4 AWQ 轻量级量化配置 # TODO: add support for nvfp4_awq # TODO: 添加 nvfp4_awq 支持
}
