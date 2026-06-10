# 本文件是 EVS（Efficient Video Sampling）模块的初始化文件，
# 导出 EVS、EVSConfig、EVSEmbeddingResult 和 EVSProcessor 四个核心组件
"""https://arxiv.org/abs/2510.14624: Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference"""

from .evs_module import EVS, EVSConfig, EVSEmbeddingResult  # 从 evs_module 导入 EVS 核心类、配置类和嵌入结果类
from .evs_processor import EVSProcessor  # 从 evs_processor 导入 EVS 处理器类

__all__ = [  # 定义模块公开接口列表
    "EVS",  # EVS 剪枝模块基类
    "EVSConfig",  # EVS 配置类
    "EVSEmbeddingResult",  # EVS 嵌入结果类
    "EVSProcessor",  # EVS 处理器类
]
