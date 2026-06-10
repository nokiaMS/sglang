# DeepSeek注意力前向方法子包初始化文件
# 导出DeepSeek模型中各种注意力机制的前向计算方法类，包括MHA和MLA的不同实现

from .forward_methods import AttnForwardMethod  # 导入注意力前向方法枚举基类
from .forward_mha import DeepseekMHAForwardMixin  # 导入DeepSeek MHA前向计算混入类
from .forward_mla import DeepseekMLAForwardMixin  # 导入DeepSeek MLA前向计算混入类
from .forward_mla_fused_rope_cpu import DeepseekMLACpuForwardMixin  # 导入DeepSeek MLA CPU融合RoPE前向计算混入类
from .forward_mla_fused_rope_rocm import DeepseekMLARocmForwardMixin  # 导入DeepSeek MLA ROCm融合RoPE前向计算混入类

__all__ = [  # 模块公开导出列表
    "AttnForwardMethod",  # 注意力前向方法枚举
    "DeepseekMHAForwardMixin",  # MHA前向计算混入类
    "DeepseekMLACpuForwardMixin",  # MLA CPU融合RoPE前向计算混入类
    "DeepseekMLAForwardMixin",  # MLA前向计算混入类
    "DeepseekMLARocmForwardMixin",  # MLA ROCm融合RoPE前向计算混入类
]
