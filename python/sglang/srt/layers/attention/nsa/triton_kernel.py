# NSA (Native Sparse Attention) Triton 内核模块的废弃兼容性重新导出垫片
# 实际实现已迁移至 dsa.triton_kernel，此文件仅用于向后兼容
# [Deprecated] Re-export shim for backward compatibility. Use dsa.triton_kernel instead. # 废弃的重新导出垫片，用于向后兼容，请改用 dsa.triton_kernel
import warnings # 导入警告模块

warnings.warn( # 发出废弃警告
    "sglang.srt.layers.attention.nsa.triton_kernel is deprecated; " # nsa.triton_kernel 已废弃
    "use sglang.srt.layers.attention.dsa.triton_kernel instead.", # 请改用 dsa.triton_kernel
    DeprecationWarning, # 废弃警告类型
    stacklevel=2, # 堆栈层级设为2，指向调用者
)
from sglang.srt.layers.attention.dsa.triton_kernel import *  # noqa: F401, F403 # 从 dsa 模块重新导出所有内容
