# 文件说明：tilelang_kernel.py - TileLang内核的向后兼容重导出垫片模块
# [Deprecated] Re-export shim for backward compatibility. Use dsa.tilelang_kernel instead. # [已弃用] 为向后兼容提供的重导出垫片，请使用 dsa.tilelang_kernel 代替。
import warnings  # 导入警告模块 # import warnings module

warnings.warn(  # 发出弃用警告 # issue deprecation warning
    "sglang.srt.layers.attention.nsa.tilelang_kernel is deprecated; "  # NSA模块的tilelang_kernel已弃用 # NSA module tilelang_kernel is deprecated
    "use sglang.srt.layers.attention.dsa.tilelang_kernel instead.",  # 请改用DSA模块的tilelang_kernel # use DSA module tilelang_kernel instead
    DeprecationWarning,  # 弃用警告类型 # deprecation warning type
    stacklevel=2,  # 堆栈级别设为2，指向调用方 # stack level 2, points to caller
)
from sglang.srt.layers.attention.dsa.tilelang_kernel import *  # noqa: F401, F403 # 从DSA模块重新导出所有内容 # re-export all from DSA module
