# NSA（Native Sparse Attention）模块的向后兼容重导出垫片，已弃用，请使用DSA模块替代
# [Deprecated] attention/nsa/ is a thin re-export shim for backward compatibility. # NSA注意力模块的向后兼容重导出垫片（已弃用）
# Use attention/dsa/ instead. This directory will be removed in a future release. # 请改用attention/dsa/，此目录将在未来版本中移除
import warnings  # 导入警告模块

warnings.warn(  # 发出弃用警告
    "sglang.srt.layers.attention.nsa is deprecated; "  # NSA模块已弃用提示信息
    "use sglang.srt.layers.attention.dsa instead.",  # 提示使用DSA模块替代
    DeprecationWarning,  # 弃用警告类型
    stacklevel=2,  # 堆栈层级设为2，指向调用者
)
from sglang.srt.layers.attention.dsa import *  # noqa: F401, F403 从DSA模块导入所有公开符号 # noqa: F401, F403（忽略未使用和通配符导入警告）
