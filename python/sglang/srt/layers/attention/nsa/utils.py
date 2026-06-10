# NSA工具模块 - 已废弃的向后兼容性重新导出垫片，实际请使用DSA工具模块
# [Deprecated] Re-export shim for backward compatibility. Use dsa.utils instead. # [已废弃] 重新导出垫片用于向后兼容，请使用dsa.utils代替
import warnings  # 导入警告模块 # 导入标准库warnings用于发出弃用警告

warnings.warn(  # 发出弃用警告 # 调用warnings.warn发出弃用警告
    "sglang.srt.layers.attention.nsa.utils is deprecated; "  # NSA工具模块已废弃的提示信息
    "use sglang.srt.layers.attention.dsa.utils instead.",  # 应使用DSA工具模块的提示信息
    DeprecationWarning,  # 弃用警告类型 # 指定警告类别为DeprecationWarning
    stacklevel=2,  # 堆栈级别为2，指向调用者 # 设置stacklevel使警告指向本模块的调用者
)
from sglang.srt.layers.attention.dsa.utils import *  # noqa: F401, F403 # 从DSA工具模块导入所有内容，作为向后兼容的替代 # noqa: F401, F403 忽略未使用的导入警告
