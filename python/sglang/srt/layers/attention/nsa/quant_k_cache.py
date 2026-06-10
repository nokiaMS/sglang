# 文件说明：quant_k_cache.py - 量化K缓存的向后兼容重导出垫片模块
# [Deprecated] Re-export shim for backward compatibility. Use dsa.quant_k_cache instead. # [已弃用] 为向后兼容提供的重导出垫片，请使用 dsa.quant_k_cache 代替。
import warnings  # 导入警告模块 # import warnings module

warnings.warn(  # 发出弃用警告 # issue deprecation warning
    "sglang.srt.layers.attention.nsa.quant_k_cache is deprecated; "  # NSA模块的quant_k_cache已弃用 # NSA module quant_k_cache is deprecated
    "use sglang.srt.layers.attention.dsa.quant_k_cache instead.",  # 请改用DSA模块的quant_k_cache # use DSA module quant_k_cache instead
    DeprecationWarning,  # 弃用警告类型 # deprecation warning type
    stacklevel=2,  # 堆栈级别设为2，指向调用方 # stack level 2, points to caller
)
from sglang.srt.layers.attention.dsa.quant_k_cache import *  # noqa: F401, F403 # 从DSA模块重新导出所有内容 # re-export all from DSA module
