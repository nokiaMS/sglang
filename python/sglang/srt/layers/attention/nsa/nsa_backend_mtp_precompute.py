# 文件说明：NSA注意力层的MTP后端预计算模块（已弃用），向后兼容的重导出垫片，实际实现已迁移至dsa.dsa_backend_mtp_precompute
# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_backend_mtp_precompute instead.  # [已弃用] 向后兼容的重导出垫片，请使用dsa.dsa_backend_mtp_precompute替代
import warnings  # 导入警告模块

warnings.warn(  # 发出弃用警告
    "sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute is deprecated; "  # nsa.nsa_backend_mtp_precompute已弃用
    "use sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute instead.",  # 请改用dsa.dsa_backend_mtp_precompute
    DeprecationWarning,  # 弃用警告类型
    stacklevel=2,  # 堆栈层级设为2，指向调用方
)
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import *  # noqa: F401, F403  # 从dsa模块重导出所有内容，# noqa: F401, F403 忽略未使用导入警告
