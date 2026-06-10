# 文件说明：NSA注意力层的索引器模块（已弃用），向后兼容的重导出垫片，实际实现已迁移至dsa.dsa_indexer
# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.  # [已弃用] 向后兼容的重导出垫片，请使用dsa.dsa_indexer替代
import warnings  # 导入警告模块

warnings.warn(  # 发出弃用警告
    "sglang.srt.layers.attention.nsa.nsa_indexer is deprecated; "  # nsa.nsa_indexer已弃用
    "use sglang.srt.layers.attention.dsa.dsa_indexer instead.",  # 请改用dsa.dsa_indexer
    DeprecationWarning,  # 弃用警告类型
    stacklevel=2,  # 堆栈层级设为2，指向调用方
)
from sglang.srt.layers.attention.dsa.dsa_indexer import *  # noqa: F401, F403  # 从dsa模块重导出所有内容，# noqa: F401, F403 忽略未使用导入警告
