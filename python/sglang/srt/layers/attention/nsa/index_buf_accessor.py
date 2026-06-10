# 文件说明：NSA注意力层的索引缓冲区访问器模块（已弃用），向后兼容的重导出垫片，实际实现已迁移至dsa.index_buf_accessor
# [Deprecated] Re-export shim for backward compatibility. Use dsa.index_buf_accessor instead.  # [已弃用] 向后兼容的重导出垫片，请使用dsa.index_buf_accessor替代
import warnings  # 导入警告模块

warnings.warn(  # 发出弃用警告
    "sglang.srt.layers.attention.nsa.index_buf_accessor is deprecated; "  # nsa.index_buf_accessor已弃用
    "use sglang.srt.layers.attention.dsa.index_buf_accessor instead.",  # 请改用dsa.index_buf_accessor
    DeprecationWarning,  # 弃用警告类型
    stacklevel=2,  # 堆栈层级设为2，指向调用方
)
from sglang.srt.layers.attention.dsa.index_buf_accessor import *  # noqa: F401, F403  # 从dsa模块重导出所有内容，# noqa: F401, F403 忽略未使用导入警告
