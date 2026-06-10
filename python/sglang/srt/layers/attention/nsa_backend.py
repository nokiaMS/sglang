# NSA后端兼容性重导出模块
# 此文件已弃用，仅为向后兼容而保留。实际实现已迁移到dsa_backend.py。
# 使用sglang.srt.layers.attention.dsa_backend替代此模块。

# [Deprecated] nsa_backend.py is a thin re-export shim for backward compatibility.
# Use dsa_backend.py instead. This file will be removed in a future release.
# [已弃用] nsa_backend.py是向后兼容的薄重导出垫片。
# 请使用dsa_backend.py替代。此文件将在未来版本中删除。
import warnings  # 警告模块

warnings.warn(  # 发出弃用警告
    "sglang.srt.layers.attention.nsa_backend is deprecated; "  # nsa_backend已弃用
    "use sglang.srt.layers.attention.dsa_backend instead.",  # 请使用dsa_backend
    DeprecationWarning,  # 弃用警告类型
    stacklevel=2,  # 堆栈层级
)
from sglang.srt.layers.attention.dsa_backend import *  # noqa: F401, F403  # 从dsa_backend重导出所有
from sglang.srt.layers.attention.dsa_backend import (  # noqa: F401  # 显式重导出供外部引用的类
    DeepseekSparseAttnBackend,  # Deepseek稀疏注意力后端
    DeepseekSparseAttnMultiStepBackend,  # Deepseek稀疏注意力多步后端
    DSAFlashMLAMetadata,  # DSA Flash MLA元数据
    DSAIndexerMetadata,  # DSA索引器元数据
    DSAMetadata,  # DSA元数据
    NativeSparseAttnBackend,  # 原生稀疏注意力后端
    NativeSparseAttnMultiStepBackend,  # 原生稀疏注意力多步后端
    NSAFlashMLAMetadata,  # NSA Flash MLA元数据（兼容别名）
    NSAIndexerMetadata,  # NSA索引器元数据（兼容别名）
    NSAMetadata,  # NSA元数据（兼容别名）
)
