# 稀疏注意力后端适配器模块的初始化文件，导出BackendAdaptor、FlashAttentionAdaptor和DSABackendAdaptor类
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (  # 从backend_adaptor模块导入后端适配器类
    BackendAdaptor,  # 后端适配器基类
    DSABackendAdaptor,  # DSA（DeepSeek稀疏注意力）后端适配器
    FlashAttentionAdaptor,  # FlashAttention后端适配器
)

__all__ = ["BackendAdaptor", "FlashAttentionAdaptor", "DSABackendAdaptor"]  # 模块公开导出的类列表
