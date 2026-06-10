# 稀疏注意力模块的顶层初始化文件，导出稀疏注意力相关的核心类和工厂函数
from sglang.srt.mem_cache.sparsity.algorithms import (  # 从算法子模块导入稀疏注意力算法基类和具体实现
    BaseSparseAlgorithm,  # 稀疏注意力算法抽象基类
    BaseSparseAlgorithmImpl,  # 稀疏注意力算法实现基类
    DeepSeekDSAAlgorithm,  # DeepSeek DSA稀疏注意力算法
    QuestAlgorithm,  # Quest稀疏注意力算法
)
from sglang.srt.mem_cache.sparsity.backend import BackendAdaptor, FlashAttentionAdaptor  # 从后端模块导入后端适配器类
from sglang.srt.mem_cache.sparsity.core import SparseConfig, SparseCoordinator  # 从核心模块导入稀疏配置类和协调器类
from sglang.srt.mem_cache.sparsity.factory import (  # 从工厂模块导入稀疏协调器的工厂函数
    create_sparse_coordinator,  # 创建稀疏协调器实例
    get_sparse_coordinator,  # 获取已注册的稀疏协调器
    parse_hisparse_config,  # 解析HiSparse配置
    register_sparse_coordinator,  # 注册稀疏协调器
)

__all__ = [  # 模块公开导出的符号列表
    "BaseSparseAlgorithm",  # 稀疏注意力算法抽象基类
    "BaseSparseAlgorithmImpl",  # 稀疏注意力算法实现基类
    "QuestAlgorithm",  # Quest稀疏注意力算法
    "DeepSeekDSAAlgorithm",  # DeepSeek DSA稀疏注意力算法
    "BackendAdaptor",  # 后端适配器基类
    "FlashAttentionAdaptor",  # FlashAttention后端适配器
    "SparseConfig",  # 稀疏注意力配置类
    "SparseCoordinator",  # 稀疏注意力协调器类
    "create_sparse_coordinator",  # 创建稀疏协调器函数
    "get_sparse_coordinator",  # 获取稀疏协调器函数
    "parse_hisparse_config",  # 解析HiSparse配置函数
    "register_sparse_coordinator",  # 注册稀疏协调器函数
]
