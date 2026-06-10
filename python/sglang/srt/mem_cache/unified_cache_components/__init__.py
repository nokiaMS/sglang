# 文件说明：统一缓存组件（unified_cache_components）包的初始化模块
# 导出缓存组件相关的类和函数，包括FullComponent、MambaComponent、SWAComponent、TreeComponent等组件类型，
# 以及ComponentType、ComponentData、EvictLayer、CacheTransferPhase、LRURefreshPhase等核心数据结构。

from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent  # 导入Full组件
from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent  # 导入Mamba组件
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent  # 导入SWA组件
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # 导入树组件及相关类型
    _NUM_COMPONENT_TYPES,  # 组件类型数量
    BASE_COMPONENT_TYPE,  # 基础组件类型
    CacheTransferPhase,  # 缓存传输阶段枚举
    ComponentData,  # 组件数据类
    ComponentType,  # 组件类型枚举
    EvictLayer,  # 淘汰层枚举
    LRURefreshPhase,  # LRU刷新阶段枚举
    TreeComponent,  # 树组件类
    get_and_increase_time_counter,  # 获取并递增时间计数器
    next_component_uuid,  # 生成下一个组件UUID
)

__all__ = [  # 公开导出列表
    "BASE_COMPONENT_TYPE",  # 基础组件类型
    "ComponentData",  # 组件数据类
    "ComponentType",  # 组件类型枚举
    "EvictLayer",  # 淘汰层枚举
    "FullComponent",  # Full组件
    "CacheTransferPhase",  # 缓存传输阶段
    "LRURefreshPhase",  # LRU刷新阶段
    "MambaComponent",  # Mamba组件
    "SWAComponent",  # SWA组件
    "TreeComponent",  # 树组件
    "_NUM_COMPONENT_TYPES",  # 组件类型数量
    "next_component_uuid",  # 生成下一个组件UUID
    "get_and_increase_time_counter",  # 获取并递增时间计数器
]
