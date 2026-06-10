# Mamba操作模块初始化文件
# 该模块导出Mamba SSM相关的核心操作，包括填充槽ID、
# 分块扫描组合操作、选择性状态更新及其后端初始化函数。

from .mamba_ssm import PAD_SLOT_ID  # 从mamba_ssm模块导入填充槽ID常量
from .ssd_combined import mamba_chunk_scan_combined  # 从ssd_combined模块导入Mamba分块扫描组合操作
from .ssu_dispatch import (  # 从ssu_dispatch模块导入选择性状态更新相关函数
    initialize_mamba_selective_state_update_backend,  # 初始化Mamba选择性状态更新后端
    selective_state_update,  # 选择性状态更新函数
)

__all__ = [  # 模块公开接口列表
    "PAD_SLOT_ID",  # 填充槽ID常量
    "selective_state_update",  # 选择性状态更新函数
    "mamba_chunk_scan_combined",  # Mamba分块扫描组合操作
    "initialize_mamba_selective_state_update_backend",  # 初始化选择性状态更新后端
]
