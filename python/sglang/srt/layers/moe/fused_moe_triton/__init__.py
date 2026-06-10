# 文件说明：fused_moe_triton包的初始化模块
# 本模块负责导出FusedMoE层类及相关MoE工具函数，
# 包括融合专家计算、配置获取、块大小对齐等核心接口

from sglang.srt.layers.moe.fused_moe_triton.layer import (  # 从layer模块导入MoE层类
    FusedMoE,  # 融合MoE层类
    FusedMoeWeightScaleSupported,  # MoE权重缩放支持类型
)
from sglang.srt.layers.moe.moe_runner.triton_utils import (  # 从triton_utils导入MoE工具函数
    fused_experts,  # 融合专家计算函数
    get_config,  # 获取MoE配置函数
    get_config_file_name,  # 获取配置文件名函数
    moe_align_block_size,  # MoE块大小对齐函数
    override_config,  # 覆盖配置函数
    try_get_optimal_moe_config,  # 尝试获取最优MoE配置函数
)

__all__ = [  # 定义模块公开接口列表
    "FusedMoE",  # 融合MoE层类
    "FusedMoeWeightScaleSupported",  # MoE权重缩放支持类型
    "override_config",  # 覆盖配置函数
    "get_config",  # 获取MoE配置函数
    "fused_experts",  # 融合专家计算函数
    "get_config_file_name",  # 获取配置文件名函数
    "moe_align_block_size",  # MoE块大小对齐函数
    "try_get_optimal_moe_config",  # 尝试获取最优MoE配置函数
]
