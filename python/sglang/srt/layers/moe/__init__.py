# MoE（混合专家）模块的初始化文件，导出MoE运行器及相关配置和工具函数
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig  # 导入MoE运行器及其配置类
from sglang.srt.layers.moe.utils import (  # 导入MoE工具函数和枚举
    DeepEPMode,  # DeepEP模式枚举
    MoeA2ABackend,  # MoE全互联后端枚举
    MoeRunnerBackend,  # MoE运行器后端枚举
    get_deepep_config,  # 获取DeepEP配置
    get_deepep_mode,  # 获取DeepEP模式
    get_moe_a2a_backend,  # 获取MoE全互联后端
    get_moe_runner_backend,  # 获取MoE运行器后端
    get_tbo_token_distribution_threshold,  # 获取TBO令牌分布阈值
    initialize_moe_config,  # 初始化MoE配置
    is_tbo_enabled,  # 判断TBO是否启用
    should_skip_post_experts_all_reduce,  # 判断是否应跳过专家后的全归约
    should_use_dp_reduce_scatterv,  # 判断是否应使用数据并行归约散射v
    should_use_flashinfer_cutlass_moe_fp4_allgather,  # 判断是否应使用FlashInfer CUTLASS MoE FP4全收集
)

__all__ = [  # 模块公开导出列表
    "DeepEPMode",  # DeepEP模式
    "MoeA2ABackend",  # MoE全互联后端
    "MoeRunner",  # MoE运行器
    "MoeRunnerConfig",  # MoE运行器配置
    "MoeRunnerBackend",  # MoE运行器后端
    "initialize_moe_config",  # 初始化MoE配置
    "get_moe_a2a_backend",  # 获取MoE全互联后端
    "get_moe_runner_backend",  # 获取MoE运行器后端
    "get_deepep_mode",  # 获取DeepEP模式
    "should_skip_post_experts_all_reduce",  # 判断是否跳过专家后全归约
    "should_use_dp_reduce_scatterv",  # 判断是否使用DP归约散射v
    "should_use_flashinfer_cutlass_moe_fp4_allgather",  # 判断是否使用FlashInfer CUTLASS MoE FP4全收集
    "is_tbo_enabled",  # 判断TBO是否启用
    "get_tbo_token_distribution_threshold",  # 获取TBO令牌分布阈值
    "get_deepep_config",  # 获取DeepEP配置
]
