# Triton 工具包初始化模块 - 导出融合 MoE 专家函数、配置管理函数及块对齐函数，提供全局配置覆盖机制
from contextlib import contextmanager  # 上下文管理器装饰器
from typing import Any, Dict, Optional  # 类型提示工具

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts  # 融合专家计算函数
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (  # MoE Triton 配置相关函数
    get_config_file_name,  # 获取配置文件名
    try_get_optimal_moe_config,  # 尝试获取最优 MoE 配置
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (  # MoE 块大小对齐函数
    moe_align_block_size,  # 对齐块大小
)

_config: Optional[Dict[str, Any]] = None  # 全局配置字典，初始为None


@contextmanager  # 配置覆盖上下文管理器
def override_config(config):  # 临时覆盖全局配置
    global _config  # 声明使用全局变量
    old_config = _config  # 保存旧配置
    _config = config  # 设置新配置
    yield  # 执行上下文中的代码
    _config = old_config  # 恢复旧配置


def get_config() -> Optional[Dict[str, Any]]:  # 获取当前全局配置
    return _config  # 返回全局配置字典


__all__ = [  # 模块公开导出的符号列表
    "override_config",  # 配置覆盖上下文管理器
    "get_config",  # 获取配置函数
    "fused_experts",  # 融合专家函数
    "get_config_file_name",  # 获取配置文件名函数
    "moe_align_block_size",  # 块大小对齐函数
    "try_get_optimal_moe_config",  # 获取最优配置函数
]
