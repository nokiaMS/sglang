# 文件说明：LoRA CSGMV内核自动调优块大小配置加载器
# 本文件实现了从JSON文件加载预调优的LoRA内核块大小配置的功能。
# 遵循与fused_moe_triton_config.py相同的模式：离线调优脚本写入JSON文件，
# 服务器启动时读取最佳块大小，内核使用这些配置代替硬编码默认值。

"""
Configuration loader for auto-tuned LoRA CSGMV kernel block sizes.

Follows the same pattern as fused_moe_triton_config.py:
- Offline tuning script writes JSON files keyed by chunk_size (BLOCK_M)
- At server startup, the config loader reads the best block sizes for each kernel
- Kernels use these instead of hardcoded defaults

Config file naming: lora_{kernel},K={K},R={R},S={S},device={device}.json
Where kernel is "shrink" or "expand", K is input_dim, R is max_rank, S is num_slices.

Config file format (keyed by chunk_size):
{
    "16": {"BLOCK_N": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
    "32": {"BLOCK_N": 32, "BLOCK_K": 128, "num_warps": 4, "num_stages": 4},
    "128": {"BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 8, "num_stages": 3}
}

Usage:
    python3 benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --model Qwen/Qwen3-Embedding-0.6B --max-lora-rank 64

    # Configs saved to python/sglang/srt/lora/triton_ops/configs/

    # Server automatically picks them up:
    python3 -m sglang.launch_server --model ... --enable-lora --lora-backend csgmv
"""

from __future__ import annotations  # 启用延迟类型注解求值

import functools  # 导入functools模块，用于lru_cache装饰器
import json  # 导入JSON解析模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
from typing import Any, Dict, Optional  # 导入类型提示

import triton  # 导入Triton框架

from sglang.srt.utils import get_device_name  # 导入设备名称获取工具

logger = logging.getLogger(__name__)  # 创建模块级日志记录器


def get_lora_config_file_name(  # 生成LoRA内核配置文件名
    kernel: str,  # 内核类型："shrink"或"expand"
    K: int,  # 大维度（shrink为input_dim，expand为output_dim）
    R: int,  # 最大LoRA秩
    S: int,  # 切片数（qkv=3, gate_up=2, 其他=1）
) -> str:  # 返回配置文件名字符串
    """Generate config filename for a LoRA kernel configuration.

    Args:
        kernel: "shrink" or "expand"
        K: The large dimension (input_dim for shrink, output_dim for expand)
        R: The max LoRA rank
        S: num_slices (qkv=3, gate_up=2, others=1)
    """
    device_name = get_device_name().replace(" ", "_")  # 获取设备名并替换空格
    return f"lora_{kernel},K={K},R={R},S={S},device={device_name}.json"  # 格式化配置文件名


@functools.lru_cache  # 使用LRU缓存装饰器，避免重复加载同一配置
def get_lora_configs(  # 从JSON文件加载预调优的LoRA内核配置
    kernel: str,  # 内核类型："shrink"或"expand"
    K: int,  # 大维度
    R: int,  # 最大LoRA秩
    S: int,  # 切片数
) -> Optional[Dict[int, Dict[str, Any]]]:  # 返回chunk_size到块大小配置的映射，或None
    """Load pre-tuned LoRA kernel configs from JSON files.

    Returns a dict mapping chunk_size (BLOCK_M) to block size configs,
    or None if no config file is found.
    """
    json_file_name = get_lora_config_file_name(kernel, K, R, S)  # 生成配置文件名

    config_dir = os.environ.get(
        "SGLANG_LORA_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )  # 获取配置目录，优先使用环境变量
    configs_root = os.path.join(config_dir, "csgmv_configs")  # 拼接csgmv配置根目录

    triton_version = triton.__version__  # 获取Triton版本号
    version_dir = f"triton_{triton_version.replace('.', '_')}"  # 格式化版本目录名

    # Try exact triton version first
    # 首先尝试精确匹配Triton版本
    config_file_path = os.path.join(configs_root, version_dir, json_file_name)  # 拼接精确版本配置路径
    if os.path.exists(config_file_path):  # 如果精确版本配置存在
        with open(config_file_path) as f:  # 打开配置文件
            logger.info(f"Using LoRA {kernel} config from {config_file_path}.")  # 记录使用配置
            return {int(key): val for key, val in json.load(f).items()}  # 解析并返回配置

    # Scan existing version directories as fallback (newest first)
    # 扫描现有版本目录作为回退（从最新版本开始）
    if os.path.isdir(configs_root):  # 如果配置根目录存在
        version_dirs = sorted(
            (d for d in os.listdir(configs_root) if d.startswith("triton_")),  # 筛选triton版本目录
            reverse=True,  # 降序排列（最新版本在前）
        )
        for vdir in version_dirs:  # 遍历版本目录
            if vdir == version_dir:  # 跳过已尝试的精确版本
                continue
            try_path = os.path.join(configs_root, vdir, json_file_name)  # 拼接回退版本配置路径
            if os.path.exists(try_path):  # 如果回退版本配置存在
                with open(try_path) as f:  # 打开配置文件
                    logger.warning(
                        f"LoRA {kernel} config not found for Triton {triton_version}. "
                        f"Falling back to {try_path}."
                    )  # 记录回退警告
                    return {int(key): val for key, val in json.load(f).items()}  # 解析并返回配置

    return None  # 未找到配置，返回None


# Default block sizes (current hardcoded values)
# 默认块大小（当前硬编码值）
DEFAULT_SHRINK_CONFIG = {"BLOCK_N": 16, "BLOCK_K": 256}  # shrink内核默认配置
DEFAULT_EXPAND_CONFIG = {"BLOCK_N": 64, "BLOCK_K": 16}  # expand内核默认配置

# Track which configs have been logged to avoid spamming on every forward pass
# 跟踪已记录的配置，避免每次前向传播时重复打日志
_logged_configs: set = set()  # 已记录配置的集合


def get_lora_shrink_config(  # 获取CSGMV shrink（lora_a）内核的块大小配置
    K: int,  # input_dim
    R: int,  # max_rank
    num_slices: int,  # 切片数（qkv=3, gate_up=2, 其他=1）
    chunk_size: int,  # BLOCK_M值（= batch_info.max_len）
) -> Dict[str, int]:  # 返回块大小配置字典
    """Get block sizes for the CSGMV shrink (lora_a) kernel.

    Args:
        K: input_dim
        R: max_rank
        num_slices: number of slices (qkv=3, gate_up=2, others=1)
        chunk_size: BLOCK_M value (= batch_info.max_len)
    """
    log_key = ("shrink", K, R, num_slices, chunk_size)  # 构建日志键
    configs = get_lora_configs("shrink", K, R, num_slices)  # 加载shrink配置
    if configs is not None:  # 如果找到了配置
        config = configs.get(chunk_size)  # 获取精确chunk_size的配置
        if config is None:  # 如果没有精确匹配
            closest = min(configs.keys(), key=lambda x: abs(x - chunk_size))  # 找最接近的chunk_size
            config = configs[closest]  # 使用最接近的配置
            if log_key not in _logged_configs:  # 如果尚未记录过
                _logged_configs.add(log_key)  # 添加到已记录集合
                logger.info(
                    f"LoRA shrink (K={K}, R={R}): no config for chunk_size={chunk_size}, "
                    f"using closest={closest}: {config}"
                )  # 记录使用最接近配置的日志
        else:  # 找到了精确匹配
            if log_key not in _logged_configs:  # 如果尚未记录过
                _logged_configs.add(log_key)  # 添加到已记录集合
                logger.info(
                    f"LoRA shrink (K={K}, R={R}, chunk_size={chunk_size}): tuned config {config}"
                )  # 记录使用调优配置的日志
        return config  # 返回配置
    if log_key not in _logged_configs:  # 未找到任何配置且未记录过
        _logged_configs.add(log_key)  # 添加到已记录集合
        logger.info(
            f"LoRA shrink (K={K}, R={R}): no tuned config, using defaults {DEFAULT_SHRINK_CONFIG}"
        )  # 记录使用默认配置的日志
    return dict(DEFAULT_SHRINK_CONFIG)  # 返回默认配置


def get_lora_expand_config(  # 获取CSGMV expand（lora_b）内核的块大小配置
    K: int,  # output_dim
    R: int,  # max_rank
    num_slices: int,  # 切片数（qkv=3, gate_up=2, 其他=1）
    chunk_size: int,  # BLOCK_M值（= batch_info.max_len）
) -> Dict[str, int]:  # 返回块大小配置字典
    """Get block sizes for the CSGMV expand (lora_b) kernel.

    Args:
        K: output_dim
        R: max_rank
        num_slices: number of slices (qkv=3, gate_up=2, others=1)
        chunk_size: BLOCK_M value (= batch_info.max_len)
    """
    log_key = ("expand", K, R, num_slices, chunk_size)  # 构建日志键
    configs = get_lora_configs("expand", K, R, num_slices)  # 加载expand配置
    if configs is not None:  # 如果找到了配置
        config = configs.get(chunk_size)  # 获取精确chunk_size的配置
        if config is None:  # 如果没有精确匹配
            closest = min(configs.keys(), key=lambda x: abs(x - chunk_size))  # 找最接近的chunk_size
            config = configs[closest]  # 使用最接近的配置
            if log_key not in _logged_configs:  # 如果尚未记录过
                _logged_configs.add(log_key)  # 添加到已记录集合
                logger.info(
                    f"LoRA expand (K={K}, R={R}): no config for chunk_size={chunk_size}, "
                    f"using closest={closest}: {config}"
                )  # 记录使用最接近配置的日志
        else:  # 找到了精确匹配
            if log_key not in _logged_configs:  # 如果尚未记录过
                _logged_configs.add(log_key)  # 添加到已记录集合
                logger.info(
                    f"LoRA expand (K={K}, R={R}, chunk_size={chunk_size}): tuned config {config}"
                )  # 记录使用调优配置的日志
        return config  # 返回配置
    if log_key not in _logged_configs:  # 未找到任何配置且未记录过
        _logged_configs.add(log_key)  # 添加到已记录集合
        logger.info(
            f"LoRA expand (K={K}, R={R}): no tuned config, using defaults {DEFAULT_EXPAND_CONFIG}"
        )  # 记录使用默认配置的日志
    return dict(DEFAULT_EXPAND_CONFIG)  # 返回默认配置
