# 融合 MoE Triton 配置模块 - 管理 MoE 内核的调优配置，包括从磁盘加载预调优配置、生成默认配置、获取最优配置等
from __future__ import annotations  # 启用延迟注解评估，支持前向引用类型

import functools  # 函数工具模块，提供 lru_cache 等装饰器
import json  # JSON 解析模块
import logging  # 日志模块
import os  # 操作系统接口模块
from typing import Any, Dict, List, Optional, Tuple  # 类型提示工具

import torch  # PyTorch 深度学习框架
import triton  # Triton 编译器

from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.utils import get_device_name, is_hip  # 工具函数：获取设备名称、判断是否HIP

logger = logging.getLogger(__name__)  # 获取当前模块的日志器
_is_hip = is_hip()  # 是否为 HIP 平台


def get_config_file_name(  # 生成 MoE 配置文件名，基于专家数、维度、数据类型等参数
    E: int,  # 专家数量
    N: int,  # 中间维度
    dtype: Optional[str],  # 数据类型字符串（可选）
    block_shape: Optional[int] = None,  # 块形状（可选）
    per_channel_quant: bool = False,  # 是否逐通道量化
    down_moe: bool = False,  # 是否为下投影 MoE
) -> str:  # 返回配置文件名字符串
    device_name = get_device_name().replace(" ", "_")  # 获取设备名称并替换空格
    dtype_selector = "" if not dtype else f",dtype={dtype}"  # 数据类型选择器后缀
    block_shape_selector = (  # 块形状选择器后缀
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"  # 无块形状时为空
    )
    per_channel_quant_selector = ",per_channel_quant=True" if per_channel_quant else ""  # 逐通道量化选择器后缀
    down_moe_selector = "_down" if down_moe else ""  # 下投影选择器后缀
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}{per_channel_quant_selector}{down_moe_selector}.json"  # 拼接并返回配置文件名


@functools.lru_cache  # 带缓存装饰器，避免重复加载配置文件
def get_moe_configs(  # 获取 MoE 优化的内核配置映射
    E: int,  # 专家数量
    N: int,  # 中间维度
    dtype: Optional[str],  # 数据类型字符串（可选）
    block_n: Optional[int] = 0,  # 块大小N（可选）
    block_k: Optional[int] = 0,  # 块大小K（可选）
    per_channel_quant: bool = False,  # 是否逐通道量化
    down_moe: bool = False,  # 是否为下投影 MoE
) -> Optional[Dict[int, Any]]:  # 返回批次大小到配置的映射，或None
    """
    Return optimized configurations for the fused MoE kernel.
    返回融合 MoE 内核的优化配置。

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    返回值是一个字典，将不规则网格的批次大小映射到 fused_moe 内核的配置。
    要在给定批次大小 bs 上评估内核，应选择网格中最接近的批次大小，
    并使用关联的配置来调用内核。
    """
    if get_global_server_args().enable_deterministic_inference:  # 如果启用了确定性推理
        logger.warning(  # 记录警告
            "Deterministic inference is enabled, using default MoE kernel config."  # 确定性推理已启用，使用默认 MoE 内核配置
        )
        return None  # 返回 None，使用默认配置

    # First look up if an optimized configuration is available in the configs
    # directory
    # 首先查找配置目录中是否有优化配置可用
    json_file_name = get_config_file_name(  # 生成配置文件名
        E,
        N,
        dtype,
        [block_n, block_k],
        per_channel_quant,
        down_moe=down_moe,
    )

    # We found that using the fused_moe_kernel config from Triton 3.1.0 with Triton 3.2.0 results in negative performance gains,
    # so we also include the Triton version as a key for finding the fused_moe_kernel config to achieve the best performance.
    # 我们发现将 Triton 3.1.0 的 fused_moe_kernel 配置用于 Triton 3.2.0 会导致性能下降，
    # 因此我们也将 Triton 版本作为查找 fused_moe_kernel 配置的键，以实现最佳性能。
    config_dir = os.environ.get(  # 获取配置目录，默认为当前文件所在目录
        "SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )

    triton_version = triton.__version__  # 获取当前 Triton 版本
    version_dir = f"triton_{triton_version.replace('.', '_')}"  # 将版本号中的点替换为下划线作为目录名
    config_file_path = os.path.join(  # 构建配置文件完整路径
        config_dir,  # 配置根目录
        "configs",  # configs 子目录
        version_dir,  # 版本子目录
        json_file_name,  # 配置文件名
    )
    if os.path.exists(config_file_path):  # 如果配置文件存在
        with open(config_file_path) as f:  # 打开配置文件
            # Please note that although we find the config files, performance might still be suboptimal.
            # 请注意，虽然我们找到了配置文件，性能可能仍然不是最优的。
            # This is because the tuning environment might differ from your current environment.
            # 这是因为调优环境可能与当前环境不同。
            # For example, updating the Triton version might cause all old configs to become suboptimal.
            # 例如，更新 Triton 版本可能导致所有旧配置变得次优。
            # To achieve the best performance, consider re-tuning the Triton fused MOE kernel in your environment.
            # 要实现最佳性能，请考虑在您的环境中重新调优 Triton 融合 MOE 内核。
            # For the tuning method, refer to: https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            # 调优方法请参考：https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            logger.info(f"Using MoE kernel config from {config_file_path}.")  # 记录使用的配置文件路径
            # If a configuration has been found, return it
            # 如果找到了配置，返回它
            return {int(key): val for key, val in json.load(f).items()}  # 将 JSON 键转为整数，返回配置字典

    # Discover available triton config dirs on disk and search newest-first.
    # 发现磁盘上可用的 Triton 配置目录，并按最新版本优先搜索。
    configs_root = os.path.join(config_dir, "configs")  # 配置根目录下的 configs 子目录
    available_versions = sorted(  # 对可用版本排序
        (
            d.removeprefix("triton_").replace("_", ".")  # 从目录名中提取版本号
            for d in os.listdir(configs_root)  # 遍历 configs 目录下的所有条目
            if d.startswith("triton_")  # 只选择以 "triton_" 开头的目录
        ),
        key=lambda v: tuple(int(x) for x in v.split(".")),  # 按版本号数值排序
        reverse=True,  # 降序排列（最新版本在前）
    )

    for try_triton_version in available_versions:  # 遍历可用版本（从新到旧）
        if try_triton_version == triton_version:  # 跳过当前版本（已在上面处理）
            continue
        try_config_file_path = os.path.join(  # 构建尝试版本的配置文件路径
            configs_root,  # 配置根目录
            f"triton_{try_triton_version.replace('.', '_')}",  # 版本目录
            json_file_name,  # 配置文件名
        )
        if os.path.exists(try_config_file_path):  # 如果配置文件存在
            with open(try_config_file_path) as f:  # 打开配置文件
                logger.warning(  # 记录警告
                    f"Config file not found at {config_file_path}. Fallback to triton version {try_triton_version} and use MoE kernel config from {try_config_file_path}. Performance might be sub-optimal!",  # 配置文件未找到，回退到其他 Triton 版本
                )
                # If a configuration has been found, return it
                # 如果找到了配置，返回它
                return {int(key): val for key, val in json.load(f).items()}  # 返回配置字典

    # If no optimized configuration is available, we will use the default configuration when down_moe is False
    # 如果没有优化配置可用，当 down_moe 为 False 时使用默认配置
    # When down_moe is True, we will try to use the config for down_moe=False
    # 当 down_moe 为 True 时，尝试使用 down_moe=False 的配置
    if down_moe:  # 下投影 MoE 模式
        logger.warning(  # 记录警告
            (
                "Using MoE kernel config with down_moe=False. Performance might be sub-optimal! "
                "使用 down_moe=False 的 MoE 内核配置。性能可能不是最优的！"
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
                "配置文件未找到于 %s，可通过 https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton 创建"
            ),
            config_file_path,
        )
    else:  # 非下投影 MoE 模式
        logger.warning(  # 记录警告
            (
                "Using default MoE kernel config. Performance might be sub-optimal! "
                "使用默认 MoE 内核配置。性能可能不是最优的！"
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
                "配置文件未找到于 %s，可通过 https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton 创建"
            ),
            config_file_path,
        )
    return None  # 返回 None，使用默认配置


def get_default_config(  # 获取默认的 MoE 内核配置
    M: int,  # 批次大小（令牌数）
    E: int,  # 专家数量
    N: int,  # 输出维度
    K: int,  # 输入维度
    topk: int,  # Top-K 值
    dtype: Optional[str],  # 数据类型字符串（可选）
    is_marlin: bool,  # 是否使用 Marlin 格式
    block_shape: Optional[List[int]] = None,  # 块形状（可选）
) -> Dict[str, int]:  # 返回配置字典
    if get_global_server_args().enable_deterministic_inference:  # 确定性推理模式
        config = {  # 使用固定的确定性配置
            "BLOCK_SIZE_M": 64,  # M 维度块大小
            "BLOCK_SIZE_N": 64,  # N 维度块大小
            "BLOCK_SIZE_K": 32,  # K 维度块大小
            "GROUP_SIZE_M": 8,  # M 维度分组大小
        }
        return config  # 返回确定性配置
    if dtype == "fp8_w8a8":  # FP8 量化模式
        if block_shape is None:  # 无块状量化
            config = {  # FP8 默认配置
                "BLOCK_SIZE_M": 128,  # M 维度块大小
                "BLOCK_SIZE_N": 256,  # N 维度块大小
                "BLOCK_SIZE_K": 128,  # K 维度块大小
                "GROUP_SIZE_M": 32,  # M 维度分组大小
                "num_warps": 8,  # warp 数量
                "num_stages": 2 if _is_hip else 4,  # 流水线级数（HIP 较少）
            }
            if M <= E:  # 令牌数少于等于专家数时使用更小的配置
                config = {  # 小批次 FP8 配置
                    "BLOCK_SIZE_M": 64,  # 较小的 M 块
                    "BLOCK_SIZE_N": 128,  # 较小的 N 块
                    "BLOCK_SIZE_K": 128,  # K 块不变
                    "GROUP_SIZE_M": 1,  # 分组大小为1
                    "num_warps": 4,  # 较少的 warp
                    "num_stages": 2 if _is_hip else 4,  # 流水线级数
                }
        else:  # 块状量化模式
            # Block-wise quant: BLOCK_SIZE_K must be divisible by block_shape[1]
            # 块状量化：BLOCK_SIZE_K 必须能被 block_shape[1] 整除
            config = {  # 块状量化 FP8 配置
                "BLOCK_SIZE_M": 64,  # M 维度块大小
                "BLOCK_SIZE_N": block_shape[0],  # N 维度块大小与量化块一致
                "BLOCK_SIZE_K": block_shape[1],  # K 维度块大小与量化块一致
                "GROUP_SIZE_M": 32,  # M 维度分组大小
                "num_warps": 4,  # warp 数量
                "num_stages": 2 if _is_hip else 3,  # 流水线级数
            }
    else:  # 非 FP8 模式
        config = {  # 通用默认配置
            "BLOCK_SIZE_M": 64,  # M 维度块大小
            "BLOCK_SIZE_N": 64,  # N 维度块大小
            "BLOCK_SIZE_K": 32,  # K 维度块大小
            "GROUP_SIZE_M": 8,  # M 维度分组大小
        }
        # A heuristic: fused marlin works faster with this config for small M
        # 启发式规则：融合 Marlin 在小 M 时使用此配置更快
        if M <= E or (is_marlin and M <= 32):  # 小批次或 Marlin 小令牌数
            config = {  # 小批次配置
                "BLOCK_SIZE_M": 16,  # 更小的 M 块
                "BLOCK_SIZE_N": 32,  # 更小的 N 块
                "BLOCK_SIZE_K": 64,  # 更大的 K 块
                "GROUP_SIZE_M": 1,  # 分组大小为1
            }
    return config  # 返回配置


def try_get_optimal_moe_config(  # 尝试获取最优的 MoE 内核配置
    w1_shape: Tuple[int, ...],  # w1 权重形状
    w2_shape: Tuple[int, ...],  # w2 权重形状
    top_k: int,  # Top-K 值
    dtype: Optional[str],  # 数据类型字符串（可选）
    M: int,  # 批次大小（令牌数）
    is_marlin: bool = False,  # 是否使用 Marlin 格式
    block_shape: Optional[List[int]] = None,  # 块形状（可选）
    per_channel_quant: bool = False,  # 是否逐通道量化
    return_down_config: bool = False,  # 是否返回下投影配置
):
    from sglang.srt.layers.moe.moe_runner.triton_utils import get_config  # 延迟导入获取配置函数

    down_config = None  # 下投影配置，初始为None
    max_block_m = None  # 最大块大小M，初始为None
    override_config = get_config()  # 获取全局覆盖配置
    if override_config:  # 如果存在覆盖配置
        config = override_config  # 使用覆盖配置
    else:  # 没有覆盖配置
        # First try to load optimal config from the file
        # 首先尝试从文件加载最优配置
        E, _, N = w2_shape  # 从 w2 形状中提取专家数E和输出维度N
        block_n = block_shape[0] if block_shape else 0  # 获取块大小N
        block_k = block_shape[1] if block_shape else 0  # 获取块大小K
        configs = get_moe_configs(  # 尝试加载优化配置
            E,  # 专家数
            N,  # 输出维度
            dtype,  # 数据类型
            block_n,  # 块大小N
            block_k,  # 块大小K
            per_channel_quant=per_channel_quant,  # 逐通道量化
            down_moe=False,  # 非下投影
        )

        if configs:  # 如果找到了优化配置
            # If an optimal configuration map has been found, look up the
            # optimal config
            # 如果找到了优化配置映射，查找最优配置
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]  # 选择最接近当前批次大小的配置
        else:  # 没有优化配置
            # Else use the default config
            # 否则使用默认配置
            config = get_default_config(  # 获取默认配置
                M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape  # 传入各参数
            )
        if return_down_config:  # 如果需要返回下投影配置
            down_configs = get_moe_configs(  # 加载下投影的优化配置
                E,  # 专家数
                N,  # 输出维度
                dtype,  # 数据类型
                block_n,  # 块大小N
                block_k,  # 块大小K
                per_channel_quant=per_channel_quant,  # 逐通道量化
                down_moe=True,  # 下投影模式
            )
            if down_configs:  # 如果找到了下投影优化配置
                down_config = down_configs[  # 选择最接近当前批次大小的配置
                    min(down_configs.keys(), key=lambda x: abs(x - M))
                ]
                down_config = dict(**down_config)  # 复制配置字典（避免修改原始数据）
                max_block_m = max(  # 计算所有配置中最大的 BLOCK_SIZE_M
                    [cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()]
                )
    if return_down_config:  # 如果需要返回下投影配置
        assert (  # 断言上投影和下投影的 BLOCK_SIZE_M 一致
            down_config is None or config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]
        )
        return config, (down_config, max_block_m)  # 返回配置元组
    return config  # 仅返回上投影配置


def get_config_dtype_str(  # 根据量化标志获取配置用的数据类型字符串
    dtype: torch.dtype,  # PyTorch 数据类型
    use_int8_w8a16: Optional[bool] = False,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: Optional[bool] = False,  # 是否使用 INT4 W4A16 量化
    use_fp8_w8a8: Optional[bool] = False,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: Optional[bool] = False,  # 是否使用 INT8 W8A8 量化
):
    if use_fp8_w8a8:  # FP8 量化
        return "fp8_w8a8"  # 返回 FP8 标识
    elif use_int8_w8a8:  # INT8 W8A8 量化
        return "int8_w8a8"  # 返回 INT8 W8A8 标识
    elif use_int4_w4a16:  # INT4 W4A16 量化
        return "int4_w4a16"  # 返回 INT4 W4A16 标识
    elif use_int8_w8a16:  # INT8 W8A16 量化
        return "int8_w8a16"  # 返回 INT8 W8A16 标识
    elif dtype == torch.float:  # float32 类型
        # avoiding cases where kernel fails when float32 MoE
        # 避免 float32 MoE 时内核失败的情况
        # use fp16/bfloat16 configs
        # 使用 fp16/bfloat16 的配置
        return "float32"  # 返回 float32 标识
    return None  # 其他类型返回 None
