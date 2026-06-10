# TorchAO 量化工具模块
# 本文件提供了基于 TorchAO 的模型量化工具函数，
# 支持 int8/int4/fp8 等多种量化配置，并可按层名过滤进行选择性量化。
"""
Common utilities for torchao.
"""  # TorchAO 通用工具

import logging  # 导入日志模块
from typing import Callable, Optional  # 导入类型提示

import torch  # 导入 PyTorch 库

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


def proj_filter(  # 投影层过滤函数，用于筛选名称中包含 "proj" 的层进行量化
    module: torch.nn.Module,  # PyTorch 模块
    fqn: str,  # 模块的完全限定名称
):
    """Filter function for quantizing projection layers."""  # 用于量化投影层的过滤函数
    return "proj" in fqn  # 检查模块名称中是否包含 "proj"


# TODO: implement a more general filter function  # TODO: 实现更通用的过滤函数
def proj_filter_conv3d(  # 投影层过滤函数（排除 Conv3d），用于筛选名称中包含 "proj" 但非 Conv3d 的层
    module: torch.nn.Module,  # PyTorch 模块
    fqn: str,  # 模块的完全限定名称
):
    if isinstance(module, torch.nn.Conv3d):  # 如果模块是 Conv3d 类型
        logger.warning(f"Quantize: skipping {fqn} because it's a Conv3d")  # 记录警告：跳过 Conv3d 层
        return False  # 返回 False，不量化该层
    return "proj" in fqn  # 否则检查模块名称中是否包含 "proj"


def apply_torchao_config_to_model(  # 将 TorchAO 量化配置应用到模型
    model: torch.nn.Module,  # 待量化的模型
    torchao_config: str,  # TorchAO 量化配置字符串
    filter_fn: Optional[Callable] = proj_filter,  # 过滤函数，默认为 proj_filter
):
    """Quantize a modelwith torchao quantization specified by torchao_config  # 使用 torchao_config 指定的量化方式量化模型

    Args:  # 参数说明
       `model`: a model to be quantized based on torchao_config  # model: 基于 torchao_config 待量化的模型
       `torchao_config` (str): type of quantization and their arguments we want to use to  # torchao_config (str): 要使用的量化类型及其参数
        quantize the model, e.g. int4wo-128 means int4 weight only quantization with group_size  # 例如 int4wo-128 表示分组大小为 128 的 int4 仅权重量化
        128  # 128
    """
    if torchao_config == "" or torchao_config is None:  # 如果配置为空或 None
        return model  # 直接返回未量化的模型

    # Lazy import to suppress some warnings  # 延迟导入以抑制部分警告
    from torchao.quantization import (  # 从 torchao.quantization 导入量化函数
        float8_dynamic_activation_float8_weight,  # FP8 动态激活 FP8 权重量化
        float8_weight_only,  # FP8 仅权重量化
        int4_weight_only,  # INT4 仅权重量化
        int8_dynamic_activation_int8_weight,  # INT8 动态激活 INT8 权重量化
        int8_weight_only,  # INT8 仅权重量化
        quantize_,  # 量化主函数
    )
    from torchao.quantization.observer import PerRow, PerTensor  # 导入逐行和逐张量观察器

    if "int8wo" in torchao_config:  # 如果配置包含 "int8wo"（INT8 仅权重量化）
        quantize_(model, int8_weight_only(), filter_fn=proj_filter_conv3d)  # 应用 INT8 仅权重量化
    elif "int8dq" in torchao_config:  # 如果配置包含 "int8dq"（INT8 动态量化）
        quantize_(model, int8_dynamic_activation_int8_weight(), filter_fn=filter_fn)  # 应用 INT8 动态量化
    elif "int4wo" in torchao_config:  # 如果配置包含 "int4wo"（INT4 仅权重量化）
        group_size = int(torchao_config.split("-")[-1])  # 从配置字符串中解析分组大小
        assert group_size in [  # 断言分组大小必须为合法值
            32,  # 分组大小 32
            64,  # 分组大小 64
            128,  # 分组大小 128
            256,  # 分组大小 256
        ], f"int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}"  # 分组大小不合法时报错
        quantize_(model, int4_weight_only(group_size=group_size), filter_fn=filter_fn)  # 应用 INT4 仅权重量化
    elif "fp8wo" in torchao_config:  # 如果配置包含 "fp8wo"（FP8 仅权重量化）
        # this requires newer hardware  # 需要较新的硬件
        # [rank0]: AssertionError: fp8e4nv data type is not supported on CUDA arch < 89  # fp8e4nv 数据类型不支持 CUDA 架构 < 89
        quantize_(model, float8_weight_only(), filter_fn=proj_filter_conv3d)  # 应用 FP8 仅权重量化
    elif "fp8dq" in torchao_config:  # 如果配置包含 "fp8dq"（FP8 动态量化）
        granularity = torchao_config.split("-")[-1]  # 从配置字符串中解析粒度
        GRANULARITY_MAP = {  # 粒度映射表
            "per_row": PerRow(),  # 逐行粒度
            "per_tensor": PerTensor(),  # 逐张量粒度
        }
        assert (  # 断言粒度必须合法
            granularity in GRANULARITY_MAP  # 检查粒度是否在映射表中
        ), f"Supported granularity are: {GRANULARITY_MAP.keys()}, got {granularity}"  # 粒度不合法时报错
        quantize_(  # 应用 FP8 动态量化
            model,  # 待量化的模型
            float8_dynamic_activation_float8_weight(  # FP8 动态激活 FP8 权重量化
                granularity=GRANULARITY_MAP[granularity]  # 指定量化粒度
            ),
            filter_fn=proj_filter_conv3d,  # 使用排除 Conv3d 的过滤函数
        )
    else:  # 其他未识别的配置
        raise ValueError(f"Unexpected config: {torchao_config}")  # 抛出值错误异常

    return model  # 返回量化后的模型
