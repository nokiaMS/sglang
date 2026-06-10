# 压缩张量量化工具函数
# 本文件提供了压缩张量量化方案的工具函数，包括激活量化格式检测、
# 层忽略判断、目标匹配（支持精确匹配和正则表达式）以及融合层匹配等功能。
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0  # Apache-2.0许可证声明
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # vLLM项目版权声明

import re  # 导入正则表达式模块
from types import MappingProxyType  # 导入不可变映射代理类型
from typing import Iterable, List, Mapping, Optional  # 导入类型提示

from compressed_tensors import CompressionFormat  # 导入压缩格式枚举
from torch.nn import Module  # 导入PyTorch模块基类


def is_activation_quantization_format(format: str) -> bool:  # 判断是否为激活量化格式
    """判断给定的压缩格式是否支持激活量化。"""
    _ACTIVATION_QUANTIZATION_FORMATS = [  # 支持激活量化的格式列表
        CompressionFormat.naive_quantized.value,  # 朴素量化格式
        CompressionFormat.int_quantized.value,  # 整数量化格式
        CompressionFormat.float_quantized.value,  # 浮点量化格式
        CompressionFormat.nvfp4_pack_quantized.value,  # NVFP4打包量化格式
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS  # 检查格式是否在列表中


def should_ignore_layer(  # 判断是否应忽略该层的量化
    layer_name: Optional[str],  # 层名称
    ignore: Iterable[str] = tuple(),  # 要忽略的层名列表
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),  # 融合层到子层的映射
) -> bool:  # 返回是否应忽略
    """判断给定层是否应被忽略（不进行量化），处理融合层和普通层两种情况。"""
    if layer_name is None:  # 如果层名为空
        return False  # 不忽略

    # layer_name = model.layers.0.self_attn.qkv_proj  # 层名示例：model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj  # 投影名示例：qkv_proj
    proj_name = layer_name.split(".")[-1]  # 从层名中提取最后一部分作为投影名

    # Fused layers like gate_up_proj or qkv_proj will not be fused  # 融合层如gate_up_proj或qkv_proj在safetensors检查点中
    # in the safetensors checkpoint. So, we convert the name  # 不会被融合。因此，我们将名称
    # from the fused version to unfused + check to make sure that  # 从融合版本转换为非融合版本，并检查确保
    # each shard of the fused layer has the same scheme.  # 融合层的每个分片使用相同的量化方案
    if proj_name in fused_mapping and layer_name not in ignore:  # 如果投影名在融合映射中且层名不在忽略列表中
        shard_proj_names = fused_mapping[proj_name]  # 获取融合层对应的分片投影名列表

        # Convert fused_name --> [shard_names]  # 将融合名转换为分片名列表
        shard_names = [  # 构建分片名称列表
            layer_name.replace(proj_name, shard_proj_name)  # 替换投影名为分片投影名
            for shard_proj_name in shard_proj_names  # 遍历所有分片投影名
        ]

        # Layer should be ignored if shards are ignored.  # 如果分片被忽略，则层也应被忽略
        should_ignore_layer = None  # 初始化忽略标志为None
        for shard_name in shard_names:  # 遍历所有分片名
            should_ignore_shard = check_equal_or_regex_match(  # 检查分片是否匹配忽略规则
                layer_name=shard_name, targets=ignore  # 传入分片名和忽略列表
            )

            # If shard_idx=0, set layer ignore to match shard.  # 如果是第一个分片，设置层的忽略标志匹配该分片
            if should_ignore_layer is None:  # 如果忽略标志尚未设置
                should_ignore_layer = should_ignore_shard  # 设置为当前分片的忽略状态

            # If shard_idx=1+ confirm scheme matches prior shards.  # 如果是第二个及以后的分片，确认方案与之前的分片匹配
            elif should_ignore_shard != should_ignore_layer:  # 如果当前分片与之前分片的忽略状态不同
                raise ValueError(  # 抛出值错误
                    f"Found a different quantization schemes for "  # 发现不同的量化方案
                    f"{shard_proj_names} in {layer_name}. vLLM "  # 在该层中。vLLM
                    "requires all to use the same scheme."  # 要求所有分片使用相同的方案
                )

    # Unfused layers like down_proj and o_proj will match  # 非融合层如down_proj和o_proj将直接匹配
    # the safetensors checkpoint already.  # safetensors检查点中的名称
    else:  # 否则（非融合层或在忽略列表中）
        should_ignore_layer = check_equal_or_regex_match(  # 检查层名是否匹配忽略规则
            layer_name=layer_name, targets=ignore  # 传入层名和忽略列表
        )

    assert should_ignore_layer is not None  # 断言忽略标志已设置
    return should_ignore_layer  # 返回是否忽略


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:  # 检查精确匹配或正则匹配
    """
    Checks whether a layer_name is exactly equal or a regex match for  # 检查层名是否精确匹配或正则匹配
    if target starts with 're:' to any target in list.  # 如果目标以're:'开头则进行正则匹配
    """
    for target in targets:  # 遍历所有目标
        if _is_equal_or_regex_match(layer_name, target, check_contains=True):  # 检查是否匹配（包含子串检查）
            return True  # 匹配成功
    return False  # 无匹配


def find_matched_target(  # 查找匹配的目标
    layer_name: Optional[str],  # 层名称
    module: Module,  # PyTorch模块
    targets: Iterable[str],  # 目标列表
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),  # 融合层映射
) -> str:  # 返回匹配的目标字符串
    """
    Helper function to look up which "target" in the compressed-tensors  # 辅助函数，查找压缩张量配置中
    config that a layer corresponds to.  # 某层对应的"目标"

    Recall that a compressed-tensors configs has a concept of  # 回顾：压缩张量配置有配置组的概念，
    config_groups, where each layer can be quantized with with a different  # 每层可以用不同的方案进行量化
    scheme.  # 量化方案

    targets in each config_group will be a list of either layer names  # 每个配置组中的目标是层名列表
    (or regexes corresponding to layer names) or names of torch Modules.  # （或对应层名的正则表达式）或torch模块名

    First, we try to match the layer_name with a target  # 首先，尝试将层名与目标匹配
    Second, we try to match the module's name with a target  # 其次，尝试将模块名与目标匹配
    Third, we try to map the layer_name to a list of fused module names.  # 第三，尝试将层名映射到融合模块名列表
        *All* component module names must match in order for a match to be  # 所有组件模块名都必须匹配才算匹配成功
        successful. A successful match returns the first component target  # 成功匹配返回第一个组件目标

    :param layer_name: layer name  # 参数：层名
    :param module: torch.nn.Module  # 参数：torch.nn.Module模块
    :param targets: list of targets to match the layer against  # 参数：要匹配的目标列表
    :param fused_mapping: map from fused layer names to its components  # 参数：从融合层名到其组件的映射
    :param fused_strategy: either "all" or "any". If using "all", fused  # 参数：融合策略，"all"或"any"。使用"all"时，
        layers match if "all" of its components match  # 融合层匹配当其所有组件都匹配
    """

    if layer_name is None:  # 如果层名为空
        layer_name = ""  # 设为空字符串

    matched_target = (  # 尝试三种匹配方式
        _find_first_match(layer_name, targets)  # 第一种：用层名精确匹配
        or _find_first_match(module.__class__.__name__, targets, True)  # 第二种：用模块类名匹配（包含检查）
        or _match_fused_layer(layer_name, targets, fused_mapping)  # 第三种：融合层匹配
    )

    if matched_target is None:  # 如果没有匹配到任何目标
        raise ValueError(  # 抛出值错误
            f"Unable to find matching target for {layer_name} in the "  # 无法在压缩张量配置中找到匹配目标
            "compressed-tensors config."  # 压缩张量配置
        )

    return matched_target  # 返回匹配的目标


def _find_first_match(  # 查找第一个匹配项（内部函数）
    value: str, targets: Iterable[str], check_contains: bool = False  # 值、目标列表、是否检查包含
) -> Optional[str]:  # 返回匹配的目标或None
    """
    Returns first element of target that matches value either  # 返回目标列表中第一个匹配值的元素，
    exactly or as a regex after 're:'. If check_contains is set to True,  # 可以精确匹配或're:'后的正则匹配。如果check_contains为True，
    additionally checks if the target string is contained within the value.  # 额外检查目标字符串是否包含在值中

    :param value: string to compare the list of targets against  # 参数：要与目标列表比较的字符串
    :param targets: list of targets to match the layer against  # 参数：要匹配的目标列表
    :param check_contains: whether or not to do a substring match  # 参数：是否进行子串匹配
    """

    for target in targets:  # 遍历所有目标
        if _is_equal_or_regex_match(value, target, check_contains=check_contains):  # 检查是否匹配
            return target  # 返回匹配的目标
    return None  # 无匹配返回None


def _is_equal_or_regex_match(  # 判断是否精确匹配或正则匹配（内部函数）
    value: str, target: str, check_contains: bool = False  # 值、目标、是否检查包含
) -> bool:  # 返回是否匹配
    """
    Checks whether a value is exactly equal or a regex match for target  # 检查值是否精确匹配或正则匹配目标
    if target starts with 're:'. If check_contains is set to True,  # 如果目标以're:'开头则正则匹配。如果check_contains为True，
    additionally checks if the target string is contained within the value.  # 额外检查目标字符串是否包含在值中
    """

    if target.startswith("re:"):  # 如果目标以"re:"开头（正则表达式）
        pattern = target[3:]  # 提取正则表达式模式（去掉"re:"前缀）
        if re.match(pattern, value):  # 尝试正则匹配
            return True  # 匹配成功
    elif check_contains:  # 如果启用包含检查
        if target.lower() in value.lower():  # 检查目标（小写）是否包含在值（小写）中
            return True  # 包含匹配成功
    elif target == value:  # 精确匹配
        return True  # 精确匹配成功
    return False  # 无匹配返回False


def _match_fused_layer(  # 匹配融合层（内部函数）
    layer_name: str,  # 层名称
    target_layers: Iterable[str],  # 目标层列表
    fused_mapping: Mapping[str, List[str]],  # 融合层映射
) -> Optional[str]:  # 返回匹配的目标或None
    """
    Match a fused layer name to its corresponding individual layer in  # 将融合层名匹配到目标层中对应的单独层
    target_layers. Returns first value in fused_mapping which matches targets  # 返回fused_mapping中第一个匹配目标的值

    Implements an "all" matching strategy where a fused layer matches iff  # 实现"全部"匹配策略：融合层匹配当且仅当
    "all" of its components match  # 其所有组件都匹配

    :param layer_name: layer name  # 参数：层名
    :param target_layers: list of targets to match the layer against  # 参数：要匹配的目标层列表
    :param fused_mapping: map from fused layer names to its components  # 参数：从融合层名到其组件的映射

    Examples:  # 示例：
        layer_name = "model.layers.0.self_attn.qkv_proj"  # 层名示例
        target_layers = ["model.layers.0.self_attn.q_proj",  # 目标层示例
                        "model.layers.0.self_attn.k_proj",  # 目标层示例
                        "model.layers.0.self_attn.v_proj"]  # 目标层示例
    """
    # find layer_name in mapping  # 在映射中查找层名
    fused = next((key for key in fused_mapping if layer_name.endswith(key)), None)  # 查找层名结尾匹配的融合层键
    if fused is None:  # 如果没有找到匹配的融合层
        return None  # 返回None

    # expand path of unfused components  # 展开非融合组件的路径
    unfused_paths = [  # 构建非融合组件路径列表
        layer_name.replace(fused, unfused) for unfused in fused_mapping[fused]  # 替换融合名为非融合名
    ]

    # for each unfused component, find a match in targets  # 对每个非融合组件，在目标中查找匹配
    unfused_matches: List[Optional[str]] = []  # 非融合组件匹配结果列表
    for unfused in unfused_paths:  # 遍历所有非融合路径
        for target in target_layers:  # 遍历所有目标层
            if _is_equal_or_regex_match(unfused, target):  # 检查是否匹配
                unfused_matches.append(target)  # 添加匹配结果
                break  # 找到匹配后跳出内层循环
        else:  # 如果内层循环正常结束（没有找到匹配）
            unfused_matches.append(None)  # 添加None表示未匹配

    return unfused_matches[0] if all(unfused_matches) else None  # 如果所有组件都匹配则返回第一个匹配，否则返回None
