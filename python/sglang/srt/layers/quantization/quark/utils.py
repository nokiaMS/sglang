# Quark量化工具模块，提供深度比较、层忽略判断、正则匹配、MXFP4量化/反量化及权重后处理等辅助函数
# SPDX-License-Identifier: Apache-2.0

import re  # 导入正则表达式模块 # 正则表达式模块
from collections.abc import Iterable, Mapping  # 导入抽象集合类型 # 导入可迭代和映射类型
from types import MappingProxyType  # 导入不可变映射代理类型 # 导入映射代理类型
from typing import Any, Optional  # 导入类型注解 # 导入类型提示

import torch  # 导入PyTorch # 导入PyTorch框架

try:  # 尝试导入aiter的动态MXFP4量化 # 尝试导入aiter动态MXFP4量化
    from aiter.ops.triton.quant import dynamic_mxfp4_quant  # 从aiter导入动态MXFP4量化函数 # 从aiter导入动态MXFP4量化
except ImportError as err:  # 导入失败时定义替代函数 # 导入失败时定义替代函数

    def raise_aiter_import_error(*args, **kwargs):  # 当aiter未安装时抛出导入错误 # 抛出aiter导入错误的替代函数
        raise ImportError(
            "Failed to import aiter. " "Make sure AITER is installed and accessible."
        )

    dynamic_mxfp4_quant = raise_aiter_import_error  # 将替代函数赋值给动态量化变量 # 将替代函数赋给变量
from torch import nn  # 从torch导入神经网络模块 # 导入PyTorch神经网络模块


def deep_compare(dict1: Any, dict2: Any) -> bool:  # 深度比较两个对象是否相等，支持字典和列表 # 深度比较两个对象是否相等
    if type(dict1) is not type(dict2):  # 类型不同则不相等 # 类型不同则返回False
        return False
    if isinstance(dict1, dict):  # 如果是字典类型 # 如果是字典
        if dict1.keys() != dict2.keys():  # 键不同则不相等 # 键不同返回False
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)  # 递归比较每个键对应的值 # 递归比较每个值
    elif isinstance(dict1, list):  # 如果是列表类型 # 如果是列表
        return set(dict1) == set(dict2)  # 转为集合比较（忽略顺序） # 转为集合比较
    else:  # 其他类型直接比较 # 其他类型直接比较
        return dict1 == dict2


def should_ignore_layer(  # 判断某层是否应被忽略，支持融合层的分片检查 # 判断某层是否应被忽略
    layer_name: Optional[str],
    ignore: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    if layer_name is None:  # 层名为None则不忽略 # 层名为None不忽略
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj  # 层名示例 # 层名示例
    # proj_name = qkv_proj  # 投影名示例 # 投影名示例
    proj_name = layer_name.split(".")[-1]  # 提取最后一部分作为投影名 # 提取投影名

    # Fused layers like gate_up_proj or qkv_proj will not be fused  # 融合层如gate_up_proj或qkv_proj在safetensors检查点中不会被融合
    # in the safetensors checkpoint. So, we convert the name  # 因此我们转换名称
    # from the fused version to unfused + check to make sure that  # 从融合版本转为非融合版本，并检查确保
    # each shard of the fused layer has the same scheme.  # 融合层的每个分片使用相同的量化方案。
    if proj_name in fused_mapping:  # 如果是融合投影名 # 如果投影名在融合映射中
        shard_proj_names = fused_mapping[proj_name]  # 获取分片投影名列表 # 获取分片投影名列表

        # Convert fused_name --> [shard_names]  # 将融合名转换为分片名列表 # 将融合名转换为分片名列表
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)  # 替换投影名为分片投影名 # 替换投影名
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.  # 如果分片被忽略则该层也应被忽略 # 层应在分片被忽略时被忽略
        should_ignore_layer = None  # 初始化忽略标志 # 初始化忽略标志
        for shard_name in shard_names:  # 遍历每个分片名 # 遍历分片名
            should_ignore_shard = check_equal_or_regex_match(  # 检查分片是否匹配忽略目标 # 检查分片是否匹配
                layer_name=shard_name, targets=ignore
            )

            # If shard_idx=0, set layer ignore to match shard.  # 如果是第一个分片，设置层忽略标志与分片一致 # 第一个分片设置忽略标志
            if should_ignore_layer is None:  # 第一个分片 # 第一个分片时
                should_ignore_layer = should_ignore_shard  # 设置忽略标志 # 设置忽略标志

            # If shard_idx=1+ confirm scheme matches prior shards.  # 如果是后续分片，确认量化方案与之前的分片一致 # 后续分片确认方案一致
            elif should_ignore_shard != should_ignore_layer:  # 量化方案不一致 # 方案不一致
                raise ValueError(
                    f"Found a different quantization schemes for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )

    # Unfused layers like down_proj and o_proj will match  # 非融合层如down_proj和o_proj直接匹配safetensors检查点
    # the safetensors checkpoint already.  # 已经与safetensors检查点匹配。
    else:  # 非融合层 # 非融合层
        should_ignore_layer = check_equal_or_regex_match(  # 直接检查层名是否匹配 # 直接检查层名
            layer_name=layer_name, targets=ignore
        )

    assert should_ignore_layer is not None  # 确保忽略标志已设置 # 确保忽略标志已设置

    return should_ignore_layer  # 返回是否应忽略 # 返回是否忽略


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:  # 检查层名是否等于或正则匹配任一目标 # 检查层名是否匹配目标
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    检查layer_name是否精确等于或正则匹配（当目标以're:'开头时）列表中的任一目标。
    """
    for target in targets:  # 遍历每个目标 # 遍历目标
        if _is_equal_or_regex_match(layer_name, target):  # 检查是否匹配 # 检查是否匹配
            return True
    return False


def _is_equal_or_regex_match(  # 内部函数：检查值是否等于或正则匹配目标 # 检查值是否等于或正则匹配目标
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    检查值是否精确等于目标，或当目标以're:'开头时进行正则匹配。
    如果check_contains设为True，额外检查目标字符串是否包含在值中。
    """

    if target.startswith("re:"):  # 如果目标以're:'开头，进行正则匹配 # 正则匹配
        pattern = target[3:]  # 提取正则模式 # 提取正则模式
        if re.match(pattern, value):  # 正则匹配成功 # 匹配成功
            return True
    elif check_contains:  # 如果启用包含检查 # 包含检查
        if target.lower() in value.lower():  # 忽略大小写检查包含关系 # 忽略大小写检查包含
            return True
    elif target == value:  # 精确相等 # 精确相等
        return True
    return False  # 均不匹配 # 不匹配


# utility for tensor dims > 2 cases  # 张量维度大于2的情况的工具函数
def b_dynamic_mxfp4_quant(x):  # 批量动态MXFP4量化，处理维度大于2的张量 # 批量动态MXFP4量化
    h, b, d = x.shape  # 解包三维形状 # 解包形状
    x, x_scales = dynamic_mxfp4_quant(x.reshape(-1, d))  # 展平后进行动态MXFP4量化 # 展平后量化
    return x.view(h, b, d // 2), x_scales.view(h, b, d // 32)  # 恢复原始维度并返回量化结果和缩放因子 # 恢复维度返回结果


def mxfp4_to_f32(x, is_3d):  # 将MXFP4格式转换为float32 # 将MXFP4转为float32
    # 2 because we pack fp4 in uint8.  # 2是因为我们将fp4打包在uint8中。 # 将fp4打包到uint8中因此需要重复
    x = x.repeat_interleave(2, dim=-1)  # 沿最后一维重复交错 # 沿最后一维重复
    if is_3d:  # 如果是3D张量 # 3D张量情况
        x[..., ::2] = x[..., ::2] & 0xF  # 取低4位 # 取低4位
        x[..., 1::2] = x[..., 1::2] >> 4  # 右移4位取高4位 # 右移4位取高4位
    else:  # 2D张量情况 # 2D张量情况
        x[:, ::2] = x[:, ::2] & 0xF  # 取低4位 # 取低4位
        x[:, 1::2] = x[:, 1::2] >> 4  # 右移4位取高4位 # 右移4位取高4位

    mxfp4_list = [  # MXFP4格式的16个可能值 # MXFP4的16个可能值
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")  # 创建查找表 # 创建查找表张量
    return mxfp4_in_f32[x.long()]  # 使用索引查找转换为float32 # 通过索引查找转换


def e8m0_to_f32(x):  # 将e8m0格式转换为float32 # 将e8m0格式转为float32
    # Convert the input tensor `x` (assumed to be in e8m0 format) to float32.  # 将输入张量x（假设为e8m0格式）转换为float32。
    # e8m0 is a custom 8-bit floating point format with 8 bits for exponent, 0 for mantissa.  # e8m0是一种自定义8位浮点格式，8位指数，0位尾数。
    # This means the value is essentially 2^(exponent - 127), similar to how IEEE-754 stores floats.  # 这意味着值本质上是2^(指数-127)，类似于IEEE-754存储浮点数的方式。

    # Convert x to float32 for computation, and compute the power of 2 by subtracting the bias (127).  # 将x转为float32进行计算，通过减去偏置(127)计算2的幂次。
    x_f32 = 2 ** ((x.to(torch.float32)) - 127)  # 计算2^(exponent-127) # 计算e8m0对应的float32值

    # If the exponent value was 255 (i.e., 2^(128)), this is a special case usually used to represent NaN or Inf.  # 如果指数值为255（即2^128），这是通常用于表示NaN或Inf的特殊情况。
    # Since this custom format has no mantissa, treat 2^128 as NaN.  # 由于此自定义格式没有尾数，将2^128视为NaN。
    x_f32[x_f32 == 128] = float("nan")  # 将2^128替换为NaN # 将2^128替换为NaN
    return x_f32  # 返回转换结果 # 返回float32结果


def quark_post_load_weights(self_attn: nn.Module, w: torch.Tensor, quant_format: str):  # Quark权重后处理：分离kc和vc权重并量化 # Quark权重加载后处理
    if "mxfp4" in quant_format:  # 如果量化格式包含mxfp4 # 如果是MXFP4格式
        # when dtype is bf16, the processing flow is to dynamic quantize bf16 tensor to uint8 tensor  # 当数据类型为bf16时，处理流程是动态量化bf16张量为uint8张量
        # do w_kc (bf16) first to get the w_kc(uint8) w_s_kc(uint8)  # 先处理w_kc(bf16)得到w_kc(uint8)和w_s_kc(uint8)
        # and w_vc repeating the same procedure of w_kc to get  w_vc(uint8) w_s_vc(uint8)  # 然后对w_vc重复相同流程得到w_vc(uint8)和w_s_vc(uint8)
        if w.dtype == torch.bfloat16:  # bf16权重的动态量化处理 # bf16数据类型
            w_kc, w_vc = w.unflatten(  # 将权重展开并分离为kc和vc # 展开并分离kc和vc
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)  # 按维度分离 # 按维度切分
            w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))  # 对kc权重转置后量化 # 对kc转置后量化
            w_kc = w_kc.transpose(-2, -1)  # 转置回原形状 # 转置回原形状
            w_s_kc = w_s_kc.transpose(-2, -1)  # 转置回原形状 # 转置回原形状
            w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)  # 对vc权重量化 # 对vc量化
            w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)  # 调整kc缩放因子内存布局 # 调整kc缩放因子布局
            w_s_vc = w_s_vc.contiguous().transpose(1, 2)  # 调整vc缩放因子内存布局 # 调整vc缩放因子布局
        elif w.dtype == torch.uint8:  # static quant for mxfp4  # uint8表示已静态量化的mxfp4格式  # 静态量化的MXFP4
            # when dtype is uint8, it means the w has been quantized to mxfp4 format  # 当数据类型为uint8时，表示w已被量化为mxfp4格式
            # but we must separate it to w_kc and w_vc.  # 但我们必须将其分离为w_kc和w_vc。
            # The quantized tensor size is only half of original tensor size  # 量化后的张量大小只有原始张量大小的一半
            # and the scaling factor is 1/32, the transpose behavior will be not correct  # 且缩放因子为1/32，转置行为将不正确
            # need to upcast it to fp32 to separate w to w_kc and w_vc  # 需要上溯到fp32以将w分离为w_kc和w_vc
            # to ensure the following transpose behavior is correct  # 以确保后续转置行为正确
            # and then do mxfp4 quant again  # 然后再次进行mxfp4量化
            w = mxfp4_to_f32(w, True).to(torch.bfloat16)  # 反量化为float32再转为bf16 # 反量化后转为bf16
            w_scales = self_attn.kv_b_proj.weight_scale.repeat_interleave(32, dim=-1)  # 扩展缩放因子 # 扩展缩放因子
            w_scales = e8m0_to_f32(w_scales).to(torch.bfloat16)  # 将e8m0缩放因子转为bf16 # 转换缩放因子格式
            w = w * w_scales  # 恢复原始权重值 # 恢复原始权重
            w_kc, w_vc = w.unflatten(  # 展开并分离为kc和vc # 展开并分离kc和vc
                0, (-1, (self_attn.qk_nope_head_dim + self_attn.v_head_dim))
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)  # 按维度分离 # 按维度切分
            w_kc, w_s_kc = b_dynamic_mxfp4_quant(w_kc.transpose(-2, -1))  # 对kc权重转置后量化 # 对kc转置后量化
            w_kc = w_kc.transpose(-2, -1)  # 转置回原形状 # 转置回原形状
            w_s_kc = w_s_kc.transpose(-2, -1)  # 转置回原形状 # 转置回原形状
            w_vc, w_s_vc = b_dynamic_mxfp4_quant(w_vc)  # 对vc权重量化 # 对vc量化
            w_s_kc = w_s_kc.transpose(1, 2).contiguous().transpose(1, 2)  # 调整kc缩放因子内存布局 # 调整kc缩放因子布局
            w_s_vc = w_s_vc.contiguous().transpose(1, 2)  # 调整vc缩放因子内存布局 # 调整vc缩放因子布局

        return w_kc, w_s_kc, w_vc, w_s_vc  # 返回kc权重、kc缩放、vc权重、vc缩放 # 返回量化和缩放结果
