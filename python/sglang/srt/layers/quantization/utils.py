# 量化工具函数模块
# 提供量化相关的辅助函数，包括层跳过判断、权重打包/解包、GPTQ量化、
# FP4/NVFP4缩放因子重排、TRT-LLM FP4 MoE静态权重准备等功能
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py

from __future__ import annotations  # 启用延迟类型注解求值

import re  # 导入正则表达式模块
from copy import deepcopy  # 导入深拷贝函数
from types import MappingProxyType  # 导入映射代理类型(只读字典)
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union  # 导入类型提示

import numpy  # 导入NumPy数值计算库
import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant  # 导入FP8缩放量化函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类


def get_scalar_types():  # 获取标量类型枚举
    """
    Returns:
        tuple: (ScalarType, scalar_types)
    # 返回:
    #     元组: (标量类型类, 标量类型枚举)
    """
    try:  # 尝试从sgl_kernel导入
        from sgl_kernel.scalar_type import ScalarType, scalar_types  # 导入标量类型和枚举

        return ScalarType, scalar_types  # 返回标量类型类和枚举
    except ImportError:  # 如果导入失败

        class MockScalarType:  # 模拟标量类型类
            pass  # 空实现

        class MockScalarTypes:  # 模拟标量类型枚举
            uint4b8 = "uint4b8"  # 4位无符号整数(偏移8)
            uint8b128 = "uint8b128"  # 8位无符号整数(偏移128)

            def __getattr__(self, name):  # 动态获取属性
                return f"mock_{name}"  # 返回模拟名称

        return MockScalarType, MockScalarTypes()  # 返回模拟类


ScalarType, scalar_types = get_scalar_types()  # 初始化标量类型和枚举


def _module_path_match(ignored: str, prefix: str) -> bool:  # 模块路径匹配函数
    # Match on dotted module-path boundaries so that `mlp.gate` does NOT
    # match `mlp.gate_up_proj`. Needed for quant configs (e.g. Qwen3.6-FP8)
    # whose `modules_to_not_convert` lists MoE-template names like `mlp.gate`
    # that collide with fused dense MLP names by plain substring.
    # 在点分隔的模块路径边界上匹配，使得`mlp.gate`不会
    # 匹配`mlp.gate_up_proj`。这是量化配置(例如Qwen3.6-FP8)所需的，
    # 其`modules_to_not_convert`列出了MoE模板名称如`mlp.gate`，
    # 这些名称通过普通子串匹配会与融合密集MLP名称冲突。
    ignored = ignored.rstrip(".")  # 去除忽略模式末尾的点号
    prefix = prefix.rstrip(".")  # 去除前缀末尾的点号
    if ignored == prefix:  # 如果完全相等
        return True  # 匹配成功
    if prefix.startswith(ignored + "."):  # 如果前缀以忽略模式+点号开头
        return True  # 匹配成功
    return ("." + ignored + ".") in ("." + prefix + ".")  # 检查忽略模式是否作为中间路径段出现


# Known fused-linear -> shard names. Used as a fallback when the quant
# config doesn't ship packed_modules_mapping (typical for HF FP8 configs).
# 已知的融合线性层 -> 分片名称映射。当量化配置不包含packed_modules_mapping时
# (典型的HF FP8配置)用作回退。
_FALLBACK_FUSED_SHARDS: Mapping[str, List[str]] = {  # 回退融合分片映射
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],  # QKV投影 -> Q, K, V分片
    "gate_up_proj": ["gate_proj", "up_proj"],  # gate_up投影 -> gate, up分片
}


def is_layer_skipped(  # 判断层是否被跳过(不量化)
    prefix: str,  # 层的完整路径前缀
    ignored_layers: List[str],  # 被忽略(不量化)的层列表
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),  # 融合层映射
) -> bool:
    # prefix: model.layers.0.self_attn.q_proj
    # 前缀: model.layers.0.self_attn.q_proj
    # proj_name: q_proj
    # 投影名称: q_proj
    proj_name = prefix.split(".")[-1]  # 获取最后一部分作为投影名称

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    # 融合层如gate_up_proj或qkv_proj在safetensors检查点中不会被融合。
    # 因此，我们将名称从融合版本转换为非融合版本，并检查融合层的
    # 每个分片是否具有相同的方案。
    effective_fused = (  # 确定有效的融合映射
        fused_mapping if proj_name in fused_mapping else _FALLBACK_FUSED_SHARDS  # 优先使用传入的映射，否则使用回退映射
    )
    if proj_name in effective_fused:  # 如果投影名称在融合映射中
        shard_prefixes = [  # 生成每个分片的前缀列表
            prefix.replace(proj_name, shard_proj_name)  # 将投影名称替换为分片名称
            for shard_proj_name in effective_fused[proj_name]  # 遍历每个分片名称
        ]

        is_skipped = None  # 初始化跳过标志
        for shard_prefix in shard_prefixes:  # 遍历每个分片前缀
            is_shard_skipped = any(  # 检查分片是否被任何忽略模式匹配
                _module_path_match(ignored, shard_prefix) for ignored in ignored_layers  # 对每个忽略模式进行匹配
            )

            if is_skipped is None:  # 如果是第一个分片
                is_skipped = is_shard_skipped  # 设置初始值
            elif is_shard_skipped != is_skipped:  # 如果分片之间不一致
                raise ValueError(  # 抛出错误
                    f"Detected some but not all shards of {prefix} "  # 检测到部分分片被跳过
                    "are quantized. All shards of fused layers "  # 融合层的所有分片
                    "to have the same precision."  # 必须具有相同的精度
                )
    else:  # 非融合层
        is_skipped = any(  # 检查是否被任何忽略模式匹配
            _module_path_match(ignored, prefix) for ignored in ignored_layers  # 对每个忽略模式进行匹配
        )
        if "gate_up_proj" in prefix:  # 如果前缀包含gate_up_proj
            prefix_gate = prefix.replace("gate_up_proj", "gate_proj")  # 生成gate投影前缀
            prefix_up = prefix.replace("gate_up_proj", "up_proj")  # 生成up投影前缀
            if prefix_gate in ignored_layers and prefix_up in ignored_layers:  # 如果两个分片都被忽略
                is_skipped = True  # 标记为跳过
        elif "experts" in prefix:  # 如果前缀包含experts(专家层)
            # Expert names can include full module paths; keep coarse prefix matches
            # (e.g., "model.layers.{i}.") while also checking expert-specific entries.
            # 专家名称可以包含完整的模块路径；保留粗粒度前缀匹配
            # (例如"model.layers.{i}.")，同时检查专家特定条目。
            is_skipped = is_skipped or any(  # 检查专家特定条目
                prefix in layer_name  # 检查前缀是否在忽略层名称中
                for layer_name in ignored_layers  # 遍历忽略层
                if "experts" in layer_name  # 仅检查包含experts的忽略层
            )

    assert is_skipped is not None  # 断言跳过标志已设置
    return is_skipped  # 返回是否跳过


def per_tensor_dequantize(  # 逐张量反量化
    tensor: torch.Tensor, inv_scale: Union[float, torch.Tensor]  # 量化的张量和逆缩放因子
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)  # 将量化张量转为float16
    dq_weight = fake_qweight * inv_scale  # 乘以逆缩放因子得到反量化权重
    return dq_weight  # 返回反量化权重


def all_close_1d(x: torch.Tensor) -> bool:  # 检查1维张量所有元素是否接近
    assert len(x.shape) == 1  # 断言为1维张量
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))  # 检查所有元素是否与第一个元素接近


def convert_to_channelwise(  # 将逐张量缩放因子转换为逐通道缩放因子
    weight_scale: torch.Tensor, logical_widths: List[int]  # 权重缩放因子和逻辑宽度列表
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create channelwise buffer
    # 创建逐通道缓冲区
    weight_scale_channel = torch.empty(  # 创建空张量
        (sum(logical_widths), 1), dtype=torch.float32, device=weight_scale.device  # 总行数x1，float32
    )

    # Handle scalar tensor case: broadcast same scale to all channels
    # 处理标量张量情况: 将相同的缩放因子广播到所有通道
    if weight_scale.dim() == 0:  # 如果缩放因子是标量(0维)
        weight_scale_channel.fill_(weight_scale.item())  # 用标量值填充所有通道
        return weight_scale_channel  # 返回填充后的通道缩放因子

    # Expand each scale to match the size of each logical matrix.
    # 扩展每个缩放因子以匹配每个逻辑矩阵的大小。
    start = 0  # 起始索引
    for idx, logical_width in enumerate(logical_widths):  # 遍历每个逻辑宽度
        end = start + logical_width  # 计算结束索引
        weight_scale_channel[start:end, :] = weight_scale[idx]  # 将对应的缩放因子填充到通道
        start = end  # 更新起始索引

    return weight_scale_channel  # 返回逐通道缩放因子


def requantize_with_max_scale(  # 使用最大缩放因子重新量化
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: List[int]  # 权重、缩放因子、逻辑宽度
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    # 用于重新量化的最大缩放因子。
    max_w_scale = weight_scale.max()  # 获取缩放因子的最大值

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    # 如果磁盘检查点中的QKV/MLP是融合的，则某些权重缩放因子
    # 仍设为默认值，因为我们为N个分片初始化了N个缩放因子，
    # 但在这种情况下只从磁盘加载1个缩放因子。此时跳过重新量化
    # (因为我们已经用单个缩放因子完成了量化)。
    # * 示例模型: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (  # 判断检查点中的模块是否非融合
        weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min  # 检查最后一个缩放因子是否大于最小值
    )

    # If unfused checkpoint, need requanize with the single scale.
    # 如果是非融合检查点，需要用单个缩放因子重新量化。
    if unfused_module_in_checkpoint:  # 如果是非融合检查点
        start = 0  # 起始索引
        for idx, logical_width in enumerate(logical_widths):  # 遍历每个逻辑宽度
            end = start + logical_width  # 计算结束索引
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])  # 反量化
            weight[start:end, :], _ = scaled_fp8_quant(weight_dq, max_w_scale)  # 用最大缩放因子重新量化
            start = end  # 更新起始索引

    return max_w_scale, weight  # 返回最大缩放因子和重新量化后的权重


def update_tensor_inplace(old: torch.Tensor, new: torch.Tensor) -> None:  # 原地更新张量
    old.copy_(new)  # 将新张量的值复制到旧张量


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/layer_utils.py
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/layer_utils.py
# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
# 新生成的张量需要替换vLLM已注册为参数的现有张量(这些张量不会被释放)
def replace_parameter(  # 替换模块中的参数
    mod: torch.nn.Module, name: str, new: Union[torch.Tensor, torch.nn.Parameter]  # 模块、参数名、新参数
) -> None:

    old = getattr(mod, name)  # 获取旧参数
    if (  # 如果可以原地更新
        type(old) is type(new)  # 类型相同
        and old.dtype == new.dtype  # 数据类型相同
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()  # 存储大小相同
    ):
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        # 如果可以原地更新以避免重新注册
        #   当底层存储相同时会更快
        update_tensor_inplace(old, new)  # 原地更新
    else:  # 不能原地更新
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        # 回退方案: 重新注册参数，必要时转换为Parameter
        # 这不仅确保我们不会将张量注册为参数，还确保所有参数子类
        # 都被重新注册为参数以兼容`torch.compile`
        if not isinstance(new, torch.nn.Parameter):  # 如果新值不是Parameter
            new = torch.nn.Parameter(new, requires_grad=False)  # 转换为Parameter
        mod.register_parameter(name, torch.nn.Parameter(new, requires_grad=False))  # 重新注册参数


def assert_fp8_all_close(a: torch.Tensor, b: torch.Tensor):  # 断言两个FP8张量近似相等
    assert a.shape == b.shape  # 断言形状相同
    assert a.dtype == b.dtype == torch.float8_e4m3fn  # 断言数据类型为FP8 E4M3

    a_u8 = a.view(torch.uint8)  # 将a视为uint8视图
    b_u8 = b.view(torch.uint8)  # 将b视为uint8视图
    diff_u8 = (a_u8.to(torch.int16) - b_u8.to(torch.int16)).abs()  # 计算uint8差值的绝对值

    numel = a.numel()  # 获取元素总数

    count_diff_sign = ((a_u8 >= 0) & (b_u8 < 0)).sum().item()  # 统计符号不同的元素数
    count_tiny_diff = (diff_u8 >= 1).sum().item()  # 统计差值>=1的元素数
    count_large_diff = (diff_u8 >= 2).sum().item()  # 统计差值>=2的元素数

    assert (  # 断言误差在可接受范围内
        (count_diff_sign == 0)  # 没有符号差异
        and (count_tiny_diff / numel < 0.005)  # 小差异比例<0.5%
        and (count_large_diff == 0)  # 没有大差异
    ), f"{count_diff_sign=} {count_tiny_diff=} {count_large_diff=} {numel=}"  # 否则输出统计信息


# Match dynamic rules with module name (prefix) and override quantize
# config if module (prefix) matches a rule
# 将动态规则与模块名(前缀)匹配，如果模块(前缀)匹配规则则覆盖量化配置
def override_config(config: QuantizationConfig, prefix: str):  # 覆盖量化配置
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)  # 获取动态覆盖的权重位数
    if isinstance(weight_bits, int):  # 如果是整数
        config.weight_bits = weight_bits  # 更新权重位数
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)  # 获取动态覆盖的组大小
    if isinstance(group_size, int):  # 如果是整数
        config.group_size = group_size  # 更新组大小
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)  # 获取动态覆盖的desc_act
    if isinstance(desc_act, bool):  # 如果是布尔值
        config.desc_act = desc_act  # 更新desc_act

    config.pack_factor = 32 // config.weight_bits  # packed into int32 # 打包到int32中的因子
    if config.get_name() == "gptq_marlin":  # 如果是GPTQ Marlin量化
        is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)  # 获取动态覆盖的对称性
        if isinstance(is_sym, bool):  # 如果是布尔值
            config.is_sym = is_sym  # 更新对称性

        if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:  # 如果配置不在类型映射中
            raise ValueError(  # 抛出错误
                "Unsupported quantization config: "  # 不支持的量化配置
                f"bits={config.weight_bits}, sym={config.is_sym}"  # 显示位数和对称性
            )

        config.quant_type = config.TYPE_MAP[(config.weight_bits, config.is_sym)]  # 设置量化类型
    elif config.get_name() == "gptq":  # 如果是GPTQ量化
        if config.weight_bits not in [2, 3, 4, 8]:  # 如果权重位数不在支持范围内
            raise ValueError(  # 抛出错误
                "Currently, only 2/3/4/8-bit weight quantization is "  # 当前仅支持2/3/4/8位权重量化
                f"supported for GPTQ, but got {config.weight_bits} bits."  # 但得到了指定位数
            )


def get_dynamic_override(  # 获取动态覆盖配置
    config: QuantizationConfig,  # 量化配置
    layer_name: str,  # 层名称
    key: Optional[str] = None,  # 配置键(可选)
    default_value: Union[int, bool, None] = None,  # 默认值
) -> Union[Dict, int, bool, None]:
    for pattern, pattern_dict in config.dynamic.items():  # 遍历动态配置模式
        # Negative match: matched modules are excluded from quantized init
        # 负向匹配: 匹配的模块从量化初始化中排除
        if pattern.startswith("-:"):  # 如果是负向匹配模式
            if re.match(pattern.removeprefix("-:"), layer_name):  # 匹配层名称
                return False  # 返回False表示排除
        # Positive match: matched modules have quant properties overrides
        # base quant config
        # 正向匹配: 匹配的模块有量化属性覆盖
        # 基础量化配置
        elif re.match(pattern.removeprefix("+:"), layer_name):  # 正向匹配层名称
            if key is None:  # 如果未指定键
                return pattern_dict  # 返回整个模式字典
            else:  # 指定了键
                return pattern_dict.get(key, default_value)  # 返回指定键的值或默认值
    return default_value  # 无匹配时返回默认值


def get_linear_quant_method(  # 获取线性层量化方法
    config: QuantizationConfig,  # 量化配置
    layer: torch.nn.Module,  # 网络层
    prefix: str,  # 层前缀
    linear_method_cls: type,  # 线性方法类
):
    from sglang.srt.layers.linear import LinearBase  # 导入线性基类
    from sglang.srt.layers.quantization.unquant import (  # 导入未量化方法
        UnquantizedEmbeddingMethod,  # 未量化嵌入方法
        UnquantizedLinearMethod,  # 未量化线性方法
    )
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行词表嵌入头

    cloned_config = deepcopy(config)  # 深拷贝配置以避免修改原始配置
    parallel_lm_head_quantized = (  # 判断并行语言模型头是否被量化
        isinstance(layer, ParallelLMHead) and cloned_config.lm_head_quantized  # 是并行语言模型头且配置了量化
    )

    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:  # 如果是线性层或量化的语言模型头
        # False = skip module, None = no override, else = Positive match
        # False = 跳过模块, None = 无覆盖, 否则 = 正向匹配
        if get_dynamic_override(cloned_config, layer_name=prefix) is False:  # 如果动态覆盖为False
            if parallel_lm_head_quantized:  # 如果是量化的语言模型头
                return UnquantizedEmbeddingMethod()  # 返回未量化嵌入方法
            return UnquantizedLinearMethod()  # 返回未量化线性方法

        if prefix:  # 如果前缀不为空
            # Dynamic per module/layer rules may override base config
            # 每个模块/层的动态规则可能覆盖基础配置
            override_config(cloned_config, prefix=prefix)  # 覆盖配置

        return linear_method_cls(cloned_config)  # 返回线性量化方法实例
    return None  # 非线性层返回None


def get_pack_factor(num_bits):  # 获取打包因子(每个int32能打包多少个num_bits位数)
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"  # 断言32能被位数整除
    return 32 // num_bits  # 返回打包因子


def permute_rows(  # 行排列函数(用于模拟act_order)
    q_w: torch.Tensor,  # 量化权重
    w_ref: torch.Tensor,  # 参考权重
    group_size: int,  # 组大小
    test_perm: Optional[torch.Tensor] = None,  # 测试用排列(可选)
):
    assert q_w.shape == w_ref.shape  # 断言形状相同

    orig_device = q_w.device  # 保存原始设备
    k_size, _ = q_w.shape  # 获取K维度大小

    g_idx = torch.zeros((k_size,), dtype=torch.int32)  # 初始化组索引
    for i in range(k_size):  # 遍历每行
        g_idx[i] = i // group_size  # 计算组索引

    # Simulate act_order by doing a random permutation on K
    # 通过对K维度进行随机排列来模拟act_order
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)  # 使用测试排列或随机排列

    g_idx = g_idx[rand_perm].contiguous()  # 按排列重排组索引
    q_w = q_w[rand_perm, :].contiguous()  # 按排列重排量化权重
    w_ref = w_ref[rand_perm, :].contiguous()  # 按排列重排参考权重

    return (  # 返回元组
        w_ref.to(device=orig_device),  # 参考权重(移回原设备)
        q_w.to(device=orig_device),  # 量化权重(移回原设备)
        g_idx.to(device=orig_device),  # 组索引(移回原设备)
        rand_perm.to(device=orig_device),  # 排列(移回原设备)
    )


def pack_cols(  # 按列打包量化权重
    q_w: torch.Tensor,  # 量化权重
    num_bits: int,  # 每个值的位数
    size_k: int,  # K维度大小
    size_n: int,  # N维度大小
):
    assert q_w.shape == (size_k, size_n)  # 断言形状匹配

    pack_factor = get_pack_factor(num_bits)  # 获取打包因子
    assert size_n % pack_factor == 0  # 断言N维度能被打包因子整除

    orig_device = q_w.device  # 保存原始设备

    q_w = q_w.cpu().numpy().astype(numpy.uint32)  # 转为CPU上的uint32 NumPy数组

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)  # 创建结果数组

    for i in range(pack_factor):  # 遍历每个打包位
        q_res |= q_w[:, i::pack_factor] << num_bits * i  # 将值左移并按位或

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)  # 转回PyTorch张量并移回原设备
    q_res = q_res.contiguous()  # 使内存连续

    return q_res  # 返回打包结果


def pack_rows(  # 按行打包量化权重
    q_w: torch.Tensor,  # 量化权重
    num_bits: int,  # 每个值的位数
    size_k: int,  # K维度大小
    size_n: int,  # N维度大小
):
    assert q_w.shape == (size_k, size_n)  # 断言形状匹配

    pack_factor = get_pack_factor(num_bits)  # 获取打包因子
    assert size_k % pack_factor == 0  # 断言K维度能被打包因子整除

    orig_device = q_w.device  # 保存原始设备

    q_w = q_w.cpu().numpy().astype(numpy.uint32)  # 转为CPU上的uint32 NumPy数组

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)  # 创建结果数组

    for i in range(pack_factor):  # 遍历每个打包位
        q_res |= q_w[i::pack_factor, :] << num_bits * i  # 将值左移并按位或

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)  # 转回PyTorch张量并移回原设备
    return q_res  # 返回打包结果


def unpack_cols(  # 按列解包量化权重
    packed_q_w: torch.Tensor,  # 打包的量化权重
    num_bits: int,  # 每个值的位数
    size_k: int,  # K维度大小
    size_n: int,  # N维度大小
):
    pack_factor = get_pack_factor(num_bits)  # 获取打包因子
    assert size_n % pack_factor == 0  # 断言N维度能被打包因子整除
    assert packed_q_w.shape == (  # 断言打包权重形状正确
        size_k,
        size_n // pack_factor,
    ), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(  # 错误信息模板
        packed_q_w.shape, size_k, size_n, pack_factor  # 格式化参数
    )

    orig_device = packed_q_w.device  # 保存原始设备

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)  # 转为CPU上的uint32 NumPy数组
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)  # 创建结果数组

    mask = (1 << num_bits) - 1  # 计算低位掩码
    for i in range(pack_factor):  # 遍历每个打包位
        vals = packed_q_w_cpu & mask  # 提取低num_bits位
        packed_q_w_cpu >>= num_bits  # 右移以处理下一个值
        q_res[:, i::pack_factor] = vals  # 将提取的值放入结果数组的对应位置

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)  # 转回PyTorch张量并移回原设备
    q_res = q_res.contiguous()  # 使内存连续

    return q_res  # 返回解包结果


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py
def quantize_weights(  # 量化权重函数
    w: torch.Tensor,  # 权重张量
    quant_type: ScalarType,  # 量化标量类型
    group_size: Optional[int],  # 组大小(可选)
    zero_points: bool = False,  # 是否使用零点
    ref_zero_points_after_scales: bool = False,  # 参考零点是否在缩放后应用
):
    assert (  # 断言量化类型为整数类型
        quant_type.is_integer()
    ), "Floating point quantization may work but has not been tested"  # 浮点量化可能有效但未测试
    assert not zero_points or group_size is not None, (  # 断言: 如果使用零点则组大小不能为None
        "to have group zero points, group_size must be provided "  # 要使用组零点，必须提供group_size
        "(-1 group_size is channelwise)"  # (-1表示逐通道)
    )

    orig_device = w.device  # 保存原始设备
    orig_type = w.dtype  # 保存原始数据类型
    size_k, size_n = w.shape  # 获取权重形状

    assert w.is_floating_point(), "w must be float"  # 断言权重为浮点类型

    if group_size == -1:  # 如果组大小为-1
        group_size = size_k  # 使用整个K维度作为一个组

    # Reshape to [groupsize, -1]
    # 重塑为[groupsize, -1]
    if group_size is not None and group_size < size_k:  # 如果组大小有效且小于K维度
        w = w.reshape((-1, group_size, size_n))  # 重塑为3维
        w = w.permute(1, 0, 2)  # 交换维度
        w = w.reshape((group_size, -1))  # 重塑为2维

    # Compute scale for each group
    # 计算每组的缩放因子
    max_val = torch.max(w, 0, keepdim=True).values  # 计算每列最大值
    min_val = torch.min(w, 0, keepdim=True).values  # 计算每列最小值

    max_q_val = quant_type.max()  # 获取量化类型最大值
    min_q_val = quant_type.min()  # 获取量化类型最小值

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case # 未缩放情况
    maybe_w_zp = None  # 可能的零点
    if group_size is not None:  # 如果组大小有效
        if zero_points:  # 如果使用零点
            assert not quant_type.is_signed() and quant_type.max() > 0  # 断言量化类型为无符号且最大值>0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()  # 计算缩放因子
            maybe_w_zp = (  # 计算零点
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()  # 四舍五入并钳制
            )
        else:  # 不使用零点
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            # 如果偏差导致没有可能的负/正值，
            # 将最大值设为inf以避免除以0
            w_s = torch.max(  # 计算缩放因子
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),  # 正值缩放
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),  # 负值缩放
            )

    # Quantize
    # 量化
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)  # 四舍五入并加零点
    w_q = torch.clamp(w_q, min_q_val, max_q_val)  # 钳制到有效范围

    # Compute ref (dequantized)
    # 计算参考值(反量化)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    # 对于某些内核(即Machete)，零点在缩放因子应用之后应用，
    # 对于这种情况，以类似方式计算参考值
    # 允许我们在单元测试中使用更严格的误差容限。
    if ref_zero_points_after_scales and maybe_w_zp is not None:  # 如果零点在缩放后应用
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s  # 先量化再减去零点缩放
    else:  # 零点在缩放前应用
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s  # 先减零点再乘缩放因子

    if quant_type.has_bias():  # 如果量化类型有偏移
        w_q += quant_type.bias  # 加上偏移

    # Restore original shapes
    # 恢复原始形状
    if group_size is not None and group_size < size_k:  # 如果组大小有效且小于K维度

        def reshape_w(w):  # 重塑权重的辅助函数
            w = w.reshape((group_size, -1, size_n))  # 重塑为3维
            w = w.permute(1, 0, 2)  # 交换维度
            w = w.reshape((size_k, size_n)).contiguous()  # 重塑回2维并使内存连续
            return w  # 返回重塑后的权重

        w_q = reshape_w(w_q)  # 重塑量化权重
        w_ref = reshape_w(w_ref)  # 重塑参考权重
        w_s = w_s.reshape((-1, size_n)).contiguous()  # 重塑缩放因子

    if maybe_w_zp is not None:  # 如果存在零点
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()  # 重塑零点
        maybe_w_zp = maybe_w_zp.to(device=orig_device)  # 移回原设备

    return (  # 返回元组
        w_ref.to(device=orig_device),  # 参考权重(移回原设备)
        w_q.to(device=orig_device),  # 量化权重(移回原设备)
        w_s if group_size is not None else None,  # 缩放因子(无组大小时为None)
        maybe_w_zp,  # 零点(可能为None)
    )


SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]  # 支持的GPTQ量化类型列表
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]  # 支持的组大小列表


def gptq_quantize_weights(  # GPTQ量化权重函数
    w: torch.Tensor,  # 权重张量
    quant_type: ScalarType,  # 量化标量类型
    group_size: int,  # 组大小
    act_order: bool,  # 是否使用激活排序
    test_perm: Optional[torch.Tensor] = None,  # 测试用排列(可选)
):
    size_k, _ = w.shape  # 获取K维度大小

    assert w.is_floating_point(), "w must be float"  # 断言权重为浮点类型
    assert (  # 断言量化类型受支持
        quant_type in SUPPORTED_GPTQ_QUANT_TYPES
    ), f"Unsupported gptq type = {quant_type}"  # 不支持的GPTQ类型
    assert group_size in SUPPORTED_GROUP_SIZES + [  # 断言组大小受支持
        size_k
    ], f"Unsupported groupsize = {group_size}"  # 不支持的组大小

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)  # 量化权重

    # Apply act_order
    # 应用激活排序
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)  # 初始化组索引为空
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)  # 初始化排列为空
    if act_order:  # 如果使用激活排序
        assert (  # 断言组大小小于K维度
            group_size < size_k
        ), "For act_order, groupsize = {} must be less than size_k = {}".format(  # 对于act_order，组大小必须小于K维度
            group_size, size_k  # 格式化参数
        )

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)  # 行排列

    return w_ref, w_q, w_s, g_idx, rand_perm  # 返回参考权重、量化权重、缩放因子、组索引、排列


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):  # 按组索引排序权重
    orig_device = q_w.device  # 保存原始设备

    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)  # Sort based on g_idx # 基于g_idx排序

    g_idx = g_idx[sort_indices].contiguous()  # 按排序索引重排组索引
    q_w = q_w[sort_indices, :].contiguous()  # 按排序索引重排量化权重

    return (  # 返回元组
        q_w.to(device=orig_device),  # 量化权重(移回原设备)
        g_idx.to(device=orig_device),  # 组索引(移回原设备)
        sort_indices.to(device=orig_device),  # 排序索引(移回原设备)
    )


def swizzle_blockscale(scale: torch.Tensor):  # 交错重排块缩放因子(用于NVFP4量化)
    """
    Swizzle the scale tensor into a blockwise interleaved format for NVFP4 quantization.
    # 将缩放张量交错重排为NVFP4量化的块级交错格式。
    """
    assert scale.dtype == torch.float8_e4m3fn  # 断言缩放因子为FP8 E4M3类型
    # Pad and blockwise interleave weight_scale
    # 填充并块级交错权重缩放因子
    scale_ndim = scale.ndim  # 保存原始维度数
    if scale.ndim == 2:  # 如果是2维
        scale = scale.unsqueeze(0)  # 添加批次维度
    assert scale.ndim == 3  # 断言现在是3维
    B, M, K = scale.shape  # 获取批次、行、列维度
    round_up_multiple = lambda x, m: (x + m - 1) // m * m  # 向上取整到m的倍数的lambda
    M_padded = round_up_multiple(M, 128)  # M向上取整到128的倍数
    K_padded = round_up_multiple(K, 4)  # K向上取整到4的倍数
    padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)  # 创建填充后的缩放张量
    padded_scale[:B, :M, :K] = scale  # 将原始缩放因子填入
    batches, rows, cols = padded_scale.shape  # 获取填充后的维度
    assert rows % 128 == 0  # 断言行数是128的倍数
    assert cols % 4 == 0  # 断言列数是4的倍数
    padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32, cols // 4, 4)  # 重塑为6维
    swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))  # 按指定顺序交换维度
    swizzled_scale = swizzled_scale.contiguous().cuda()  # 使内存连续并移到GPU
    return (  # 返回交错后的缩放因子
        swizzled_scale.reshape(M_padded, K_padded)  # 如果原始是2维则重塑为2维
        if scale_ndim == 2  # 原始2维
        else swizzled_scale.reshape(B, M_padded, K_padded)  # 否则重塑为3维
    )


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:  # 交换w1和w3的顺序
    return (  # 返回交换后的张量
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)  # 将w1和w3两半翻转
    )


def reorder_w1w3_to_w3w1(  # 将[w1, w3]顺序重排为[w3, w1]
    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2  # 权重、缩放因子、操作维度
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order the concatenated `[w1, w3]` tensors to `[w3, w1]`"""
    # 重新排列拼接的`[w1, w3]`张量为`[w3, w1]`
    size = weight.size(dim)  # 获取指定维度的大小
    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"  # 断言维度大小为偶数
    half = size // 2  # 计算一半的大小

    w1, w3 = weight.split(half, dim=dim)  # 将权重分为w1和w3两半
    s1, s3 = scale.split(half, dim=dim)  # 将缩放因子分为s1和s3两半

    return (  # 返回重排后的元组
        torch.cat([w3, w1], dim=dim).contiguous(),  # w3在前，w1在后
        torch.cat([s3, s1], dim=dim).contiguous(),  # s3在前，s1在后
    )


def prepare_static_weights_for_trtllm_fp4_moe(  # 为TRT-LLM FP4 MoE准备静态权重
    gemm1_weights,  # GEMM1权重(w13)
    gemm2_weights,  # GEMM2权重(w2)
    gemm1_scales_linear_fp4_bytes,  # GEMM1 FP4线性缩放因子字节
    gemm2_scales_linear_fp4_bytes,  # GEMM2 FP4线性缩放因子字节
    hidden_size,  # 隐藏层大小
    intermediate_size,  # 中间层大小
    num_experts,  # 专家数量
    is_gated: bool = True,  # 是否为门控激活
):
    from flashinfer import nvfp4_block_scale_interleave  # 导入NVFP4块缩放交错函数
    from flashinfer.fused_moe.core import (  # 导入flashinfer融合MoE核心函数
        _maybe_get_cached_w3_w1_permute_indices,  # 获取缓存的w3/w1排列索引
        get_w2_permute_indices_with_cache,  # 获取缓存的w2排列索引
    )

    """Prepare quantized weights for kernel (done offline with weights)."""
    # 为内核准备量化权重(离线与权重一起完成)。
    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}  # 排列索引缓存
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals # 注意: 这取决于内核内部实现

    gemm1_rows = (2 if is_gated else 1) * intermediate_size  # 计算GEMM1行数

    # Convert quantized weights to proper formats
    # 将量化权重转换为正确的格式
    gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(  # 转换GEMM1权重为FP4格式
        num_experts, gemm1_rows, hidden_size // 2
    )  # packed fp4 # 打包的FP4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(  # 转换GEMM1缩放因子
        torch.float8_e4m3fn
    ).reshape(
        num_experts, gemm1_rows, hidden_size // 16
    )  # fp8 scaling factors # FP8缩放因子

    gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(  # 转换GEMM2权重为FP4格式
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4 # 打包的FP4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(  # 转换GEMM2缩放因子
        torch.float8_e4m3fn
    ).reshape(
        num_experts, hidden_size, intermediate_size // 16
    )  # fp8 scaling factors # FP8缩放因子

    gemm1_weights_fp4_shuffled = []  # GEMM1权重重排结果列表
    gemm1_scales_fp4_shuffled = []  # GEMM1缩放因子重排结果列表
    gemm2_weights_fp4_shuffled = []  # GEMM2权重重排结果列表
    gemm2_scales_fp4_shuffled = []  # GEMM2缩放因子重排结果列表
    for i in range(num_experts):  # 遍历每个专家
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(  # 获取GEMM1权重排列索引
            _cache_permute_indices,  # 缓存字典
            gemm1_weights_fp4[i].view(torch.uint8),  # GEMM1权重(uint8视图)
            epilogue_tile_m,  # 尾声平铺大小
            is_gated_act_gemm=is_gated,  # 是否为门控激活GEMM
        )
        gemm1_weights_fp4_shuffled.append(  # 添加重排后的GEMM1权重
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(  # 获取GEMM1缩放因子排列索引
            _cache_permute_indices,  # 缓存字典
            gemm1_scales_linear_fp4[i].view(torch.uint8),  # GEMM1缩放因子(uint8视图)
            epilogue_tile_m,  # 尾声平铺大小
            num_elts_per_sf=16,  # 每个缩放因子对应的元素数
            is_gated_act_gemm=is_gated,  # 是否为门控激活GEMM
        )
        gemm1_scales_fp4_shuffled.append(  # 添加重排并交错后的GEMM1缩放因子
            nvfp4_block_scale_interleave(  # NVFP4块缩放交错
                gemm1_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

        permute_indices = get_w2_permute_indices_with_cache(  # 获取GEMM2权重排列索引
            _cache_permute_indices,  # 缓存字典
            gemm2_weights_fp4[i].view(torch.uint8),  # GEMM2权重(uint8视图)
            epilogue_tile_m,  # 尾声平铺大小
        )
        gemm2_weights_fp4_shuffled.append(  # 添加重排后的GEMM2权重
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(  # 获取GEMM2缩放因子排列索引
            _cache_permute_indices,  # 缓存字典
            gemm2_scales_linear_fp4[i].view(torch.uint8),  # GEMM2缩放因子(uint8视图)
            epilogue_tile_m,  # 尾声平铺大小
            num_elts_per_sf=16,  # 每个缩放因子对应的元素数
        )
        gemm2_scales_fp4_shuffled.append(  # 添加重排并交错后的GEMM2缩放因子
            nvfp4_block_scale_interleave(  # NVFP4块缩放交错
                gemm2_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

    # Stack weights for all experts
    # 堆叠所有专家的权重
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)  # 堆叠GEMM1权重
    gemm1_scales_fp4_shuffled = (  # 堆叠GEMM1缩放因子并重塑
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, gemm1_rows, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)  # 堆叠GEMM2权重
    gemm2_scales_fp4_shuffled = (  # 堆叠GEMM2缩放因子并重塑
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return (  # 返回四个张量
        gemm1_weights_fp4_shuffled,  # GEMM1重排后的FP4权重
        gemm1_scales_fp4_shuffled,  # GEMM1重排后的FP4缩放因子
        gemm2_weights_fp4_shuffled,  # GEMM2重排后的FP4权重
        gemm2_scales_fp4_shuffled,  # GEMM2重排后的FP4缩放因子
    )
