# SPDX-License-Identifier: Apache-2.0 # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project # SPDX版权声明
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py # 改编自vLLM项目的Marlin量化工具模块

# Marlin量化工具模块：提供Marlin稀疏/密集量化格式的工具函数
# 包括Marlin格式支持检测、权重重排、缩放因子排列、零点处理、GPTQ/AWQ Marlin线性层应用等

from __future__ import annotations  # 启用延迟类型注解评估 # 启用PEP 563延迟注解评估

import logging  # 导入日志模块 # 导入Python标准日志模块
from dataclasses import dataclass  # 导入数据类装饰器 # 导入dataclass装饰器用于定义数据类
from typing import TYPE_CHECKING, Any, Optional  # 导入类型提示 # 导入类型提示工具

import numpy  # 导入NumPy库 # 导入NumPy数值计算库
import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架

from sglang.srt.layers.parameter import (  # 从参数模块导入参数类 # 从自定义参数模块导入各种参数类型
    BasevLLMParameter,  # 基础vLLM参数类 # vLLM基础参数类
    ChannelQuantScaleParameter,  # 通道级量化缩放参数类 # 通道级量化缩放因子参数类
    GroupQuantScaleParameter,  # 组级量化缩放参数类 # 组级量化缩放因子参数类
    PackedvLLMParameter,  # 打包vLLM参数类 # 打包存储的vLLM参数类
)
from sglang.srt.layers.quantization.base_config import (  # 从基础配置导入量化基类 # 从量化基础配置模块导入基类
    LinearMethodBase,  # 线性方法基类 # 线性层量化方法基类
    QuantizationConfig,  # 量化配置基类 # 量化配置基类
)
from sglang.srt.layers.quantization.utils import (  # 从量化工具模块导入工具函数 # 从量化工具模块导入辅助函数
    get_scalar_types,  # 获取标量类型 # 获取标量量化类型集合
    pack_cols,  # 打包列 # 按列打包量化权重
    unpack_cols,  # 解包列 # 按列解包量化权重
)
from sglang.srt.utils import get_device_capability, is_cuda  # 导入设备工具函数 # 导入GPU能力检测和CUDA检测函数
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册装饰器 # 导入自定义算子注册装饰器

if TYPE_CHECKING:  # 仅在类型检查时执行 # 类型检查条件分支
    from sglang.srt.layers.linear import LinearBase  # 导入线性层基类 # 导入线性层基类（仅类型检查时）
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入MoE层类 # 导入融合MoE层类（仅类型检查时）

from sglang.srt.compilation.piecewise_context_manager import get_forward_context  # 导入前向上下文获取函数 # 导入分段编译上下文管理器的前向上下文获取函数

try:  # 尝试导入vLLM自定义算子 # 尝试导入vLLM的自定义C++算子
    from vllm import _custom_ops as ops  # 导入vLLM自定义算子模块 # 将vLLM的C++自定义算子模块别名为ops
except ImportError:  # 导入失败 # 如果vLLM不可用
    ops = None  # 设为None # 将ops设为None


_is_cuda = is_cuda()  # 检测是否为CUDA环境 # 检测当前是否在CUDA环境下运行

if _is_cuda:  # 如果是CUDA环境 # 如果在CUDA环境下
    from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm  # 导入GPTQ Marlin GEMM核函数 # 导入GPTQ Marlin矩阵乘法JIT核函数

logger = logging.getLogger(__name__)  # 创建日志记录器 # 创建当前模块的日志记录器

ScalarType, scalar_types = get_scalar_types()  # 获取标量类型和标量类型集合 # 获取标量量化类型类和类型集合

GPTQ_MARLIN_TILE = 16  # GPTQ Marlin瓦片大小 # Marlin核函数使用的瓦片大小
GPTQ_MARLIN_MIN_THREAD_N = 64  # GPTQ Marlin最小线程N # Marlin核函数N维最小线程数
GPTQ_MARLIN_MIN_THREAD_K = 128  # GPTQ Marlin最小线程K # Marlin核函数K维最小线程数
GPTQ_MARLIN_MAX_PARALLEL = 16  # GPTQ Marlin最大并行数 # Marlin核函数最大并行问题数

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]  # Marlin支持的组大小 # Marlin量化支持的分组大小列表

# In case there is a performance issue with Marlin, the variable below can be # 如果Marlin存在性能问题，下面的变量可以
# changed to False, which allows Marlin to perform global reductions in fp16 # 改为False，允许Marlin以fp16精度执行全局归约
# precision (instead of fp32), and therefore, save on some memory movements. # （而不是fp32），从而节省一些内存搬运
USE_FP32_REDUCE_DEFAULT = True  # 默认使用FP32归约 # 默认使用FP32精度进行全局归约


@dataclass  # 数据类装饰器 # 使用dataclass装饰器定义数据类
class MarlinLinearLayerConfig:  # Marlin线性层配置数据类 # Marlin线性层的配置数据类
    full_weight_shape: tuple[int, int]  # [in, out] # 完整权重形状 [输入维度, 输出维度]
    partition_weight_shape: tuple[int, int]  # 分区权重形状 # 分区后的权重形状
    weight_type: ScalarType  # 权重数据类型 # 量化的权重标量类型
    act_type: torch.dtype  # 激活数据类型 # 激活值的数据类型
    group_size: int  # 组大小 # 量化分组大小
    zero_points: bool  # 是否使用零点 # 是否使用零点
    has_g_idx: bool  # 是否有组索引 # 是否包含组索引


# For binary size and compile time, we don't support the same types for with and # 为了二进制大小和编译时间，我们不为有无运行时零点支持相同类型
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ. # 我们支持常见情况，即AWQ和GPTQ
#  TODO: we may want to move this into the C++ so its closer to the actual impl # 待办：我们可能想将此移到C++中，使其更接近实际实现
def query_marlin_supported_quant_types(  # 查询Marlin支持的量化类型 # 查询Marlin框架支持的量化数据类型
    has_zp: Optional[bool] = None,  # 是否有零点 # 是否包含零点，None表示两者都返回
    include_fp_type: bool = True,  # 是否包含浮点类型 # 是否在结果中包含浮点量化类型
    device_capability: Optional[int] = None,  # 设备计算能力 # GPU计算能力版本号
):  # 返回支持的量化类型列表 # 返回支持的ScalarType列表
    if device_capability is None:  # 如果未指定设备能力 # 如果未提供设备计算能力
        major, minor = get_device_capability()  # 获取设备主次版本号 # 获取GPU的主版本号和次版本号
        capability = major * 10 + minor  # 计算能力值 # 将主次版本号组合为计算能力值
        device_capability = -1 if capability is None else capability  # 未检测到则设为-1 # 如果无法获取则设为-1

    if device_capability < 80:  # 如果设备能力低于SM80 # 如果GPU架构版本低于SM80
        return []  # 返回空列表，不支持Marlin # SM80以下不支持Marlin

    # - has_zp is True: return quant_types that has zero points # has_zp为True：返回有零点的量化类型
    # - has_zp is False: return quant_types that has not zero points # has_zp为False：返回无零点的量化类型
    # - has_zp is None: both # has_zp为None：两者都返回
    if has_zp is None:  # 如果未指定零点选项 # 如果不限定零点条件
        types0 = query_marlin_supported_quant_types(  # 查询无零点类型 # 递归查询无零点的量化类型
            False, include_fp_type, device_capability  # 传入False表示无零点 # 参数：无零点、是否包含浮点类型、设备能力
        )
        types1 = query_marlin_supported_quant_types(  # 查询有零点类型 # 递归查询有零点的量化类型
            True, include_fp_type, device_capability  # 传入True表示有零点 # 参数：有零点、是否包含浮点类型、设备能力
        )
        return types0 + types1  # 合并返回 # 合并两种类型的列表

    if has_zp:  # 如果有零点 # 如果需要运行时零点
        # AWQ style, unsigned + runtime zero-point # AWQ风格，无符号 + 运行时零点
        return [scalar_types.uint4]  # 返回uint4类型 # 返回AWQ风格的无符号4位整型
    else:  # 如果没有零点 # GPTQ风格分支
        # GPTQ style, unsigned + symmetric bias # GPTQ风格，无符号 + 对称偏置
        res = [scalar_types.uint4b8, scalar_types.uint8b128]  # 基本类型列表 # 基本支持的量化类型
        if include_fp_type:  # 如果包含浮点类型 # 如果需要包含浮点量化类型
            res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]  # 添加FP8和FP4类型 # 添加FP8 E4M3和FP4 E2M1浮点类型
        return res  # 返回类型列表 # 返回GPTQ风格支持的量化类型列表


def _check_marlin_supported(  # 内部检查Marlin是否支持 # 内部函数：检查Marlin是否支持给定的量化配置
    quant_type: ScalarType,  # 量化类型 # 标量量化类型
    group_size: Optional[int],  # 组大小 # 量化分组大小
    has_zp: bool,  # 是否有零点 # 是否使用运行时零点
    device_capability: Optional[int] = None,  # 设备计算能力 # GPU计算能力版本号
) -> tuple[bool, Optional[str]]:  # 返回是否支持及错误信息 # 返回(是否支持, 错误信息)元组

    if device_capability is None:  # 如果未指定设备能力 # 如果未提供设备计算能力
        major, minor = get_device_capability()  # 获取设备主次版本号 # 获取GPU的主版本号和次版本号
        capability = major * 10 + minor  # 计算能力值 # 将主次版本号组合为计算能力值
        device_capability = -1 if capability is None else capability  # 未检测到则设为-1 # 如果无法获取则设为-1

    supported_types = query_marlin_supported_quant_types(  # 查询支持的量化类型 # 查询当前配置下支持的量化类型
        has_zp, True, device_capability  # 传入零点选项和设备能力 # 参数：零点选项、包含浮点类型、设备能力
    )

    if quant_type not in supported_types:  # 如果量化类型不在支持列表中 # 如果请求的量化类型不在支持列表中
        return (  # 返回不支持及错误信息 # 返回False及详细错误信息
            False,  # 不支持 # 不支持标志
            f"Marlin does not support weight_bits = {quant_type}. "  # 不支持的权重位数 # 错误信息：不支持的权重位数
            f"Only types = {supported_types} "  # 仅支持的类型 # 错误信息：支持的类型列表
            f"are supported (for group_size = {group_size}, "  # 对于指定的组大小 # 错误信息：对应的分组大小
            f"device_capability = {device_capability}, zp = {has_zp}).",  # 和设备能力及零点 # 错误信息：设备能力和零点配置
        )
    if group_size is None or group_size not in MARLIN_SUPPORTED_GROUP_SIZES:  # 如果组大小不在支持列表中 # 如果分组大小不在支持列表中
        return (  # 返回不支持及错误信息 # 返回False及详细错误信息
            False,  # 不支持 # 不支持标志
            f"Marlin does not support group_size = {group_size}. "  # 不支持的组大小 # 错误信息：不支持的分组大小
            f"Only group_sizes = {MARLIN_SUPPORTED_GROUP_SIZES} "  # 仅支持的组大小 # 错误信息：支持的分组大小列表
            "are supported.",  # 被支持 # 错误信息后缀
        )

    return True, None  # 返回支持，无错误 # 返回True表示支持，无错误信息


def check_marlin_supported(  # 检查Marlin是否支持（简化版） # 检查Marlin是否支持给定的量化配置（仅返回布尔值）
    quant_type: ScalarType,  # 量化类型 # 标量量化类型
    group_size: int,  # 组大小 # 量化分组大小
    has_zp: bool = False,  # 是否有零点，默认False # 是否使用运行时零点，默认否
    device_capability: Optional[int] = None,  # 设备计算能力 # GPU计算能力版本号
) -> bool:  # 返回是否支持 # 返回布尔值表示是否支持
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp, device_capability)  # 调用内部检查函数 # 调用内部检查函数获取结果
    return cond  # 返回是否支持 # 仅返回支持与否的布尔值


def verify_marlin_supported(  # 验证Marlin支持（不满足则抛异常） # 验证Marlin是否支持，不支持则抛出ValueError
    quant_type: ScalarType, group_size: int, has_zp: bool = False  # 量化类型、组大小、零点选项 # 量化类型、分组大小、是否使用零点
) -> None:  # 无返回值 # 无返回值
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)  # 调用内部检查函数 # 调用内部检查函数获取结果和错误信息
    if not cond:  # 如果不支持 # 如果不支持该配置
        assert err_msg is not None  # 确保有错误信息 # 确保错误信息不为空
        raise ValueError(err_msg)  # 抛出值错误 # 抛出ValueError异常


def verify_marlin_supports_shape(  # 验证Marlin是否支持给定形状 # 验证Marlin是否支持给定的权重形状，不支持则抛出异常
    output_size_per_partition: int,  # 分区输出大小 # 分区后的输出维度大小
    input_size_per_partition: int,  # 分区输入大小 # 分区后的输入维度大小
    input_size: int,  # 总输入大小 # 完整的输入维度大小
    group_size: int,  # 组大小 # 量化分组大小
) -> None:  # 无返回值 # 无返回值

    # Validate output_size_per_partition # 验证output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:  # 输出大小必须能被最小线程N整除 # 检查输出维度是否能被最小线程数整除
        raise ValueError(  # 抛出值错误 # 抛出数值异常
            f"Weight output_size_per_partition = "  # 权重输出大小 # 错误信息前缀
            f"{output_size_per_partition} is not divisible by "  # 不能被整除 # 错误信息：不能被整除
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "  # 最小线程N # 错误信息：最小线程N值
            "Consider reducing tensor_parallel_size or running "  # 考虑减少张量并行大小或运行 # 错误信息：建议减少并行度
            "with --quantization gptq."  # 使用--quantization gptq # 错误信息：建议使用gptq量化
        )

    # Validate input_size_per_partition # 验证input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:  # 输入大小必须能被最小线程K整除 # 检查输入维度是否能被最小线程数整除
        raise ValueError(  # 抛出值错误 # 抛出数值异常
            f"Weight input_size_per_partition = "  # 权重输入大小 # 错误信息前缀
            f"{input_size_per_partition} is not divisible "  # 不能被整除 # 错误信息：不能被整除
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "  # 最小线程K # 错误信息：最小线程K值
            "Consider reducing tensor_parallel_size or running "  # 考虑减少张量并行大小或运行 # 错误信息：建议减少并行度
            "with --quantization gptq."  # 使用--quantization gptq # 错误信息：建议使用gptq量化
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:  # 分区输入大小必须能被组大小整除 # 检查分区输入维度是否能被分组大小整除
        raise ValueError(  # 抛出值错误 # 抛出数值异常
            f"Weight input_size_per_partition = {input_size_per_partition}"  # 权重输入大小 # 错误信息前缀
            f" is not divisible by group_size = {group_size}. "  # 不能被组大小整除 # 错误信息：不能被分组大小整除
            "Consider reducing tensor_parallel_size or running "  # 考虑减少张量并行大小或运行 # 错误信息：建议减少并行度
            "with --quantization gptq."  # 使用--quantization gptq # 错误信息：建议使用gptq量化
        )


def check_marlin_supports_shape(  # 检查Marlin是否支持给定形状（返回布尔值） # 检查Marlin是否支持给定的权重形状
    output_size_per_partition: int,  # 分区输出大小 # 分区后的输出维度大小
    input_size_per_partition: int,  # 分区输入大小 # 分区后的输入维度大小
    input_size: int,  # 总输入大小 # 完整的输入维度大小
    group_size: int,  # 组大小 # 量化分组大小
) -> tuple[bool, Optional[str]]:  # 返回是否支持及错误信息 # 返回(是否支持, 错误信息)元组
    try:  # 尝试验证 # 尝试执行验证
        verify_marlin_supports_shape(  # 调用验证函数 # 调用验证函数
            output_size_per_partition, input_size_per_partition, input_size, group_size  # 传入形状参数 # 传入各维度参数
        )
    except ValueError as e:  # 捕获值错误 # 如果验证失败
        return False, e.__str__()  # 返回不支持及错误信息 # 返回False及错误信息字符串
    return True, None  # 返回支持，无错误 # 返回True表示支持


def check_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:  # 检查Marlin是否支持给定层 # 检查Marlin是否支持给定的线性层
    output_size_per_partition = (  # 获取分区输出大小 # 获取分区后的输出维度大小
        getattr(layer, "output_size_per_partition", None) or layer.output_size  # 优先使用分区大小 # 优先获取分区输出大小，否则使用完整输出大小
    )
    input_size_per_partition = (  # 获取分区输入大小 # 获取分区后的输入维度大小
        getattr(layer, "input_size_per_partition", None) or layer.input_size  # 优先使用分区大小 # 优先获取分区输入大小，否则使用完整输入大小
    )

    return check_marlin_supports_shape(  # 检查形状支持 # 调用形状支持检查函数
        output_size_per_partition=output_size_per_partition,  # 分区输出大小 # 传入分区输出维度
        input_size_per_partition=input_size_per_partition,  # 分区输入大小 # 传入分区输入维度
        input_size=layer.input_size,  # 总输入大小 # 传入完整输入维度
        group_size=group_size,  # 组大小 # 传入分组大小
    )[0]  # 取第一个元素（布尔值） # 取元组的第一个元素（是否支持的布尔值）


def check_moe_marlin_supports_layer(layer: FusedMoE, group_size: int) -> bool:  # 检查MoE Marlin是否支持给定层 # 检查MoE层是否支持Marlin量化
    hidden_size = layer.hidden_size  # 获取隐藏层大小 # 获取MoE层的隐藏维度
    intermediate_size_per_partition = layer.intermediate_size_per_partition  # 获取分区中间层大小 # 获取分区后的中间层维度大小
    # apply_router_weight_on_input is not supported for moe marlin # MoE Marlin不支持在输入上应用路由权重
    supports_router_weight = not layer.moe_runner_config.apply_router_weight_on_input  # 检查路由权重支持 # 检查是否不在输入上应用路由权重
    # moe marlin requires the activation to be silu # MoE Marlin要求激活函数为SiLU
    supports_activation = layer.moe_runner_config.activation == "silu"  # 检查激活函数是否为SiLU # 检查MoE的激活函数是否为SiLU

    # gate-up: (n, k) = (intermediate_size_per_partition * 2, hidden_size) # gate-up权重形状：(n, k) = (中间层*2, 隐藏层)
    # down: (n, k) = (hidden_size, intermediate_size_per_partition) # down权重形状：(n, k) = (隐藏层, 中间层)
    # moe marlin requires n % 128 == 0 and k % 64 == 0 # MoE Marlin要求n能被128整除，k能被64整除
    supports_shape = (  # 检查形状是否满足要求 # 检查维度是否满足Marlin要求
        hidden_size % 128 == 0  # 隐藏层大小能被128整除 # 隐藏维度能被128整除
        and intermediate_size_per_partition % max(64, group_size) == 0  # 中间层大小能被64或组大小整除 # 中间层维度能被64或分组大小的较大者整除
    )
    supports_group_size = group_size in [-1, 32, 64, 128]  # 检查组大小是否受支持 # 检查分组大小是否在支持列表中
    return (  # 返回是否全部支持 # 返回所有条件的综合判断
        supports_shape  # 形状支持 # 形状是否满足要求
        and supports_group_size  # 组大小支持 # 分组大小是否受支持
        and supports_router_weight  # 路由权重支持 # 路由权重配置是否支持
        and supports_activation  # 激活函数支持 # 激活函数是否为SiLU
    )


def marlin_make_workspace(  # 创建Marlin工作空间 # 为Marlin核函数创建工作空间张量
    device: torch.device, max_blocks_per_sm: int = 1  # 设备和每个SM的最大块数 # 计算设备和每个流多处理器的最大线程块数
) -> torch.Tensor:  # 返回工作空间张量 # 返回工作空间张量
    # In the new marlin kernel, we use the num of threadblocks as workspace # 在新的Marlin核函数中，我们使用线程块数作为工作空间大小
    # size. The num of threadblocks is sms_count * max_blocks_per_sm. # 线程块数为SM数量 * 每个SM的最大块数
    sms = torch.cuda.get_device_properties(device).multi_processor_count  # 获取SM数量 # 获取GPU的流多处理器数量
    return torch.zeros(  # 返回零张量 # 创建零初始化的工作空间张量
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False  # 形状、类型、设备和梯度设置 # 指定大小、数据类型、设备和不需要梯度
    )


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:  # 判断K维度是否完整 # 判断Marlin核函数中K维度是否完整
    return (not act_order) or (act_order and not is_row_parallel)  # 非激活排序，或激活排序且非行并行 # 无激活排序时为True，或有激活排序但非行并行时为True


def marlin_repeat_scales_on_all_ranks(  # 判断是否需要在所有秩上重复缩放因子 # 判断是否需要在所有张量并行秩上复制缩放因子
    act_order: bool, group_size: int, is_row_parallel: bool  # 激活排序、组大小、是否行并行 # 激活排序标志、分组大小、是否行并行
) -> bool:  # 返回是否需要重复 # 返回布尔值
    # Need to repeat scales on every rank if act_ordering or # 如果使用激活排序或
    # channelwise and RowParallelLinear # 通道级且为行并行线性层，则需要在每个秩上重复缩放因子
    is_channelwise = group_size == -1  # 判断是否为通道级 # 分组大小为-1表示通道级量化
    return act_order or (is_channelwise and is_row_parallel)  # 激活排序或通道级+行并行 # 满足任一条件则需要重复


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:  # 创建空的组索引 # 创建空的组索引张量
    return torch.nn.Parameter(  # 返回参数张量 # 创建不需要梯度的参数张量
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False  # 空int张量 # 空1维int张量
    )


def marlin_make_empty_zp(device: torch.device) -> torch.Tensor:  # 创建空的零点 # 创建空的零点张量
    return torch.nn.Parameter(  # 返回参数张量 # 创建不需要梯度的参数张量
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False  # 空int张量 # 空1维int张量
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # 排序组索引 # 对组索引进行排序，返回排序后的索引和排序下标
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)  # 计算排序下标 # 获取g_idx的排序索引
    return g_idx[g_idx_sort_indices], g_idx_sort_indices  # 返回排序后的g_idx和排序下标 # 返回排序后的组索引和对应的排序下标


def get_scale_perms():  # 获取缩放因子排列 # 获取Marlin格式的缩放因子排列映射
    scale_perm: list[int] = []  # 初始化缩放排列列表 # 初始化完整缩放排列列表
    for i in range(8):  # 遍历8个位置 # 循环8次
        scale_perm.extend([i + 8 * j for j in range(8)])  # 生成排列索引 # 生成8x8排列映射
    scale_perm_single: list[int] = []  # 初始化单缩放排列列表 # 初始化通道级缩放排列列表
    for i in range(4):  # 遍历4个位置 # 循环4次
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])  # 生成单排列索引 # 生成通道级排列映射
    return scale_perm, scale_perm_single  # 返回两种排列 # 返回完整排列和通道级排列


def marlin_permute_scales(  # 排列缩放因子 # 对缩放因子进行Marlin格式的排列
    s: torch.Tensor, size_k: int, size_n: int, group_size: int  # 缩放张量、K维度、N维度、组大小 # 缩放因子张量及维度参数
) -> torch.Tensor:  # 返回排列后的缩放因子 # 返回排列后的缩放因子张量

    scale_perm, scale_perm_single = get_scale_perms()  # 获取排列映射 # 获取两种排列映射
    if group_size < size_k and group_size != -1:  # 如果使用组级量化 # 如果是组级量化（非通道级）
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]  # 应用完整排列 # 应用完整缩放排列
    else:  # 通道级量化 # 通道级量化分支
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]  # 应用通道级排列 # 应用通道级缩放排列
    s = s.reshape((-1, size_n)).contiguous()  # 重塑为2D并确保连续 # 重塑为2D形状并确保内存连续

    return s  # 返回排列结果 # 返回排列后的缩放因子


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:  # 排列偏置 # 对偏置进行Marlin格式的排列
    origin_shape = s.shape  # 保存原始形状 # 保存偏置的原始形状
    _, scale_perm_single = get_scale_perms()  # 获取通道级排列映射 # 获取通道级排列映射（忽略完整排列）
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]  # 应用通道级排列 # 应用通道级排列
    return s.reshape(*origin_shape).contiguous()  # 恢复原始形状并确保连续 # 恢复原始形状并确保内存连续


def marlin_moe_permute_scales(  # 排列MoE缩放因子 # 对MoE专家的缩放因子逐个进行Marlin排列
    s: torch.Tensor,  # 缩放张量 # 缩放因子张量
    size_k: int,  # K维度 # K维度大小
    size_n: int,  # N维度 # N维度大小
    group_size: int,  # 组大小 # 量化分组大小
):  # 返回排列后的缩放因子 # 返回排列后的缩放因子张量
    num_experts = s.shape[0]  # 获取专家数量 # 获取MoE的专家数量
    output = torch.empty(  # 创建输出张量 # 创建空输出张量
        (num_experts, s.shape[1], s.shape[2]),  # 形状 # 指定输出形状
        device=s.device,  # 设备 # 使用相同设备
        dtype=s.dtype,  # 数据类型 # 使用相同数据类型
    )

    for e in range(num_experts):  # 遍历每个专家 # 逐个处理每个专家
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)  # 对每个专家排列缩放因子 # 对当前专家的缩放因子进行Marlin排列
    return output  # 返回排列结果 # 返回排列后的缩放因子张量


def marlin_zero_points(  # 处理Marlin零点 # 将零点转换为Marlin格式
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int  # 零点张量、K维度、N维度、位数 # 零点张量及维度和位宽参数
) -> torch.Tensor:  # 返回处理后的零点 # 返回Marlin格式的零点张量
    # Permute zero-points in a similar way to scales, but do not use the # 以与缩放因子类似的方式排列零点，但不使用
    # "single" permutation, since zero-points are applied on every MMA # "single"排列，因为零点在每个MMA操作上都会应用
    scale_perm, _ = get_scale_perms()  # 获取完整排列映射 # 获取完整缩放排列映射（忽略通道级映射）
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]  # 应用排列 # 对零点应用完整排列

    # Interleave column dim (for the dequantize code) and pack it to int32 # 交错列维度（用于反量化代码）并打包为int32
    if num_bits == 4:  # 如果是4位量化 # 4位量化的交错模式
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])  # 4位交错排列 # 4位量化的交错索引
    elif num_bits == 8:  # 如果是8位量化 # 8位量化的交错模式
        interleave = numpy.array([0, 2, 1, 3])  # 8位交错排列 # 8位量化的交错索引
    else:  # 其他位数 # 不支持的位宽
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))  # 抛出异常 # 抛出不支持的位宽异常

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()  # 应用交错并展平 # 应用交错排列并展平
    zp = zp.reshape((-1, size_n)).contiguous()  # 重塑为2D并确保连续 # 重塑为2D形状并确保内存连续
    zp = pack_cols(zp, num_bits, size_k, size_n)  # 打包列 # 按列打包为零点数据

    return zp  # 返回处理后的零点 # 返回Marlin格式的零点张量


def awq_to_marlin_zero_points(  # 将AWQ零点转换为Marlin格式 # 将AWQ格式的零点转换为Marlin格式
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int  # 打包的AWQ零点及维度参数 # AWQ打包零点及维度和位宽参数
) -> torch.Tensor:  # 返回Marlin格式零点 # 返回Marlin格式的零点张量
    # AWQ zero-points are quantized and packed on the column dim. # AWQ零点在列维度上量化和打包
    # In addition, the values are permuted based on dequantizer. # 此外，值根据反量化器进行了排列
    # Here we undo both of these, and then apply marlin permutation # 这里我们撤销这两个操作，然后应用Marlin排列
    # and pack it back. # 并重新打包
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)  # 解包AWQ零点 # 将AWQ打包的零点解包

    # Undo interleaving (use argsort(..) to get inverse perm) # 撤销交错（使用argsort获取逆排列）
    if num_bits == 4:  # 如果是4位量化 # 4位量化的撤销交错
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))  # 计算4位逆交错 # 获取4位交错的逆排列
    elif num_bits == 8:  # 如果是8位量化 # 8位量化的撤销交错
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))  # 计算8位逆交错 # 获取8位交错的逆排列
    else:  # 其他位数 # 不支持的位宽
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))  # 抛出异常 # 抛出不支持的位宽异常

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()  # 撤销交错并展平 # 应用逆交错排列并展平
    q_zp = q_zp.reshape((-1, size_n)).contiguous()  # 重塑为2D并确保连续 # 重塑为2D形状并确保内存连续

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)  # 转换为Marlin零点 # 调用Marlin零点处理函数
    return marlin_zp  # 返回Marlin格式零点 # 返回Marlin格式的零点张量


def moe_awq_to_marlin_zero_points(  # 将MoE的AWQ零点转换为Marlin格式 # 将MoE专家的AWQ格式零点批量转换为Marlin格式
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int  # 打包的AWQ零点及维度参数 # AWQ打包零点及维度和位宽参数
):  # 返回Marlin格式零点 # 返回Marlin格式的零点张量
    num_experts = q_zp_packed.shape[0]  # 获取专家数量 # 获取MoE的专家数量
    output = torch.empty(  # 创建输出张量 # 创建空输出张量
        (num_experts, q_zp_packed.shape[1], q_zp_packed.shape[2]),  # 形状 # 指定输出形状
        device=q_zp_packed.device,  # 设备 # 使用相同设备
        dtype=q_zp_packed.dtype,  # 数据类型 # 使用相同数据类型
    )
    for e in range(num_experts):  # 遍历每个专家 # 逐个处理每个专家
        output[e] = awq_to_marlin_zero_points(q_zp_packed[e], size_k, size_n, num_bits)  # 转换每个专家的零点 # 对当前专家的AWQ零点进行Marlin转换
    return output  # 返回转换结果 # 返回转换后的零点张量


def maybe_warn_marlin_atomic_add(device, dtype):  # 可能警告Marlin原子加法 # 在SM90之前的GPU上使用bf16时发出性能警告
    if torch.compiler.is_dynamo_compiling():  # 如果正在Dynamo编译 # 如果处于torch.compile编译过程中
        return  # 直接返回 # 跳过警告
    device_capability = torch.cuda.get_device_capability(device)  # 获取设备计算能力 # 获取GPU的计算能力版本
    if device_capability[0] < 9 and dtype == torch.bfloat16:  # SM90以下且为bf16 # 如果GPU版本低于SM90且数据类型为bfloat16
        logger.info_once(  # 记录一次性信息 # 输出一次性日志警告
            "You are running Marlin kernel with bf16 on GPUs before SM90. "  # 在SM90之前的GPU上使用bf16运行Marlin核函数 # 警告：在SM90之前的GPU上以bf16运行Marlin核函数
            "You can consider change to fp16 to achieve better performance "  # 可以考虑改为fp16以获得更好性能 # 建议：考虑改用fp16以获得更好性能
            "if possible."  # 如果可能的话 # 如果可能的话
        )


def maybe_warn_marlin_atomic_add_env():  # 可能警告Marlin原子加法环境变量 # 提示用户可以通过环境变量启用atomic_add特性
    if torch.compiler.is_dynamo_compiling():  # 如果正在Dynamo编译 # 如果处于torch.compile编译过程中
        return  # 直接返回 # 跳过警告
    # TODO(yiyun): Need to add sglang's MARLIN_USE_ATOMIC_ADD: bool = False # 待办(yiyun)：需要添加sglang的MARLIN_USE_ATOMIC_ADD: bool = False
    if True:  # 当前默认跳过 # 当前始终返回（未启用环境变量检查）
        return  # 直接返回 # 跳过警告
    # if envs.VLLM_MARLIN_USE_ATOMIC_ADD: # 如果设置了VLLM_MARLIN_USE_ATOMIC_ADD
    #     return # 返回
    logger.info_once(  # 记录一次性信息 # 输出一次性日志提示
        "Marlin kernel can achieve better performance for small size_n "  # Marlin核函数对于小size_n可以获得更好性能 # 提示：Marlin核函数在小输出维度下可获得更好性能
        "with experimental use_atomic_add feature. "  # 通过实验性的use_atomic_add特性 # 通过实验性的atomic_add特性
        "You can consider set environment variable "  # 可以考虑设置环境变量 # 建议：考虑设置环境变量
        "VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible."  # VLLM_MARLIN_USE_ATOMIC_ADD为1 # VLLM_MARLIN_USE_ATOMIC_ADD为1
    )


def should_use_atomic_add_reduce(  # 判断是否应使用原子加法归约 # 判断Marlin矩阵乘法是否应使用atomicAdd而非全局归约
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype  # 维度参数、设备和数据类型 # M/N/K维度、设备和数据类型
) -> bool:  # 返回是否使用原子加法归约 # 返回布尔值

    # the performance of atomicAdd is better than global reduce # atomicAdd的性能优于全局归约
    # only when m*n is small and k is large # 仅当m*n较小且k较大时
    if n >= 2048 or k < 2048 or device.type != "cuda":  # 条件不满足时不使用 # 当n过大、k过小或非CUDA设备时不使用atomicAdd
        return False  # 不使用原子加法 # 返回False

    # disable atomicAdd reduce by default, # 默认禁用atomicAdd归约，
    # one can enable it with VLLM_MARLIN_USE_ATOMIC_ADD=1 # 可以通过VLLM_MARLIN_USE_ATOMIC_ADD=1启用
    # TODO: Need to add sglang's MARLIN_USE_ATOMIC_ADD: bool = False # 待办：需要添加sglang的MARLIN_USE_ATOMIC_ADD: bool = False
    if not True:  # 当前默认禁用 # 当前始终为禁用状态
        maybe_warn_marlin_atomic_add_env()  # 可能发出环境变量警告 # 提示用户可以启用atomic_add
        return False  # 不使用原子加法 # 返回False

    # sm8x doesn't support atomicAdd + bfloat16 natively # SM8x不原生支持atomicAdd + bfloat16
    device_capability = torch.cuda.get_device_capability(device)  # 获取设备计算能力 # 获取GPU的计算能力版本
    if device_capability[0] < 9 and dtype == torch.bfloat16:  # SM90以下且为bf16 # 如果GPU版本低于SM90且数据类型为bfloat16
        maybe_warn_marlin_atomic_add(device, dtype)  # 发出警告 # 发出性能警告
        return False  # 不使用原子加法 # 返回False

    return True  # 使用原子加法 # 返回True，允许使用atomicAdd


def apply_gptq_marlin_linear(  # 应用GPTQ Marlin线性层 # 应用GPTQ Marlin量化的线性层计算
    input: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 权重张量 # 量化权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子 # 权重缩放因子
    weight_zp: torch.Tensor,  # 权重零点 # 权重零点
    g_idx: torch.Tensor,  # 组索引 # 量化组索引
    g_idx_sort_indices: torch.Tensor,  # 组索引排序下标 # 组索引的排序下标
    workspace: torch.Tensor,  # 工作空间 # Marlin核函数工作空间
    wtype: ScalarType,  # 权重类型 # 量化的标量权重类型
    output_size_per_partition: int,  # 分区输出大小 # 分区后的输出维度大小
    input_size_per_partition: int,  # 分区输入大小 # 分区后的输入维度大小
    is_k_full: bool,  # K维度是否完整 # K维度是否完整
    bias: Optional[torch.Tensor] = None,  # 偏置（可选） # 偏置项，默认为None
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,  # 是否使用FP32归约 # 是否使用FP32精度归约，默认True
) -> torch.Tensor:  # 返回输出张量 # 返回计算结果张量
    reshaped_x = input.reshape(-1, input.shape[-1])  # 将输入重塑为2D # 将输入张量重塑为2D矩阵
    out_shape = input.shape[:-1] + (output_size_per_partition,)  # 计算输出形状 # 计算输出张量的目标形状

    use_atomic_add = should_use_atomic_add_reduce(  # 判断是否使用原子加法归约 # 检查是否应使用atomicAdd归约方式
        m=reshaped_x.size(0),  # M维度 # 传入批次数
        n=output_size_per_partition,  # N维度 # 传入输出维度
        k=reshaped_x.size(1),  # K维度 # 传入输入维度
        device=input.device,  # 设备 # 传入设备
        dtype=input.dtype,  # 数据类型 # 传入数据类型
    )

    forward_context = get_forward_context()  # 获取前向上下文 # 获取当前前向计算的上下文信息
    if forward_context is None:  # 如果没有前向上下文 # 如果不处于编译追踪环境中
        output = gptq_marlin_gemm(  # 直接调用GPTQ Marlin GEMM # 调用JIT版本的GPTQ Marlin矩阵乘法
            reshaped_x,  # 输入 # 重塑后的输入
            None,  # 速度为空 # 速度参数为空
            weight,  # 权重 # 量化权重
            weight_scale,  # 权重缩放因子 # 权重缩放因子
            None,  # 速度为空 # 速度参数为空
            weight_zp,  # 权重零点 # 权重零点
            g_idx,  # 组索引 # 量化组索引
            g_idx_sort_indices,  # 组索引排序下标 # 组索引排序下标
            workspace,  # 工作空间 # Marlin工作空间
            wtype,  # 权重类型 # 标量权重类型
            size_m=reshaped_x.shape[0],  # M维度大小 # 批次数
            size_n=output_size_per_partition,  # N维度大小 # 输出维度
            size_k=input_size_per_partition,  # K维度大小 # 输入维度
            is_k_full=is_k_full,  # K是否完整 # K维度完整性标志
            use_atomic_add=use_atomic_add,  # 是否使用原子加法 # 是否使用atomicAdd
            use_fp32_reduce=use_fp32_reduce,  # 是否使用FP32归约 # 是否使用FP32精度归约
            is_zp_float=False,  # 零点不是浮点 # 零点不是浮点类型
        )
    else:  # 有前向上下文（编译追踪模式） # 在编译追踪环境中
        output = unified_apply_gptq_marlin_gemm_with_wtype(  # 调用统一GEMM（带wtype） # 调用自定义算子版本的GPTQ Marlin矩阵乘法
            input=reshaped_x,  # 输入 # 重塑后的输入
            weight=weight,  # 权重 # 量化权重
            weight_scale=weight_scale,  # 权重缩放因子 # 权重缩放因子
            weight_zp=weight_zp,  # 权重零点 # 权重零点
            g_idx=g_idx,  # 组索引 # 量化组索引
            g_idx_sort_indices=g_idx_sort_indices,  # 组索引排序下标 # 组索引排序下标
            workspace=workspace,  # 工作空间 # Marlin工作空间
            wtype_id=wtype.id,  # 权重类型ID # 标量权重类型的ID
            output_size_per_partition=output_size_per_partition,  # 分区输出大小 # 输出维度
            input_size_per_partition=input_size_per_partition,  # 分区输入大小 # 输入维度
            is_k_full=is_k_full,  # K是否完整 # K维度完整性标志
            use_atomic_add=use_atomic_add,  # 是否使用原子加法 # 是否使用atomicAdd
            use_fp32_reduce=use_fp32_reduce,  # 是否使用FP32归约 # 是否使用FP32精度归约
            is_zp_float=False,  # 零点不是浮点 # 零点不是浮点类型
        )

    if bias is not None:  # 如果存在偏置 # 如果偏置不为空
        output.add_(bias)  # In-place add # 原地加偏置 # 原地将偏置加到输出上

    return output.reshape(out_shape)  # 重塑输出形状后返回 # 将输出重塑为目标形状后返回


def apply_awq_marlin_linear(  # 应用AWQ Marlin线性层 # 应用AWQ Marlin量化的线性层计算
    input: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 权重张量 # 量化权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子 # 权重缩放因子
    weight_zp: torch.Tensor,  # 权重零点 # 权重零点
    g_idx: torch.Tensor,  # 组索引 # 量化组索引
    g_idx_sort_indices: torch.Tensor,  # 组索引排序下标 # 组索引的排序下标
    workspace: torch.Tensor,  # 工作空间 # Marlin核函数工作空间
    quant_type: ScalarType,  # 量化类型 # 量化的标量类型
    output_size_per_partition: int,  # 分区输出大小 # 分区后的输出维度大小
    input_size_per_partition: int,  # 分区输入大小 # 分区后的输入维度大小
    bias: Optional[torch.Tensor] = None,  # 偏置（可选） # 偏置项，默认为None
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,  # 是否使用FP32归约 # 是否使用FP32精度归约，默认True
) -> torch.Tensor:  # 返回输出张量 # 返回计算结果张量
    reshaped_x = input.reshape(-1, input.shape[-1])  # 将输入重塑为2D # 将输入张量重塑为2D矩阵
    out_shape = input.shape[:-1] + (output_size_per_partition,)  # 计算输出形状 # 计算输出张量的目标形状

    use_atomic_add = should_use_atomic_add_reduce(  # 判断是否使用原子加法归约 # 检查是否应使用atomicAdd归约方式
        m=reshaped_x.size(0),  # M维度 # 传入批次数
        n=output_size_per_partition,  # N维度 # 传入输出维度
        k=reshaped_x.size(1),  # K维度 # 传入输入维度
        device=input.device,  # 设备 # 传入设备
        dtype=input.dtype,  # 数据类型 # 传入数据类型
    )

    forward_context = get_forward_context()  # 获取前向上下文 # 获取当前前向计算的上下文信息
    if forward_context is None:  # 如果没有前向上下文 # 如果不处于编译追踪环境中
        output = gptq_marlin_gemm(  # 直接调用GPTQ Marlin GEMM # 调用JIT版本的GPTQ Marlin矩阵乘法
            reshaped_x,  # 输入 # 重塑后的输入
            None,  # 速度为空 # 速度参数为空
            weight,  # 权重 # 量化权重
            weight_scale,  # 权重缩放因子 # 权重缩放因子
            None,  # 速度为空 # 速度参数为空
            weight_zp,  # 权重零点 # 权重零点
            g_idx,  # 组索引 # 量化组索引
            g_idx_sort_indices,  # 组索引排序下标 # 组索引排序下标
            workspace,  # 工作空间 # Marlin工作空间
            quant_type,  # 量化类型 # 标量量化类型
            size_m=reshaped_x.shape[0],  # M维度大小 # 批次数
            size_n=output_size_per_partition,  # N维度大小 # 输出维度
            size_k=input_size_per_partition,  # K维度大小 # 输入维度
            use_atomic_add=use_atomic_add,  # 是否使用原子加法 # 是否使用atomicAdd
            use_fp32_reduce=use_fp32_reduce,  # 是否使用FP32归约 # 是否使用FP32精度归约
            is_zp_float=False,  # 零点不是浮点 # 零点不是浮点类型
        )
    else:  # 有前向上下文（编译追踪模式） # 在编译追踪环境中
        output = unified_apply_gptq_marlin_gemm(  # 调用统一GEMM # 调用自定义算子版本的GPTQ Marlin矩阵乘法
            input=reshaped_x,  # 输入 # 重塑后的输入
            weight=weight,  # 权重 # 量化权重
            weight_scale=weight_scale,  # 权重缩放因子 # 权重缩放因子
            weight_zp=weight_zp,  # 权重零点 # 权重零点
            g_idx=g_idx,  # 组索引 # 量化组索引
            g_idx_sort_indices=g_idx_sort_indices,  # 组索引排序下标 # 组索引排序下标
            workspace=workspace,  # 工作空间 # Marlin工作空间
            output_size_per_partition=output_size_per_partition,  # 分区输出大小 # 输出维度
            input_size_per_partition=input_size_per_partition,  # 分区输入大小 # 输入维度
            use_atomic_add=use_atomic_add,  # 是否使用原子加法 # 是否使用atomicAdd
            use_fp32_reduce=use_fp32_reduce,  # 是否使用FP32归约 # 是否使用FP32精度归约
            is_zp_float=False,  # 零点不是浮点 # 零点不是浮点类型
        )

    if bias is not None:  # 如果存在偏置 # 如果偏置不为空
        output.add_(bias)  # In-place add # 原地加偏置 # 原地将偏置加到输出上

    return output.reshape(out_shape)  # 重塑输出形状后返回 # 将输出重塑为目标形状后返回


class MarlinConfig(QuantizationConfig):  # Marlin量化配置类 # Marlin量化方法的配置类
    """Config class for Marlin. # Marlin的配置类

    Reference: https://github.com/IST-DASLab/marlin/tree/master # 参考：Marlin项目仓库
    """

    def __init__(  # 初始化方法 # Marlin配置初始化
        self,
        group_size: int,  # 组大小 # 量化分组大小
        lm_head_quantized: bool,  # 语言模型头是否量化 # 语言模型输出头是否也进行量化
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类QuantizationConfig的初始化

        # Group size for the quantization. # 量化的分组大小
        self.group_size = group_size  # 保存组大小 # 保存量化分组大小
        self.lm_head_quantized = lm_head_quantized  # 保存LM头量化标志 # 保存语言模型头是否量化的标志
        if self.group_size != 128 and self.group_size != -1:  # 验证组大小 # 检查分组大小是否合法
            raise ValueError(  # 抛出值错误 # 抛出异常
                "Currently, only group size 128 and -1 (channelwise) "  # 目前仅支持组大小128和-1（通道级） # 错误信息：仅支持128和-1
                "is supported for Marlin, but got group_size of "  # 被Marlin支持，但得到的组大小为 # 错误信息后缀
                f"{self.group_size}"  # 实际组大小 # 错误信息：实际的分组大小
            )

        # 4 Bits packed into 32 bit datatype. # 4位打包到32位数据类型中
        self.pack_factor = 32 // 4  # 打包因子 = 8 # 计算打包因子（32位/4位=8）

        # Tile size used by marlin kernels. # Marlin核函数使用的瓦片大小
        self.tile_size = 16  # 瓦片大小 # 瓦片大小为16

        # Min out_features dim # 最小输出特征维度
        self.min_n_threads = 64  # 最小N线程数 # N维最小线程数为64

        # Min in_features dim # 最小输入特征维度
        self.min_k_threads = 128  # 最小K线程数 # K维最小线程数为128

        # Max parallel problems to solve at once (improves large # 一次求解的最大并行问题数（改善大
        # batch performance) # 批次性能）
        self.max_parallel = 16  # 最大并行数 # 最大并行问题数为16

        # Permutation length used by the marlin kernels. # Marlin核函数使用的排列长度
        self.perm_len = 1024  # 排列长度 # 排列长度为1024

    def __repr__(self) -> str:  # 字符串表示 # 返回配置的字符串表示
        return (  # 返回格式化字符串 # 返回格式化字符串
            f"MarlinConfig(group_size={self.group_size}, "  # 组大小 # 分组大小
            f"lm_head_quantized={self.lm_head_quantized})"  # LM头量化标志 # 语言模型头量化标志
        )

    @classmethod  # 类方法 # 声明为类方法
    def get_name(cls) -> str:  # 获取量化方法名称 # 返回Marlin量化方法的名称
        return "marlin"  # 返回名称 # 返回"marlin"

    @classmethod  # 类方法 # 声明为类方法
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:  # 获取支持的激活数据类型 # 返回支持的激活值数据类型列表
        return [torch.half]  # 仅支持float16 # 仅支持半精度浮点数

    @classmethod  # 类方法 # 声明为类方法
    # Need to figure it out # 需要确认
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力 # 返回所需的最低GPU计算能力版本
        return 80  # 最低SM80 # 最低要求SM80架构

    @classmethod  # 类方法 # 声明为类方法
    def get_config_filenames(cls) -> list[str]:  # 获取配置文件名 # 返回量化配置文件名列表
        return ["quantize_config.json"]  # 配置文件名 # 返回量化配置JSON文件名

    @classmethod  # 类方法 # 声明为类方法
    def from_config(cls, config: dict[str, Any]) -> "MarlinConfig":  # 从配置字典创建实例 # 从配置字典创建MarlinConfig实例
        group_size = cls.get_from_keys(config, ["group_size"])  # 获取组大小 # 从配置中读取分组大小
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)  # 获取LM头量化标志 # 从配置中读取LM头量化标志，默认为False
        return cls(group_size, lm_head_quantized)  # 创建并返回实例 # 创建并返回MarlinConfig实例

    @classmethod  # 类方法 # 声明为类方法
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:  # 覆盖量化方法 # 根据检查点格式决定是否覆盖量化方法
        # compat: autogptq >=0.8.0 use checkpoint_format: str # 兼容：autogptq >=0.8.0使用checkpoint_format: 字符串
        # compat: autogptq <=0.7.1 is_marlin_format: bool # 兼容：autogptq <=0.7.1使用is_marlin_format: 布尔
        is_marlin_format = hf_quant_cfg.get(  # 检查是否为Marlin格式 # 检查配置中是否标记为Marlin格式
            "checkpoint_format"  # 检查点格式键 # 读取checkpoint_format键
        ) == "marlin" or hf_quant_cfg.get("is_marlin_format", False)  # 或is_marlin_format标志 # 或者is_marlin_format键为True

        is_valid_user_quant = (  # 检查用户量化选项是否有效 # 验证用户指定的量化方法是否兼容
            user_quant is None or user_quant == "gptq" or user_quant == "marlin"  # None、gptq或marlin # 用户未指定或指定为gptq/marlin
        )

        if is_marlin_format and is_valid_user_quant:  # 如果是Marlin格式且用户量化选项有效 # 满足条件时自动覆盖
            msg = "The model is serialized in {} format. Using {} kernel.".format(  # 模型以指定格式序列化，使用指定核函数 # 格式化提示信息
                cls.get_name(), cls.get_name()  # 量化方法名称 # 使用当前类的方法名
            )
            logger.info(msg)  # 记录信息 # 输出日志信息
            return cls.get_name()  # 返回方法名 # 返回"marlin"表示使用Marlin方法

        return None  # 不覆盖 # 返回None表示不覆盖


    def get_quant_method(  # 获取量化方法 # 获取给定层的量化方法实例
        self, layer: torch.nn.Module, prefix: str  # 层和前缀 # 目标层及参数名前缀
    ) -> Optional[MarlinLinearMethod]:  # 返回量化方法或None # 返回MarlinLinearMethod实例或None
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类 # 导入线性层基类
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行LM头 # 导入词表并行嵌入层

        if isinstance(layer, LinearBase) or (  # 如果是线性层或 # 检查是否为线性层
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized  # LM头且启用了量化 # 或者是LM头且配置了量化
        ):
            return MarlinLinearMethod(self)  # 返回Marlin线性方法 # 创建并返回MarlinLinearMethod实例
        return None  # 不支持该层 # 返回None表示不支持


class MarlinLinearMethod(LinearMethodBase):  # Marlin线性方法类 # Marlin量化线性层的方法实现
    """Linear method for Marlin. # Marlin的线性方法

    Args: # 参数：
        quant_config: The Marlin quantization config. # quant_config：Marlin量化配置
    """

    def __init__(self, quant_config: MarlinConfig):  # 初始化方法 # 使用Marlin配置初始化线性方法
        self.quant_config = quant_config  # 保存量化配置 # 保存Marlin配置实例

    def create_weights(  # 创建权重 # 为线性层创建量化权重参数
        self,
        layer: torch.nn.Module,  # 目标层 # 目标线性层
        input_size_per_partition: int,  # 分区输入大小 # 分区后的输入维度大小
        output_partition_sizes: list[int],  # 输出分区大小列表 # 各输出分区的维度大小列表
        input_size: int,  # 总输入大小 # 完整的输入维度大小
        output_size: int,  # 总输出大小 # 完整的输出维度大小
        params_dtype: torch.dtype,  # 参数数据类型 # 参数的数据类型
        **extra_weight_attrs,  # 额外权重属性 # 其他权重属性
    ):
        del output_size  # Unused. # 未使用 # 删除未使用的output_size参数
        weight_loader = extra_weight_attrs["weight_loader"]  # 获取权重加载器 # 获取自定义权重加载函数

        if params_dtype != torch.float16:  # 检查参数类型必须为float16 # 验证参数数据类型必须为float16
            raise ValueError(  # 抛出值错误 # 抛出异常
                f"The params dtype must be float16, but got {params_dtype}"  # 参数类型必须为float16 # 错误信息：实际数据类型
            )

        # Validate output_size_per_partition # 验证output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)  # 计算总分区输出大小 # 对所有输出分区大小求和
        if output_size_per_partition % self.quant_config.min_n_threads != 0:  # 检查是否能被最小N线程整除 # 验证输出维度是否满足最小线程要求
            raise ValueError(  # 抛出值错误 # 抛出异常
                f"Weight output_size_per_partition = "  # 权重输出大小 # 错误信息前缀
                f"{output_size_per_partition} is not divisible by "  # 不能被整除 # 错误信息：不能被整除
                f"min_n_threads = {self.quant_config.min_n_threads}."  # 最小N线程 # 错误信息：最小N线程值
            )
        if output_size_per_partition % self.quant_config.pack_factor != 0:  # 检查是否能被打包因子整除 # 验证输出维度是否满足打包要求
            raise ValueError(  # 抛出值错误 # 抛出异常
                f"Weight output_size_per_partition = "  # 权重输出大小 # 错误信息前缀
                f"{output_size_per_partition} is not divisible by "  # 不能被整除 # 错误信息：不能被整除
                f"pack_factor = {self.quant_config.pack_factor}."  # 打包因子 # 错误信息：打包因子值
            )

        # Validate input_size_per_partition # 验证input_size_per_partition
        if input_size_per_partition % self.quant_config.min_k_threads != 0:  # 检查是否能被最小K线程整除 # 验证输入维度是否满足最小线程要求
            raise ValueError(  # 抛出值错误 # 抛出异常
                f"Weight input_size_per_partition = "  # 权重输入大小 # 错误信息前缀
                f"{input_size_per_partition} is not divisible by "  # 不能被整除 # 错误信息：不能被整除
                f"min_k_threads = {self.quant_config.min_k_threads}."  # 最小K线程 # 错误信息：最小K线程值
            )
        if (  # 如果 # 条件检查
            self.quant_config.group_size != -1  # 非通道级量化 # 分组大小不是-1（即非通道级量化）
            and input_size_per_partition % self.quant_config.group_size != 0  # 分区输入不能被组大小整除 # 输入维度不能被分组大小整除
        ):
            raise ValueError(  # 抛出值错误 # 抛出异常
                f"Weight input_size_per_partition = "  # 权重输入大小 # 错误信息前缀
                f"{input_size_per_partition} is not divisible by "  # 不能被整除 # 错误信息：不能被整除
                f"group_size = {self.quant_config.group_size}."  # 组大小 # 错误信息：分组大小值
            )

        # Check that we have at least 4 tiles horizontally in the shard # 检查分片中水平方向至少有4个瓦片
        num_tiles_per_perm = self.quant_config.perm_len // (  # 每个排列中的瓦片数 # 计算每个排列组包含的瓦片数
            self.quant_config.tile_size**2  # 瓦片大小的平方 # 瓦片面积
        )
        if output_size_per_partition % num_tiles_per_perm != 0:  # 输出维度必须能被瓦片组整除 # 验证输出维度是否满足排列组要求
            raise ValueError("Each permutation group must reside on the same gpu")  # 每个排列组必须在同一GPU上 # 抛出异常：排列组不能跨GPU

        # Quantized 4Bit weights packed into Int32. # 量化后的4位权重打包到Int32中
        qweight = PackedvLLMParameter(  # 创建打包权重参数 # 创建打包量化的权重参数
            data=torch.empty(  # 创建空数据 # 分配空张量
                input_size_per_partition // self.quant_config.tile_size,  # 行数 # 输入维度/瓦片大小
                output_size_per_partition  # 列数 # 输出维度
                * self.quant_config.tile_size  # 乘以瓦片大小 # 乘以瓦片大小
                // self.quant_config.pack_factor,  # 除以打包因子 # 除以打包因子
                device="cuda",  # CUDA设备 # 指定CUDA设备
                dtype=torch.int32,  # int32类型 # 使用int32存储打包权重
            ),
            input_dim=0,  # 输入维度索引 # 输入维度索引为0
            output_dim=1,  # 输出维度索引 # 输出维度索引为1
            packed_dim=1,  # 打包维度索引 # 打包维度为1（列方向打包）
            packed_factor=self.quant_config.pack_factor,  # 打包因子 # 打包因子
            marlin_tile_size=self.quant_config.tile_size,  # Marlin瓦片大小 # Marlin瓦片大小
            weight_loader=weight_loader,  # 权重加载器 # 自定义权重加载函数
        )

        # Determine if channelwise or not # 确定是否为通道级量化
        input_groups = (  # 计算输入分组数 # 计算量化的分组数量
            1  # 通道级时为1 # 通道级量化时只有1组
            if self.quant_config.group_size == -1  # 如果组大小为-1 # 通道级量化（group_size=-1）
            else input_size_per_partition // self.quant_config.group_size  # 否则按组大小计算 # 否则计算分组数量
        )

        weight_scale_args = {  # 权重缩放因子参数 # 缩放因子的初始化参数
            "data": torch.empty(  # 创建空数据 # 分配空张量
                input_groups,  # 分组数 # 分组维度
                output_size_per_partition,  # 输出维度 # 输出维度大小
                device="cuda",  # CUDA设备 # 指定CUDA设备
                dtype=params_dtype,  # 参数数据类型 # 使用指定的参数数据类型
            ),
            "weight_loader": weight_loader,  # 权重加载器 # 自定义权重加载函数
        }
        if input_groups == 1:  # 通道级量化 # 如果是通道级量化
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)  # 创建通道级缩放参数 # 创建通道级缩放因子参数
        else:  # 组级量化 # 如果是组级量化
            scales = GroupQuantScaleParameter(  # 创建组级缩放参数 # 创建组级缩放因子参数
                output_dim=1, input_dim=0, **weight_scale_args  # 指定输出和输入维度索引 # 指定维度索引
            )

        # Allocate workspace (Used for internal locking mechanism) # 分配工作空间（用于内部锁定机制）
        max_workspace_size = (  # 计算最大工作空间大小 # 计算最大工作空间尺寸
            output_size_per_partition // self.quant_config.min_n_threads  # 输出维度/最小N线程 # 输出维度除以最小N线程数
        ) * self.quant_config.max_parallel  # 乘以最大并行数 # 乘以最大并行数

        workspace = BasevLLMParameter(  # 创建工作空间参数 # 创建基础工作空间参数
            data=torch.zeros(max_workspace_size, device="cuda", dtype=torch.int),  # 零初始化int张量 # 零初始化的int类型张量
            weight_loader=weight_loader,  # 权重加载器 # 自定义权重加载函数
        )

        layer.register_parameter("B", qweight)  # 注册量化权重 # 将打包权重注册为层的B参数
        layer.register_parameter("s", scales)  # 注册缩放因子 # 将缩放因子注册为层的s参数
        layer.register_parameter("workspace", workspace)  # 注册工作空间 # 将工作空间注册为层的workspace参数

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理 # 权重加载完成后进行后处理
        # required by torch.compile # torch.compile所需
        layer.B = torch.nn.Parameter(layer.B.data, requires_grad=False)  # 将B转为不需要梯度的参数 # 将量化权重转为不需要梯度的参数
        layer.s = torch.nn.Parameter(layer.s.data, requires_grad=False)  # 将s转为不需要梯度的参数 # 将缩放因子转为不需要梯度的参数
        layer.workspace = torch.nn.Parameter(layer.workspace.data, requires_grad=False)  # 将workspace转为不需要梯度的参数 # 将工作空间转为不需要梯度的参数

    def apply(  # 应用线性层计算 # 执行Marlin量化的线性层前向计算
        self,
        layer: torch.nn.Module,  # 目标层 # 目标线性层
        x: torch.Tensor,  # 输入张量 # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置（可选） # 偏置项，默认为None
    ) -> torch.Tensor:  # 返回输出张量 # 返回计算结果张量
        qweight = layer.B  # 获取量化权重 # 获取层的打包量化权重
        scales = layer.s  # 获取缩放因子 # 获取层的缩放因子
        workspace = layer.workspace  # 获取工作空间 # 获取层的工作空间

        x_2d = x.view(-1, x.shape[-1])  # 将输入重塑为2D # 将输入重塑为2D矩阵

        size_m = x_2d.shape[0]  # M维度大小 # 批次数
        size_k = x_2d.shape[1]  # K维度大小 # 输入维度
        size_n = scales.shape[1]  # N维度大小 # 输出维度

        output_2d = ops.marlin_gemm(  # 调用Marlin GEMM核函数 # 调用vLLM的Marlin矩阵乘法核函数
            x_2d, qweight, scales, workspace, size_m, size_n, size_k  # 传入所有参数 # 传入输入、权重、缩放因子、工作空间和维度信息
        )

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1],))  # 重塑输出形状 # 将2D输出重塑为与输入匹配的形状

        if bias is not None:  # 如果存在偏置 # 如果偏置不为空
            output.add_(bias)  # In-place add # 原地加偏置 # 原地将偏置加到输出上

        return output  # 返回输出 # 返回计算结果


def fake_unified_apply_gptq_marlin_gemm(  # 统一GPTQ Marlin GEMM的伪实现 # 用于torch.compile追踪的伪实现
    input: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 权重张量 # 量化权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子 # 权重缩放因子
    weight_zp: torch.Tensor,  # 权重零点 # 权重零点
    g_idx: torch.Tensor,  # 组索引 # 量化组索引
    g_idx_sort_indices: torch.Tensor,  # 组索引排序下标 # 组索引排序下标
    workspace: torch.Tensor,  # 工作空间 # Marlin工作空间
    output_size_per_partition: int,  # 分区输出大小 # 输出维度
    input_size_per_partition: int,  # 分区输入大小 # 输入维度
    use_atomic_add: bool,  # 是否使用原子加法 # 是否使用atomicAdd
    use_fp32_reduce: bool,  # 是否使用FP32归约 # 是否使用FP32精度归约
    is_zp_float: bool,  # 零点是否为浮点 # 零点是否为浮点类型
) -> torch.Tensor:  # 返回空张量 # 返回与输入同类型的空张量
    return input.new_empty(  # 返回空张量 # 创建空张量
        (input.shape[0], output_size_per_partition), dtype=input.dtype  # 形状和类型 # 指定形状和与输入相同的数据类型
    )


@register_custom_op(fake_impl=fake_unified_apply_gptq_marlin_gemm)  # 注册自定义算子 # 注册为torch.compile可识别的自定义算子
def unified_apply_gptq_marlin_gemm(  # 统一GPTQ Marlin GEMM算子 # 统一的GPTQ Marlin矩阵乘法自定义算子
    input: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 权重张量 # 量化权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子 # 权重缩放因子
    weight_zp: torch.Tensor,  # 权重零点 # 权重零点
    g_idx: torch.Tensor,  # 组索引 # 量化组索引
    g_idx_sort_indices: torch.Tensor,  # 组索引排序下标 # 组索引排序下标
    workspace: torch.Tensor,  # 工作空间 # Marlin工作空间
    output_size_per_partition: int,  # 分区输出大小 # 输出维度
    input_size_per_partition: int,  # 分区输入大小 # 输入维度
    use_atomic_add: bool,  # 是否使用原子加法 # 是否使用atomicAdd
    use_fp32_reduce: bool,  # 是否使用FP32归约 # 是否使用FP32精度归约
    is_zp_float: bool,  # 零点是否为浮点 # 零点是否为浮点类型
) -> torch.Tensor:  # 返回输出张量 # 返回计算结果张量
    quant_config = get_forward_context().quant_config  # 获取量化配置 # 从前向上下文中获取量化配置
    quant_type = quant_config.quant_type  # 获取量化类型 # 获取标量量化类型
    return gptq_marlin_gemm(  # 调用GPTQ Marlin GEMM # 调用JIT版本的GPTQ Marlin矩阵乘法
        input,  # 输入 # 输入张量
        None,  # 速度为空 # 速度参数为空
        weight,  # 权重 # 量化权重
        weight_scale,  # 权重缩放因子 # 权重缩放因子
        None,  # 速度为空 # 速度参数为空
        weight_zp,  # 权重零点 # 权重零点
        g_idx,  # 组索引 # 量化组索引
        g_idx_sort_indices,  # 组索引排序下标 # 组索引排序下标
        workspace,  # 工作空间 # Marlin工作空间
        quant_type,  # 量化类型 # 标量量化类型
        size_m=input.shape[0],  # M维度大小 # 批次数
        size_n=output_size_per_partition,  # N维度大小 # 输出维度
        size_k=input_size_per_partition,  # K维度大小 # 输入维度
        use_atomic_add=use_atomic_add,  # 是否使用原子加法 # 是否使用atomicAdd
        use_fp32_reduce=use_fp32_reduce,  # 是否使用FP32归约 # 是否使用FP32精度归约
        is_zp_float=is_zp_float,  # 零点是否为浮点 # 零点是否为浮点类型
    )


def fake_unified_apply_gptq_marlin_gemm_with_wtype(  # 带wtype的统一GPTQ Marlin GEMM伪实现 # 用于torch.compile追踪的伪实现（带权重类型ID）
    input: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 权重张量 # 量化权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子 # 权重缩放因子
    weight_zp: torch.Tensor,  # 权重零点 # 权重零点
    g_idx: torch.Tensor,  # 组索引 # 量化组索引
    g_idx_sort_indices: torch.Tensor,  # 组索引排序下标 # 组索引排序下标
    workspace: torch.Tensor,  # 工作空间 # Marlin工作空间
    wtype_id: int,  # 权重类型ID # 标量权重类型的ID
    output_size_per_partition: int,  # 分区输出大小 # 输出维度
    input_size_per_partition: int,  # 分区输入大小 # 输入维度
    is_k_full: bool,  # K是否完整 # K维度完整性标志
    use_atomic_add: bool,  # 是否使用原子加法 # 是否使用atomicAdd
    use_fp32_reduce: bool,  # 是否使用FP32归约 # 是否使用FP32精度归约
    is_zp_float: bool,  # 零点是否为浮点 # 零点是否为浮点类型
) -> torch.Tensor:  # 返回空张量 # 返回与输入同类型的空张量
    return input.new_empty(  # 返回空张量 # 创建空张量
        (input.shape[0], output_size_per_partition), dtype=input.dtype  # 形状和类型 # 指定形状和与输入相同的数据类型
    )


@register_custom_op(fake_impl=fake_unified_apply_gptq_marlin_gemm_with_wtype)  # 注册自定义算子 # 注册为torch.compile可识别的自定义算子
def unified_apply_gptq_marlin_gemm_with_wtype(  # 带wtype的统一GPTQ Marlin GEMM算子 # 统一的GPTQ Marlin矩阵乘法自定义算子（通过ID传递权重类型）
    input: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 权重张量 # 量化权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子 # 权重缩放因子
    weight_zp: torch.Tensor,  # 权重零点 # 权重零点
    g_idx: torch.Tensor,  # 组索引 # 量化组索引
    g_idx_sort_indices: torch.Tensor,  # 组索引排序下标 # 组索引排序下标
    workspace: torch.Tensor,  # 工作空间 # Marlin工作空间
    wtype_id: int,  # 权重类型ID # 标量权重类型的ID
    output_size_per_partition: int,  # 分区输出大小 # 输出维度
    input_size_per_partition: int,  # 分区输入大小 # 输入维度
    is_k_full: bool,  # K是否完整 # K维度完整性标志
    use_atomic_add: bool,  # 是否使用原子加法 # 是否使用atomicAdd
    use_fp32_reduce: bool,  # 是否使用FP32归约 # 是否使用FP32精度归约
    is_zp_float: bool,  # 零点是否为浮点 # 零点是否为浮点类型
) -> torch.Tensor:  # 返回输出张量 # 返回计算结果张量
    # Reconstruct ScalarType from id # 从ID重建ScalarType
    wtype = None  # 初始化权重类型 # 初始化权重标量类型为None
    for attr_name in dir(scalar_types):  # 遍历scalar_types的所有属性 # 遍历所有已注册的标量类型
        if not attr_name.startswith("_"):  # 跳过私有属性 # 跳过以_开头的私有属性
            st = getattr(scalar_types, attr_name)  # 获取属性值 # 获取标量类型对象
            if hasattr(st, "id") and st.id == wtype_id:  # 比较ID # 检查是否有id属性且匹配目标ID
                wtype = st  # 找到匹配的类型 # 设置为找到的标量类型
                break  # 退出循环 # 跳出循环
    return gptq_marlin_gemm(  # 调用GPTQ Marlin GEMM # 调用JIT版本的GPTQ Marlin矩阵乘法
        input,  # 输入 # 输入张量
        None,  # 速度为空 # 速度参数为空
        weight,  # 权重 # 量化权重
        weight_scale,  # 权重缩放因子 # 权重缩放因子
        None,  # 速度为空 # 速度参数为空
        weight_zp,  # 权重零点 # 权重零点
        g_idx,  # 组索引 # 量化组索引
        g_idx_sort_indices,  # 组索引排序下标 # 组索引排序下标
        workspace,  # 工作空间 # Marlin工作空间
        wtype,  # 权重类型 # 标量权重类型
        size_m=input.shape[0],  # M维度大小 # 批次数
        size_n=output_size_per_partition,  # N维度大小 # 输出维度
        size_k=input_size_per_partition,  # K维度大小 # 输入维度
        is_k_full=is_k_full,  # K是否完整 # K维度完整性标志
        use_atomic_add=use_atomic_add,  # 是否使用原子加法 # 是否使用atomicAdd
        use_fp32_reduce=use_fp32_reduce,  # 是否使用FP32归约 # 是否使用FP32精度归约
        is_zp_float=is_zp_float,  # 零点是否为浮点 # 零点是否为浮点类型
    )
