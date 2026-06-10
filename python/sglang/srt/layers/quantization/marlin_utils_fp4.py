# Marlin FP4量化工具模块：提供MXFP4格式Marlin量化的权重重排、缩放因子处理等工具函数
# 支持MoE（混合专家）模型的MXFP4层Marlin预处理

from __future__ import annotations  # 启用延迟类型注解评估 # 启用PEP 563延迟注解评估

import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架

from sglang.srt.layers.quantization.marlin_utils import (  # 从Marlin工具模块导入函数 # 从Marlin量化工具模块导入辅助函数
    marlin_make_workspace,  # 创建Marlin工作空间 # 创建Marlin核函数的工作空间张量
    marlin_permute_bias,  # 排列偏置 # 对偏置进行Marlin格式排列
    marlin_permute_scales,  # 排列缩放因子 # 对缩放因子进行Marlin格式排列
)
from sglang.srt.utils import is_cuda  # 导入CUDA检测函数 # 导入CUDA环境检测函数

_is_cuda = is_cuda()  # 检测是否为CUDA环境 # 检测当前是否在CUDA环境下运行

if _is_cuda:  # 如果是CUDA环境 # 如果在CUDA环境下
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack  # 导入Marlin重打包核函数 # 导入GPTQ Marlin权重重打包JIT核函数


def mxfp4_marlin_process_scales(  # 处理MXFP4 Marlin缩放因子 # 对MXFP4格式的Marlin缩放因子进行后处理（排列和类型转换）
    marlin_scales: torch.Tensor,  # Marlin缩放因子张量 # 已排列的Marlin缩放因子张量
    input_dtype: torch.dtype | None = None,  # 输入数据类型（可选） # 输入张量的数据类型，默认为None
) -> torch.Tensor:  # 返回处理后的缩放因子 # 返回处理后的缩放因子张量
    if input_dtype is None or input_dtype.itemsize == 2:  # 如果未指定类型或为2字节类型 # 检查是否为默认情况或2字节数据类型（fp16/bf16）
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(  # 重新排列缩放因子 # 对缩放因子进行4列排列
            marlin_scales.size(0), -1  # 恢复形状 # 恢复原始行数
        )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)  # 转换为FP8 E8M0格式 # 将缩放因子转为E8M0浮点格式（仅指数）
    if input_dtype == torch.float8_e4m3fn:  # 如果输入为FP8 E4M3格式 # 如果输入张量是FP8 E4M3类型
        marlin_scales = marlin_scales.view(torch.uint8)  # 以uint8视图查看 # 以uint8视图查看缩放因子
        assert marlin_scales.max() <= 249  # 确保最大值不超过249 # 验证指数范围不超过249
        # exponent_bias (fp4->fp8) = 2 ** 3 - 2 ** 1 = 6 # 指数偏移（FP4->FP8）= 2^3 - 2^1 = 6
        marlin_scales = marlin_scales + 6  # 加上6的指数偏移 # 调整指数偏移以补偿FP4到FP8的指数差异
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)  # 转回FP8 E8M0视图 # 转回E8M0浮点视图
    return marlin_scales  # 返回处理后的缩放因子 # 返回处理后的缩放因子


def _normalize_scale_tensor(  # 归一化缩放因子张量 # 将不同格式的缩放因子统一转换为目标数据类型
    scales: torch.Tensor, target_dtype: torch.dtype  # 缩放因子和目标类型 # 输入缩放因子张量和目标数据类型
) -> torch.Tensor:  # 返回归一化后的缩放因子 # 返回转换后的缩放因子张量
    # The kernel consumes E8M0 exponents. Regardless of the placeholder dtype # 核函数消费E8M0指数。无论占位符数据类型如何
    # the loader used, we want the *numerical* value 2**e in ``target_dtype``. # 加载器使用什么，我们都希望``target_dtype``中的*数值*为2**e
    # float32/bfloat16/float16 containers hold the numerical 2**e directly # float32/bfloat16/float16容器直接持有数值2**e
    # (they were filled via a dtype-promoting copy from uint8/e8m0). # （它们通过从uint8/e8m0的数据类型提升拷贝填充）
    # uint8/int8 containers hold the raw E8M0 byte and must be reinterpreted. # uint8/int8容器持有原始E8M0字节，必须重新解释
    if scales.dtype == torch.float8_e8m0fnu:  # 如果已经是E8M0格式 # 如果缩放因子已经是E8M0浮点格式
        return scales.to(target_dtype)  # 直接转换类型 # 直接转换为目标数据类型
    if scales.dtype == torch.uint8:  # 如果是uint8格式 # 如果缩放因子是uint8类型
        return scales.view(torch.float8_e8m0fnu).to(target_dtype)  # 重解释为E8M0后转换 # 将uint8视图解释为E8M0后转换为目标类型
    if scales.dtype == torch.int8:  # 如果是int8格式 # 如果缩放因子是int8类型
        return scales.view(torch.uint8).view(torch.float8_e8m0fnu).to(target_dtype)  # 先转uint8再转E8M0后转换 # 先转为uint8视图再解释为E8M0后转换
    if scales.dtype in (torch.float32, torch.bfloat16, torch.float16):  # 如果是浮点类型 # 如果缩放因子是常见浮点类型
        return scales.to(target_dtype)  # 直接转换类型 # 直接转换为目标数据类型
    raise TypeError(f"Unsupported MXFP4 scale dtype for Marlin: {scales.dtype}")  # 抛出类型错误 # 不支持的缩放因子数据类型


def _get_optional_param(layer: torch.nn.Module, *names: str) -> torch.Tensor | None:  # 获取可选参数 # 从层中按名称顺序获取第一个存在的参数
    for name in names:  # 遍历名称列表 # 遍历候选参数名
        value = getattr(layer, name, None)  # 获取属性 # 尝试获取指定名称的属性
        if value is not None:  # 如果存在 # 如果属性存在
            return value  # 返回该值 # 返回找到的参数值
    return None  # 都不存在则返回None # 所有名称都不存在时返回None


def prepare_moe_mxfp4_layer_for_marlin(layer: torch.nn.Module) -> None:  # 为Marlin准备MoE MXFP4层 # 将MoE层的MXFP4权重预处理为Marlin格式
    group_size = 32  # 组大小为32 # MXFP4的分组大小固定为32
    w13 = layer.w13_weight.data  # 获取gate-up权重数据 # 获取MoE的gate-up权重数据
    w2 = layer.w2_weight.data  # 获取down权重数据 # 获取MoE的down权重数据
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")  # 获取gate-up缩放因子 # 获取gate-up权重的缩放因子
    w2_scale = _get_optional_param(layer, "w2_weight_scale", "w2_weight_scale_inv")  # 获取down缩放因子 # 获取down权重的缩放因子
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")  # 获取gate-up偏置 # 获取gate-up的偏置
    w2_bias = _get_optional_param(layer, "w2_weight_bias", "w2_bias")  # 获取down偏置 # 获取down的偏置

    if w13_scale is None or w2_scale is None:  # 如果缺少缩放因子 # 检查必要的缩放因子是否存在
        raise ValueError("MXFP4 Marlin requires w13/w2 weight scales.")  # 抛出值错误 # 抛出缺少缩放因子异常

    w13_scale_data = w13_scale.data if hasattr(w13_scale, "data") else w13_scale  # 提取缩放因子数据 # 获取缩放因子的底层数据
    w2_scale_data = w2_scale.data if hasattr(w2_scale, "data") else w2_scale  # 提取缩放因子数据 # 获取缩放因子的底层数据
    w13_bias_data = w13_bias.data if hasattr(w13_bias, "data") else w13_bias  # 提取偏置数据 # 获取偏置的底层数据
    w2_bias_data = w2_bias.data if hasattr(w2_bias, "data") else w2_bias  # 提取偏置数据 # 获取偏置的底层数据

    num_experts = w13.shape[0]  # 获取专家数量 # 获取MoE模型中的专家数
    intermediate_size = w13.shape[1] // 2  # 计算中间层大小 # gate-up权重中intermediate_size是总行数的一半
    hidden_size = w13.shape[2] * 2  # 计算隐藏层大小（FP4打包，乘2恢复） # 隐藏层大小等于列数乘以2（因为FP4每2个值打包为1个uint8）
    param_dtype = getattr(  # 获取参数数据类型 # 获取权重的原始数据类型
        layer,  # 层 # 目标层
        "orig_dtype",  # 原始类型属性 # 尝试获取orig_dtype属性
        w13_bias_data.dtype if w13_bias_data is not None else torch.bfloat16,  # 默认bfloat16 # 如果有偏置则用偏置类型，否则默认bfloat16
    )

    device = w13.device  # 获取设备 # 获取权重所在的设备
    layer.workspace = marlin_make_workspace(device, 4)  # 创建工作空间，每个SM最多4个块 # 为Marlin核函数创建工作空间
    perm = torch.empty(0, dtype=torch.int, device=device)  # 创建空排列（MXFP4不需要） # 创建空的排列索引（MXFP4不使用激活排列）

    def _repack_weight(weight: torch.Tensor, is_w13: bool) -> torch.Tensor:  # 重打包权重 # 将FP4权重重打包为Marlin格式
        if is_w13:  # 如果是gate-up权重 # 如果处理的是gate-up权重
            size_n, size_k = intermediate_size * 2, hidden_size  # gate-up的N和K维度 # gate-up权重的输出维度和输入维度
        else:  # 如果是down权重 # 如果处理的是down权重
            size_n, size_k = hidden_size, intermediate_size  # down的N和K维度 # down权重的输出维度和输入维度
        assert weight.shape == (num_experts, size_n, size_k // 2)  # 验证权重形状 # 检查权重形状是否符合预期

        tensor_list = []  # 初始化结果列表 # 创建存储重打包结果的列表
        for i in range(num_experts):  # 遍历每个专家 # 逐个处理每个专家的权重
            qweight = weight[i].view(torch.int32).T.contiguous()  # 转置并转为int32 # 将权重转为int32视图并转置
            marlin_qweight = gptq_marlin_repack(  # 调用Marlin重打包 # 调用GPTQ Marlin重打包核函数
                b_q_weight=qweight,  # 量化权重 # 输入的量化权重
                perm=perm,  # 排列索引 # 排列索引（空）
                size_k=size_k,  # K维度 # 输入维度
                size_n=size_n,  # N维度 # 输出维度
                num_bits=4,  # 4位量化 # 量化位数为4
            )
            tensor_list.append(marlin_qweight)  # 添加到列表 # 将重打包结果添加到列表
        return torch.stack(tensor_list)  # 堆叠并返回 # 将所有专家的结果堆叠为新张量

    def _permute_scales(scales: torch.Tensor, is_w13: bool) -> torch.Tensor:  # 排列缩放因子 # 对缩放因子进行Marlin格式排列和MXFP4后处理
        scales = _normalize_scale_tensor(scales, param_dtype)  # 归一化缩放因子 # 将缩放因子统一转换为目标数据类型

        if is_w13:  # 如果是gate-up权重 # 如果处理的是gate-up缩放因子
            size_n, size_k = intermediate_size * 2, hidden_size  # gate-up的N和K维度 # gate-up的输出和输入维度
        else:  # 如果是down权重 # 如果处理的是down缩放因子
            size_n, size_k = hidden_size, intermediate_size  # down的N和K维度 # down的输出和输入维度

        tensor_list = []  # 初始化结果列表 # 创建存储排列结果的列表
        for i in range(num_experts):  # 遍历每个专家 # 逐个处理每个专家的缩放因子
            scale = scales[i].T.contiguous()  # 转置 # 将缩放因子转置
            marlin_scales = marlin_permute_scales(  # Marlin排列缩放因子 # 调用Marlin缩放因子排列函数
                s=scale,  # 缩放因子 # 输入缩放因子
                size_k=size_k,  # K维度 # 输入维度
                size_n=size_n,  # N维度 # 输出维度
                group_size=group_size,  # 组大小 # 分组大小
            )
            tensor_list.append(  # 添加到列表 # 将处理后的缩放因子添加到列表
                mxfp4_marlin_process_scales(  # MXFP4后处理 # 调用MXFP4缩放因子后处理函数
                    marlin_scales,  # 已排列的缩放因子 # 输入已排列的缩放因子
                    input_dtype=param_dtype,  # 输入数据类型 # 传入参数数据类型
                )
            )
        return torch.stack(tensor_list)  # 堆叠并返回 # 将所有专家的结果堆叠为新张量

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:  # 排列偏置 # 对偏置进行Marlin格式排列
        if bias is None:  # 如果偏置为空 # 如果没有偏置
            return None  # 返回None # 直接返回None
        tensor_list = []  # 初始化结果列表 # 创建存储排列结果的列表
        for i in range(num_experts):  # 遍历每个专家 # 逐个处理每个专家的偏置
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))  # 排列并转换类型 # 对每个专家的偏置进行Marlin排列并转换类型
        return torch.stack(tensor_list)  # 堆叠并返回 # 将所有专家的结果堆叠为新张量

    w13_marlin = _repack_weight(w13, True)  # 重打包gate-up权重 # 对gate-up权重进行Marlin重打包
    w2_marlin = _repack_weight(w2, False)  # 重打包down权重 # 对down权重进行Marlin重打包
    w13_scale_marlin = _permute_scales(w13_scale_data, True)  # 排列gate-up缩放因子 # 对gate-up缩放因子进行排列
    w2_scale_marlin = _permute_scales(w2_scale_data, False)  # 排列down缩放因子 # 对down缩放因子进行排列

    layer.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)  # 注册gate-up权重 # 将重打包后的gate-up权重注册为不需要梯度的参数
    layer.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)  # 注册down权重 # 将重打包后的down权重注册为不需要梯度的参数
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)  # 注册gate-up缩放因子 # 将排列后的gate-up缩放因子注册为参数
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)  # 注册down缩放因子 # 将排列后的down缩放因子注册为参数

    if w13_bias_data is not None:  # 如果gate-up有偏置 # 如果gate-up存在偏置
        layer.w13_weight_bias = torch.nn.Parameter(  # 注册gate-up偏置 # 将排列后的gate-up偏置注册为参数
            _permute_bias(w13_bias_data), requires_grad=False  # 排列偏置 # 排列后的偏置，不需要梯度
        )
    if w2_bias_data is not None:  # 如果down有偏置 # 如果down存在偏置
        layer.w2_weight_bias = torch.nn.Parameter(  # 注册down偏置 # 将排列后的down偏置注册为参数
            _permute_bias(w2_bias_data), requires_grad=False  # 排列偏置 # 排列后的偏置，不需要梯度
        )
