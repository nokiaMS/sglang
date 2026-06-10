# INT8量化工具模块：提供INT8块级量化和反量化的实用函数
# 包含W8A8块级INT8线性层应用、输入INT8量化和块级反量化功能

from typing import List, Optional, Tuple  # 导入类型提示 # 从typing模块导入类型提示

import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架

from sglang.srt.layers.quantization.int8_kernel import (  # 从int8_kernel模块导入INT8核函数 # 从INT8核函数模块导入量化计算函数
    per_token_group_quant_int8,  # 按token组量化为INT8 # 按token组进行INT8量化
    w8a8_block_int8_matmul,  # W8A8块级INT8矩阵乘法 # W8A8块级INT8矩阵乘法
)


def apply_w8a8_block_int8_linear(  # 应用W8A8块级INT8线性层 # 应用W8A8块级INT8线性变换
    input: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 权重张量 # 权重张量（INT8格式）
    block_size: List[int],  # 块大小 # 块大小列表
    weight_scale: torch.Tensor,  # 权重缩放因子 # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子（可选） # 输入缩放因子，默认为None
    bias: Optional[torch.Tensor] = None,  # 偏置（可选） # 偏置项，默认为None
) -> torch.Tensor:  # 返回输出张量 # 返回计算结果张量
    assert input_scale is None  # 断言输入缩放因子必须为None # 确保input_scale未指定
    # View input as 2D matrix for fp8 methods # 将输入视为2D矩阵（用于fp8方法） # 将输入重塑为二维矩阵
    input_2d = input.view(-1, input.shape[-1])  # 将输入重塑为2D # 将输入重塑为二维矩阵
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 计算输出形状 # 计算输出张量的形状

    q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])  # 对输入进行按token组INT8量化 # 对输入进行按token组的INT8量化，返回量化结果和缩放因子
    output = w8a8_block_int8_matmul(  # 执行W8A8块级INT8矩阵乘法 # 执行块级INT8矩阵乘法
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input.dtype  # 传入量化输入、权重、缩放因子等参数 # 传入量化输入、权重、缩放因子和块大小等参数
    )  # 矩阵乘法调用结束 # 矩阵乘法调用结束

    if bias is not None:  # 如果存在偏置 # 如果偏置不为空
        output = output + bias  # 加上偏置 # 将偏置加到输出上
    return output.to(dtype=input.dtype).view(*output_shape)  # 转换类型并重塑形状后返回 # 转换为输入数据类型并重塑为目标形状后返回


def input_to_int8(  # 将输入量化为INT8 # 将输入张量量化为INT8格式
    x: torch.Tensor, dtype: torch.dtype = torch.int8  # 输入张量和目标数据类型 # 输入张量及目标INT8数据类型
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子 # 返回量化后的张量和缩放因子
    """This function quantizes input values to int8 values with tensor-wise quantization."""  # 此函数使用张量级量化将输入值量化为INT8值 # 使用张量级量化将输入值量化为INT8
    iinfo = torch.iinfo(dtype)  # 获取数据类型信息 # 获取INT8数据类型的最小最大值信息
    min_val, max_val = x.aminmax()  # 计算最小值和最大值 # 计算输入张量的最小值和最大值
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)  # 计算绝对值最大值 # 取最小值和最大值的绝对值中较大者，并限制下界防止除零
    int8_min, int8_max = iinfo.min, iinfo.max  # 获取INT8的最小最大值 # 获取INT8类型可表示的最小和最大值
    scale = int8_max / amax  # 计算缩放因子 # 计算量化缩放因子
    x_scl_sat = (x * scale).clamp(min=int8_min, max=int8_max)  # 缩放并截断 # 将输入乘以缩放因子并截断到INT8范围内
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()  # 返回量化结果和缩放因子的倒数 # 返回INT8量化结果和缩放因子的倒数（用于反量化）


def block_dequant(  # 块级反量化 # 执行块级反量化操作
    x_q_block: torch.Tensor,  # 块级量化张量 # 块级量化后的张量
    x_s: torch.Tensor,  # 块级缩放因子 # 块级缩放因子张量
    block_size: List[int],  # 块大小 # 块大小列表
) -> torch.Tensor:  # 返回反量化张量 # 返回反量化后的张量
    """This function conducts block-wise dequantization. # 此函数执行块级反量化操作
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale # 输入为块级量化张量x_q_block、块级量化缩放因子
    and the block size. # 和块大小
    The outputs are dequantized tensor. # 输出为反量化后的张量
    """
    block_n, block_k = block_size[0], block_size[1]  # 解析块的n和k维度 # 获取块在n和k方向的尺寸
    n, k = x_q_block.shape  # 获取量化张量的形状 # 获取量化张量的n和k维度大小
    n_tiles = (n + block_n - 1) // block_n  # 计算n方向的块数 # 计算n方向的分块数（向上取整）
    k_tiles = (k + block_k - 1) // block_k  # 计算k方向的块数 # 计算k方向的分块数（向上取整）
    assert n_tiles == x_s.shape[0]  # 验证n方向块数与缩放因子一致 # 断言n方向块数与缩放因子的第0维匹配
    assert k_tiles == x_s.shape[1]  # 验证k方向块数与缩放因子一致 # 断言k方向块数与缩放因子的第1维匹配

    x_dq_block = x_q_block.to(torch.float32)  # 转换为float32 # 将量化张量转换为float32精度

    for i in range(k_tiles):  # 遍历k方向的块 # 遍历k方向的每个块
        for j in range(n_tiles):  # 遍历n方向的块 # 遍历n方向的每个块
            x_dq_block[  # 对当前块进行反量化 # 对当前块应用缩放因子
                j * block_n : min((j + 1) * block_n, n),  # n方向的切片范围 # n方向的切片范围
                i * block_k : min((i + 1) * block_k, k),  # k方向的切片范围 # k方向的切片范围
            ] *= x_s[j][i]  # 乘以对应块的缩放因子 # 乘以对应块的缩放因子

    return x_dq_block  # 返回反量化结果 # 返回反量化后的张量
