# 文件说明：MXFP4张量量化与反量化工具类
# 本模块实现了MXFP4格式的量化与反量化操作，包括E2M1浮点4位的量化、
# uint4到uint8的打包与解包、E8M0块缩放因子的计算与应用。
# 参考自NVIDIA TensorRT-Model-Optimizer的mxfp4_tensor.py实现。
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # 版权声明：NVIDIA公司及其附属公司版权所有
# SPDX-License-Identifier: Apache-2.0 # 许可证标识符：Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); # 根据Apache许可证2.0版（"许可证"）授权；
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件。
# You may obtain a copy of the License at # 可以在以下地址获取许可证副本：
#
# http://www.apache.org/licenses/LICENSE-2.0 # http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意，否则按许可证分发的软件
# distributed under the License is distributed on an "AS IS" BASIS, # 按"原样"基础分发，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不提供任何明示或暗示的担保或条件。
# See the License for the specific language governing permissions and # 参见许可证了解许可证下的特定语言管理权限和
# limitations under the License. # 限制。

from typing import Optional # 导入可选类型

import torch # 导入PyTorch


# https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/mxfp4_tensor.py # 参考链接：NVIDIA TensorRT模型优化器中的MXFP4张量实现
class MXFP4QuantizeUtil: # MXFP4量化工具类，提供量化与反量化方法
    E2M1_max = 6.0 # E2M1格式可表示的最大值

    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6] # E2M1格式可表示的所有8个值（1符号位+2指数位+1尾数位）
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]) # E2M1值之间的边界，用于量化时选择最近的表示值

    @classmethod # 类方法：将张量量化为MXFP4格式
    def quantize(cls, input: torch.Tensor, block_size: Optional[int]) -> tuple: # 输入张量和块大小，返回量化数据和缩放因子
        """Converting a tensor to a quantized format based on MXFP4 quantization. Only E4M3 is supported. # 将张量转换为基于MXFP4量化的格式。仅支持E4M3。
        Args: # 参数：
            input (torch.Tensor): The input tensor to be quantized. # input：要量化的输入张量。
            block_sizes (dict | None): The block sizes for quantization. # block_sizes：量化的块大小。
        """

        def cast_fp4(x): # 将浮点值转换为FP4格式（uint4）
            sign = torch.sign(x) # 获取符号（+1, -1, 0）
            sign_bit = (2 - sign) // 2 # 将符号转换为0/1位（正数为0，负数为1）
            ord_ = torch.sum( # 计算E2M1值的索引
                (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1 # 统计绝对值超过多少个边界
            )
            fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8) # 组合符号位（1位）和幅度索引（3位）为uint8
            return fp4_val # 返回FP4编码值

        def fuse_uint4_to_uint8(x): # 将两个uint4值打包为一个uint8值
            # If the last dimension is odd, pad with zeros # 如果最后一个维度为奇数，用零填充
            # If this behavior is not desired, please modify the code accordingly # 如果不需要此行为，请相应修改代码
            left_side = x[..., 0::2]  # Even indices (0, 2, 4...) # 偶数索引（0, 2, 4...），对应低4位
            right_side = x[..., 1::2]  # Odd indices (1, 3, 5...) # 奇数索引（1, 3, 5...），对应高4位
            new_data = ( # 构建打包后的数据
                right_side.clone() << 4 # 将奇数索引值左移4位放到高位
            )  # Put odd indices (higher addresses) in high bits # 将奇数索引（高地址）放在高位
            new_data[ # 将偶数索引值加到低4位
                ..., : left_side.shape[-1]
            ] += left_side  # Put even indices in low bits # 将偶数索引放在低位
            return new_data # 返回打包后的uint8数据

        if block_size is None: # 如果未指定块大小
            block_size = 32 # 默认块大小为32

        original_shape = input.shape # 保存原始形状
        original_dtype = input.dtype # 保存原始数据类型
        input = input.view(-1, block_size) # 重塑为(-1, block_size)以便按块计算缩放因子
        # get scales # 获取缩放因子
        input_amax = input.abs().max(dim=-1, keepdim=True).values # 计算每个块的绝对值最大值
        descale = input_amax / cls.E2M1_max # 计算去缩放因子（最大值/E2M1最大值）
        min_value = torch.tensor(-127.0, device=descale.device) # 最小指数偏移值
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value)) # 计算E8M0格式的缩放因子指数（向上取整）

        input = (input / torch.exp2(e8m0_scale)).view(original_shape) # 用缩放因子归一化输入并恢复原始形状
        input_q = cast_fp4(input) # 将归一化后的值转换为FP4编码
        input_q = fuse_uint4_to_uint8(input_q) # 将uint4打包为uint8
        e8m0_scale = (e8m0_scale + 127).to(torch.uint8) # 将E8M0指数转换为偏移编码（+127偏移）
        return cls(original_shape, original_dtype, input_q), e8m0_scale # 返回量化数据对象和E8M0缩放因子

    @classmethod # 类方法：将MXFP4量化数据反量化为目标数据类型
    def dequantize(cls, quantized_data, dtype: torch.dtype, scale, block_sizes): # 输入量化数据、目标类型、缩放因子和块大小
        """Dequantze MXFP4 packed tensor to a target dtype.""" # 将MXFP4打包张量反量化为目标数据类型。

        def unfuse_uint8_to_uint4(x): # 将uint8拆包为两个uint4值
            """Unfuse uint8 values back to uint4 values. # 将uint8值拆包回uint4值。
            This is the inverse operation of fuse_uint4_to_uint8. # 这是fuse_uint4_to_uint8的逆操作。
            """
            # Extract the lower 4 bits (even indices) # 提取低4位（偶数索引）
            left_side = x & 0x0F # 按位与0x0F获取低4位

            # Extract the upper 4 bits (odd indices) # 提取高4位（奇数索引）
            right_side = (x >> 4) & 0x0F # 右移4位后按位与0x0F获取高4位

            # Create a new tensor with alternating values # 创建一个交替值的新张量
            shape = list(x.shape) # 获取原始形状
            shape[-1] = shape[-1] * 2 # 最后一维度大小翻倍（每个uint8拆为2个uint4）
            result = torch.zeros(shape, dtype=torch.uint8, device=x.device) # 创建零张量

            # Fill in the values - even indices get low bits, odd indices get high bits # 填充值——偶数索引获取低位，奇数索引获取高位
            result[..., 0::2] = left_side  # Even indices from low bits # 偶数索引来自低4位
            result[..., 1::2] = right_side  # Odd indices from high bits # 奇数索引来自高4位

            return result # 返回拆包后的uint4张量

        e8m0_scale = scale # 获取E8M0缩放因子
        block_size = block_sizes[-1] # 获取最后一个维度的块大小

        # Unfuse the uint8 values back to uint4 # 将uint8值拆包回uint4
        x_unfused = unfuse_uint8_to_uint4(quantized_data) # 调用拆包函数
        # Extract sign and magnitude # 提取符号和幅度
        sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to( # 从第3位提取符号位并转换为+1/-1
            torch.float32
        )  # Extract sign bit and convert to +1/-1 # 提取符号位并转换为+1/-1
        magnitude = x_unfused & 0b0111  # Extract magnitude bits # 提取幅度位（低3位）
        magnitude = magnitude.to(torch.long) # 转换为long类型用于索引

        # Create a tensor with the E2M1 values # 创建E2M1值张量
        values = torch.tensor(cls.E2M1_values, device=quantized_data.device) # 获取E2M1可表示值表

        # Use gather to index the values tensor properly # 使用gather正确索引值张量
        # We need to reshape magnitude to match the dimensions we want to gather along # 需要重塑幅度以匹配我们要沿其收集的维度
        original_shape = magnitude.shape # 保存原始形状
        x_float = values[magnitude.reshape(-1)].reshape(original_shape) # 用幅度索引查表获取浮点值

        # Apply sign and scale # 应用符号和缩放
        x_float = sign.float() * x_float # 应用符号（正负）

        # Reshape to apply block-wise scaling # 重塑以应用块级缩放
        x_float = x_float.reshape(-1, block_size) # 重塑为(-1, block_size)以便按块缩放

        # Apply the E8M0 scale # 应用E8M0缩放因子
        scale_factor = torch.exp2(e8m0_scale.float() - 127) # 将偏移编码还原为实际缩放值（2^(exponent-127)）
        scale_factor = scale_factor.reshape(-1, 1)  # Reshape for proper broadcasting # 重塑以便正确广播

        # Apply scaling and reshape back to original shape # 应用缩放并恢复原始形状
        x_float = x_float * scale_factor # 乘以块缩放因子

        # Reshape back to the original shape # 恢复原始形状
        return x_float.reshape(original_shape).to(dtype) # 重塑并转换为目标数据类型返回
