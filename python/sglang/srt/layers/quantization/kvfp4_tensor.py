# Copyright 2025 SGLang Team # 版权所有2025 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License"); # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at # 可在以下地址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0 # Apache许可证链接
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 根据许可证分发的软件按"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and # 查看许可证以了解管理权限和
# limitations under the License. # 限制的特定语言
# ============================================================================== # 分隔线

# FP4 KV缓存张量模块：定义FP4格式的枚举类、E2M1常量以及BlockFP4和NVFP4的量化/反量化工具类
# 支持MXFP4（块级缩放）和NVFP4（两级缩放：全局FP32 + 块级FP8 E4M3）两种FP4格式

# Define a enum class for FP4 formats, including MXFP4, NVFP4 and future formats # 定义FP4格式的枚举类，包括MXFP4、NVFP4及未来格式
from enum import Enum  # 导入枚举类 # 从标准库导入枚举类型

import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架


class FP4KVCacheRecipe(Enum):  # FP4 KV缓存配方枚举 # FP4 KV缓存量化格式枚举类
    MXFP4 = 1  # KVFP4: block-wise scaling # MXFP4格式：块级缩放 # MXFP4：块级缩放方式
    NVFP4 = 2  # two-level scaling: global FP32 + block FP8 E4M3 # NVFP4格式：两级缩放 # NVFP4：两级缩放方式（全局FP32 + 块级FP8 E4M3）


E2M1_MAX = 6.0  # E2M1格式最大值 # E2M1浮点格式可表示的最大值
MAX_BLOCK_SCALE_FP8 = 448.0  # Maximum FP8 E4M3 value # FP8 E4M3格式最大值 # FP8 E4M3格式可表示的最大值
# Put constants directly on CUDA if available # 如果可用，将常量直接放在CUDA上
_device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备 # 根据CUDA可用性选择设备
# E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits # E2M1格式：1位符号 + 2位指数 + 1位尾数 = 4位
# 16 possible values: 0x0-0xF # 16个可能值：0x0-0xF
# Negative values: 0x8-0xF (sign bit = 1) # 负数值：0x8-0xF（符号位=1）
# Positive values: 0x0-0x7 (sign bit = 0) # 正数值：0x0-0x7（符号位=0）
E2M1_VALUES = torch.tensor(  # E2M1格式的16个可能值查找表 # E2M1格式所有可能值的查找表
    [  # 值列表 # 数值列表
        0,  # 0x0 # 0x0对应的值
        0.5,  # 0x1 # 0x1对应的值
        1,  # 0x2 # 0x2对应的值
        1.5,  # 0x3 # 0x3对应的值
        2,  # 0x4 # 0x4对应的值
        3,  # 0x5 # 0x5对应的值
        4,  # 0x6 # 0x6对应的值
        6,  # 0x0-0x7: positive values # 0x7 # 0x0-0x7：正数值
        -0,  # 0x8 # 0x8对应的值
        -0.5,  # 0x9 # 0x9对应的值
        -1,  # 0xA # 0xA对应的值
        -1.5,  # 0xB # 0xB对应的值
        -2,  # 0xC # 0xC对应的值
        -3,  # 0xD # 0xD对应的值
        -4,  # 0xE # 0xE对应的值
        -6,  # 0x8-0xF: negative values # 0xF # 0x8-0xF：负数值
    ],  # 值列表结束 # 列表结束
    dtype=torch.float32,  # 数据类型为float32 # 使用float32精度
    device=_device,  # 设备 # 指定计算设备
)  # E2M1_VALUES定义结束 # 张量定义结束
E2M1_BOUNDS = torch.tensor(  # E2M1格式量化边界值 # E2M1格式各量化区间的边界值
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device  # 边界值列表及类型 # 各量化区间的上界值
)  # E2M1_BOUNDS定义结束 # 张量定义结束


class BlockFP4KVQuantizeUtil:  # 块级FP4(E2M1)KV缓存量化工具类 # 块级FP4(E2M1)格式KV缓存量化与反量化工具类
    """Block-wise FP4 (E2M1) quantization for KV cache. # KV缓存的块级FP4(E2M1)量化

    Similar to MXFP4 but uses block_size=16 (MXFP4 spec defines block_size=32). # 类似于MXFP4，但使用block_size=16（MXFP4规范定义block_size=32）
    Each block of 16 elements shares one uint8 exponent-only scale factor. # 每16个元素的块共享一个uint8仅指数缩放因子
    """

    @staticmethod  # 静态方法 # 声明为静态方法
    @torch.compile  # 使用torch.compile编译优化 # 使用torch.compile进行编译优化
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # 批量量化方法 # 将张量批量量化为KVFP4格式
        """
        Quantize tensor to KVFP4 format # 将张量量化为KVFP4格式
        Args: # 参数：
            tensor: Input tensor of shape [B, M, N] # tensor：形状为[B, M, N]的输入张量

        Returns: # 返回值：
            quant_tensor: Quantized tensor of shape [B, M, N/2] # quant_tensor：形状为[B, M, N/2]的量化张量
            scale_factors: Scale factors of shape [B, M*N/16] # scale_factors：形状为[B, M*N/16]的缩放因子
        """
        b, m, n = tensor.shape  # 解析张量维度 # 获取输入张量的B、M、N维度

        # Reshape to [B, M*N/16, 16] for block-wise quantization # 重塑为[B, M*N/16, 16]以进行块级量化
        reshaped = tensor.view(b, m * n // 16, 16)  # 重塑为块级形状 # 将张量重塑为每块16个元素的形式

        # Compute scale factors per block # 计算每个块的缩放因子
        block_max = reshaped.abs().max(dim=-1, keepdim=True).values  # 计算每个块的绝对值最大值 # 计算每个块内元素的绝对值最大值
        scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))  # 计算缩放指数 # 计算缩放所需的2的指数
        scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)  # 将指数转为uint8格式的缩放因子 # 将指数偏移127后转为uint8存储

        # Apply scaling # 应用缩放
        scaled = reshaped / torch.exp2(scale_exp)  # 对每个块应用缩放 # 将每个块除以2的指数次方进行缩放

        # Quantize to FP4 # 量化为FP4
        sign_bits = (scaled < 0).to(torch.uint8) << 3  # 提取符号位并左移3位 # 提取负数的符号位，左移3位放到最高位
        abs_vals = scaled.abs()  # 取绝对值 # 取缩放后值的绝对值

        # Pure tensor version (CUDA Graph safe) # 纯张量版本（CUDA Graph安全）
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)  # 计算幅度位 # 通过比较边界值确定每个元素的幅度编码

        # Combine sign and magnitude # 合并符号和幅度
        fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)  # 合并符号位和幅度位为FP4值 # 将符号位和幅度位组合成4位FP4值

        # Pack two FP4 values into one uint8 # 将两个FP4值打包为一个uint8
        fp4_reshaped = fp4_vals.view(b, m, n)  # 重塑FP4值 # 将FP4值重塑回原始形状
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]  # 打包：偶数位低4位，奇数位高4位 # 奇数位置左移4位后与偶数位置相加，打包为一个uint8

        return packed, scale_factors  # 返回打包后的FP4数据和缩放因子 # 返回量化后的打包数据及缩放因子

    @staticmethod  # 静态方法 # 声明为静态方法
    @torch.compile  # 使用torch.compile编译优化 # 使用torch.compile进行编译优化
    def batched_dequantize(  # 批量反量化方法 # 将KVFP4格式张量批量反量化
        quant_tensor: torch.Tensor,  # 量化张量 # 量化后的张量
        scale_factors: torch.Tensor,  # 缩放因子 # 缩放因子张量
        dtype: torch.dtype = torch.bfloat16,  # 输出数据类型 # 目标输出数据类型，默认bfloat16
    ) -> torch.Tensor:  # 返回反量化张量 # 返回反量化后的张量
        """
        Dequantize KVFP4 tensor # 反量化KVFP4张量
        Args: # 参数：
            quant_tensor: Quantized tensor of shape [B, M, N/2] # quant_tensor：形状为[B, M, N/2]的量化张量
            scale_factors: Scale factors of shape [B, M*N/16] # scale_factors：形状为[B, M*N/16]的缩放因子
            dtype: Target dtype for output # dtype：目标输出数据类型

        Returns: # 返回值：
            Dequantized tensor of shape [B, M, N] # 形状为[B, M, N]的反量化张量
        """
        b, m, n_half = quant_tensor.shape  # 解析量化张量维度 # 获取量化张量的B、M和半宽维度
        n = n_half * 2  # 计算原始宽度 # 计算反量化后的完整宽度

        # More efficient unpacking using bit operations # 使用位操作更高效地解包
        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)  # 创建空张量用于存储解包后的FP4值 # 创建空张量存储解包后的FP4值
        fp4_vals[..., 0::2] = quant_tensor & 0x0F  # 提取低4位（偶数位置） # 提取打包数据中偶数位置的FP4值（低4位）
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F  # 提取高4位（奇数位置） # 提取打包数据中奇数位置的FP4值（高4位）

        # Extract sign and magnitude # 提取符号和幅度
        sign_mask = (fp4_vals & 0x08) != 0  # 判断符号位 # 检查最高位（符号位）是否为1
        magnitude_idx = fp4_vals & 0x07  # 提取低3位幅度索引 # 提取低3位作为幅度索引

        # Convert to float values # 转换为浮点值
        float_vals = E2M1_VALUES[magnitude_idx.long()]  # 通过查找表获取浮点值 # 使用幅度索引在E2M1查找表中获取对应的浮点值
        float_vals = torch.where(sign_mask, -float_vals, float_vals)  # 根据符号位取负 # 如果符号位为1则取负值

        # Reshape for block-wise scaling # 重塑为块级缩放形式
        reshaped = float_vals.view(b, m * n // 16, 16)  # 重塑为块级形状 # 将浮点值重塑为每块16个元素的形式

        # Apply scale factors # 应用缩放因子
        scale_exp = scale_factors.float() - 127  # 将uint8缩放因子转回指数 # 将uint8格式的缩放因子减去偏移127还原为指数
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))  # 乘以2的指数次方 # 将每个块乘以对应的2的指数次方进行反缩放

        return scaled.view(b, m, n).to(dtype)  # 重塑形状并转换类型后返回 # 重塑为目标形状并转换为目标数据类型后返回


class NVFP4KVQuantizeUtil:  # NVFP4 KV缓存量化工具类 # NVFP4格式KV缓存量化与反量化工具类
    """Utility class for NVFP4 quantization and dequantization with two-level scaling # 带有两级缩放的NVFP4量化和反量化工具类
    (global FP32 + block FP8 E4M3). # （全局FP32 + 块级FP8 E4M3）

    Quantize formula:  x_fp4 * block_scale * global_scale = x_bf16 # 量化公式：x_fp4 * block_scale * global_scale = x_bf16
    - Quantize: ``nvfp4_kv_quantize`` (SM100+), fallback ``fp4_quantize`` (SM90) # 量化：SM100+使用nvfp4_kv_quantize，SM90回退到fp4_quantize
    - Dequantize: ``nvfp4_kv_dequantize`` (SM100+) # 反量化：SM100+使用nvfp4_kv_dequantize
    """

    @staticmethod  # 静态方法 # 声明为静态方法
    def quantize(  # NVFP4量化方法 # 将BF16/FP16张量量化为NVFP4格式
        tensor: torch.Tensor, global_scale: torch.Tensor  # 输入张量和全局缩放因子 # 输入张量及全局缩放因子
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回FP4数据、块缩放因子和全局缩放因子 # 返回量化后的FP4数据、块缩放因子和全局缩放因子
        """Quantize BF16/FP16 tensor to NVFP4 format. # 将BF16/FP16张量量化为NVFP4格式

        Requires SM90+.  Uses ``nvfp4_kv_quantize`` on SM100+ (native PTX), # 需要SM90+。SM100+使用nvfp4_kv_quantize（原生PTX）
        falls back to ``fp4_quantize`` on SM90. # SM90回退到fp4_quantize

        Args: # 参数：
            tensor: Input tensor of shape [B, M, N] # tensor：形状为[B, M, N]的输入张量
            global_scale: Global scale factor (float32 scalar or 1-element tensor) # global_scale：全局缩放因子（float32标量或1元素张量）

        Returns: # 返回值：
            (fp4_data, block_scales, global_scale): # 三元组：FP4数据、块缩放因子、全局缩放因子
                fp4_data: shape [B, M, N/2], dtype uint8 # fp4_data：形状[B, M, N/2]，uint8类型
                block_scales: shape [B, M, N/16], dtype float8_e4m3fn # block_scales：形状[B, M, N/16]，float8_e4m3fn类型
                global_scale: passthrough # global_scale：直接透传
        """
        from sglang.srt.utils import is_sm90_supported, is_sm100_supported  # 导入SM版本检测函数 # 导入GPU架构版本检测函数

        assert is_sm90_supported(), "NVFP4 KV cache quantize requires SM90+ GPU"  # 断言需要SM90+GPU # 确保GPU架构版本不低于SM90

        b, m, n = tensor.shape  # 解析张量维度 # 获取输入张量的B、M、N维度
        tensor_2d = tensor.reshape(b * m, n)  # 重塑为2D # 将3D张量重塑为2D

        if isinstance(global_scale, (int, float)):  # 如果全局缩放因子是标量 # 检查全局缩放因子是否为Python标量
            global_scale = torch.tensor(  # 将标量转为张量 # 将Python标量转为PyTorch张量
                [global_scale], dtype=torch.float32, device=tensor.device  # 创建float32张量 # 创建1元素float32张量
            )
        elif global_scale.dim() == 0:  # 如果是0维张量 # 检查是否为0维张量（标量张量）
            global_scale = global_scale.unsqueeze(0)  # 增加一个维度 # 增加一维使其变为1维张量

        if is_sm100_supported():  # 如果支持SM100架构 # 检测是否支持SM100架构
            from flashinfer import nvfp4_kv_quantize  # 导入SM100+专用量化函数 # 从flashinfer导入SM100+原生的NVFP4 KV量化函数

            # nvfp4_kv_quantize takes global_scale directly (not inverted) # nvfp4_kv_quantize直接接受global_scale（不取倒数）
            fp4_2d, scales_2d = nvfp4_kv_quantize(tensor_2d, global_scale)  # 调用SM100+量化 # 调用SM100+原生NVFP4量化函数
        else:  # 否则使用SM90回退方案 # SM90回退分支
            # SM90: fp4_quantize takes inverted global_scale # SM90：fp4_quantize接受倒数的global_scale
            from flashinfer import fp4_quantize  # 导入SM90回退量化函数 # 从flashinfer导入SM90回退的FP4量化函数

            global_scale_inv = 1.0 / global_scale  # 计算全局缩放因子的倒数 # 计算global_scale的倒数
            fp4_2d, scales_2d = fp4_quantize(  # 调用SM90量化函数 # 调用SM90回退的FP4量化函数
                tensor_2d,  # 输入2D张量 # 输入的2D张量
                global_scale_inv,  # 全局缩放因子倒数 # 传入倒数的全局缩放因子
                sf_vec_size=16,  # 缩放因子向量大小为16 # 每个缩放因子覆盖16个元素
                sf_use_ue8m0=False,  # 不使用UE8M0格式 # 不使用无符号E8M0缩放因子格式
                is_sf_swizzled_layout=False,  # 不使用swizzled布局 # 缩放因子不使用swizzled内存布局
                is_sf_8x4_layout=False,  # 不使用8x4布局 # 缩放因子不使用8x4内存布局
                enable_pdl=None,  # 不启用PDL # 不指定PDL（预取延迟隐藏）
            )  # 量化调用结束 # 函数调用结束

        fp4_data = fp4_2d.view(b, m, fp4_2d.shape[-1])  # 重塑FP4数据为3D # 将2D量化结果重塑为3D形状
        block_scales = scales_2d.view(b, m, scales_2d.shape[-1]).view(  # 重塑缩放因子为3D并转换类型 # 将2D缩放因子重塑为3D形状
            torch.float8_e4m3fn  # 转换为float8_e4m3fn类型 # 将视图转为FP8 E4M3格式
        )  # 视图转换结束 # 视图操作结束
        return fp4_data, block_scales, global_scale  # 返回FP4数据、块缩放因子和全局缩放因子 # 返回量化结果三元组

    @staticmethod  # 静态方法 # 声明为静态方法
    def dequantize(  # NVFP4反量化方法 # 将NVFP4格式张量反量化为BF16/FP16
        quant_tensor: torch.Tensor,  # 量化张量 # 量化后的FP4张量
        block_scales: torch.Tensor,  # 块缩放因子 # 块级FP8 E4M3缩放因子
        global_scale: torch.Tensor,  # 全局缩放因子 # 全局FP32缩放因子
        dtype: torch.dtype = torch.bfloat16,  # 输出数据类型 # 目标输出数据类型，默认bfloat16
    ) -> torch.Tensor:  # 返回反量化张量 # 返回反量化后的张量
        """Dequantize NVFP4 tensor to BF16/FP16. # 将NVFP4张量反量化为BF16/FP16

        Uses ``nvfp4_kv_dequantize`` on SM100+, falls back to pure PyTorch # SM100+使用nvfp4_kv_dequantize，SM90回退到纯PyTorch
        E2M1 LUT on SM90. # E2M1查找表

        Args: # 参数：
            quant_tensor: Packed FP4 data of shape [B, M, N/2] (uint8) # quant_tensor：形状[B, M, N/2]的打包FP4数据（uint8）
            block_scales: Per-block FP8 E4M3 scales of shape [B, M, N/16] # block_scales：形状[B, M, N/16]的逐块FP8 E4M3缩放因子
            global_scale: Global scale factor (float32) # global_scale：全局缩放因子（float32）
            dtype: Output dtype (bfloat16 or float16) # dtype：输出数据类型（bfloat16或float16）

        Returns: # 返回值：
            Dequantized tensor of shape [B, M, N] # 形状为[B, M, N]的反量化张量
        """
        from sglang.srt.utils import is_sm100_supported  # 导入SM100版本检测函数 # 导入SM100架构版本检测函数

        b, m, n_half = quant_tensor.shape  # 解析量化张量维度 # 获取量化张量的B、M和半宽维度

        if isinstance(global_scale, (int, float)):  # 如果全局缩放因子是标量 # 检查全局缩放因子是否为Python标量
            global_scale = torch.tensor(  # 将标量转为张量 # 将Python标量转为PyTorch张量
                [global_scale], dtype=torch.float32, device=quant_tensor.device  # 创建float32张量 # 创建1元素float32张量
            )
        elif global_scale.dim() == 0:  # 如果是0维张量 # 检查是否为0维张量（标量张量）
            global_scale = global_scale.unsqueeze(0)  # 增加一个维度 # 增加一维使其变为1维张量

        if is_sm100_supported():  # 如果支持SM100架构 # 检测是否支持SM100架构
            from flashinfer import nvfp4_kv_dequantize  # 导入SM100+专用反量化函数 # 从flashinfer导入SM100+原生的NVFP4 KV反量化函数

            quant_2d = quant_tensor.view(torch.uint8).reshape(b * m, n_half)  # 重塑量化数据为2D # 将量化数据重塑为2D
            scales_2d = block_scales.view(torch.uint8).reshape(b * m, -1)  # 重塑缩放因子为2D # 将块缩放因子重塑为2D
            output_2d = nvfp4_kv_dequantize(  # 调用SM100+反量化 # 调用SM100+原生NVFP4 KV反量化函数
                quant_2d, scales_2d, global_scale, output_dtype=dtype  # 传入量化数据、缩放因子和目标类型 # 传入量化数据、块缩放因子、全局缩放因子和输出类型
            )  # 反量化调用结束 # 函数调用结束
            return output_2d.reshape(b, m, -1)  # 重塑为3D并返回 # 将2D输出重塑为3D形状后返回
        else:  # SM90回退方案 # SM90纯PyTorch回退分支
            # Pure PyTorch fallback for SM90 # SM90的纯PyTorch回退方案
            n = n_half * 2  # 计算原始宽度 # 计算反量化后的完整宽度
            fp4_vals = torch.empty(  # 创建空张量用于存储解包后的FP4值 # 创建空张量存储解包后的FP4值
                b, m, n, dtype=torch.uint8, device=quant_tensor.device  # 指定形状、类型和设备 # 指定张量的维度、数据类型和设备
            )  # 空张量创建结束 # 张量创建结束
            fp4_vals[..., 0::2] = quant_tensor & 0x0F  # 提取低4位（偶数位置） # 提取打包数据中偶数位置的FP4值（低4位）
            fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F  # 提取高4位（奇数位置） # 提取打包数据中奇数位置的FP4值（高4位）
            float_vals = E2M1_VALUES[fp4_vals.long()]  # 通过查找表获取浮点值 # 使用FP4值作为索引在E2M1查找表中获取对应浮点值
            reshaped = float_vals.view(b, m * n // 16, 16)  # 重塑为块级形状 # 将浮点值重塑为每块16个元素的形式
            block_scales_float = block_scales.float().unsqueeze(-1)  # 将块缩放因子转为float并增加维度 # 将FP8块缩放因子转为float32并扩展维度用于广播
            scaled = reshaped * block_scales_float  # 应用块缩放因子 # 将每个块乘以对应的块缩放因子
            return (scaled.view(b, m, n) * global_scale).to(dtype)  # 应用全局缩放因子并转换类型后返回 # 重塑形状、应用全局缩放因子并转换为目标数据类型后返回
