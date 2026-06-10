# FP8量化内核模块 - 提供FP8量化和矩阵乘法的Triton加速实现
# 包含逐token量化、逐token组量化、分块FP8矩阵乘法、MXFP8矩阵乘法、MLA专用量化等功能
# 支持CUDA、HIP(AMD)和MUSA平台，自动检测硬件能力选择最优实现

# Copyright 2024 SGLang Team  # 版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # Apache 2.0许可证
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 软件按"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 查看许可证了解权限和限制
# limitations under the License.  # 许可证下的限制
# ==============================================================================  # 分隔线

import functools  # 导入functools模块
import json  # 导入json模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import Any, Dict, List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块

try:  # 尝试导入Triton张量描述符
    from triton.tools.tensor_descriptor import TensorDescriptor  # 从Triton导入张量描述符
except:  # 导入失败则忽略
    pass  # 忽略错误

from sglang.srt.layers import deep_gemm_wrapper  # 导入DeepGEMM包装器
from sglang.srt.utils import (  # 从sglang工具模块导入
    ceil_align,  # 向上对齐函数
    get_bool_env_var,  # 获取布尔环境变量
    get_device_core_count,  # 获取设备核心数
    get_device_name,  # 获取设备名称
    is_cpu,  # 是否为CPU
    is_cuda,  # 是否为CUDA
    is_hip,  # 是否为HIP
    is_musa,  # 是否为MUSA
    is_sm100_supported,  # 是否支持SM100
    is_sm120_supported,  # 是否支持SM120
    log_info_on_rank0,  # 在rank0上记录信息
)
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册
from sglang.srt.utils.patch_torch import register_fake_if_exists  # 导入torch补丁注册

_is_hip = is_hip()  # 检测是否为HIP环境
_is_cuda = is_cuda()  # 检测是否为CUDA环境
_is_cpu = is_cpu()  # 检测是否为CPU环境
_is_musa = is_musa()  # 检测是否为MUSA环境
_is_sm100_supported = is_sm100_supported()  # 检测是否支持SM100
_is_sm120_supported = is_sm120_supported()  # 检测是否支持SM120
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER（仅HIP平台）

if _is_cuda or _is_musa:  # CUDA或MUSA平台
    from sgl_kernel import sgl_per_token_quant_fp8  # 导入逐token FP8量化内核

    from sglang.jit_kernel.per_tensor_quant_fp8 import (  # 导入逐张量FP8量化JIT内核
        per_tensor_quant_fp8 as sgl_per_tensor_quant_fp8,
    )

    # Temporary  # 临时处理
    try:  # 尝试导入v2版本
        from sgl_kernel import sgl_per_token_group_quant_8bit  # v2版本的8bit组量化

        enable_sgl_per_token_group_quant_8bit = True  # 启用v2
    except ImportError:  # 导入失败
        from sgl_kernel import sgl_per_token_group_quant_fp8  # 旧版本的FP8组量化

        enable_sgl_per_token_group_quant_8bit = False  # 禁用v2

    from sglang.jit_kernel.per_token_group_quant_8bit import (  # 导入8bit组量化JIT内核
        per_token_group_quant_8bit as sgl_per_token_group_quant_8bit_jit,
    )

if _is_hip:  # HIP平台
    _has_vllm = False  # 是否有vllm可用
    if _use_aiter:  # 如果使用AITER
        try:  # 尝试导入AITER量化函数
            from aiter import (  # v0.1.3  # AITER v0.1.3版本
                dynamic_per_tensor_quant,  # 动态逐张量量化
                dynamic_per_token_scaled_quant,  # 动态逐token缩放量化
                static_per_tensor_quant,  # 静态逐张量量化
            )
        except ImportError:  # 导入失败
            raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")  # AITER是必需的
    else:  # 不使用AITER
        try:  # 尝试导入vllm
            import vllm._C  # noqa: F401  # 导入vllm C扩展

            _has_vllm = True  # vllm可用
        except ImportError:  # 导入失败
            # Fallback: vllm not available, will use native PyTorch implementation  # 回退：vllm不可用，使用原生PyTorch实现
            _has_vllm = False  # vllm不可用

if _is_musa:  # MUSA平台

    @register_fake_if_exists("sgl_kernel::sgl_per_token_group_quant_8bit_v2")  # 注册fake实现
    def _(  # v2版本的fake实现
        input,  # 输入
        output_q,  # 量化输出
        output_s,  # 缩放因子输出
        group_size,  # 组大小
        eps,  # 最小值阈值
        fp8_min,  # FP8最小值
        fp8_max,  # FP8最大值
        scale_ue8m0,  # UE8M0缩放格式
        fuse_silu_and_mul,  # 是否融合SiLU和乘法
        masked_m,  # 掩码M
    ):
        return  # 返回空


logger = logging.getLogger(__name__)  # 创建日志记录器


@lru_cache()  # LRU缓存装饰器
def is_fp8_fnuz() -> bool:  # 检测FP8是否为FNUZ格式
    """检测当前平台是否使用FP8 E4M3FNUZ格式（仅MI300系列AMD GPU）。"""  # 中文函数说明
    if _is_hip:  # 如果是HIP平台
        # only device 0 is checked, this assumes MI300 platforms are homogeneous  # 仅检查设备0，假设MI300平台是同构的
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName  # 检查GPU架构名是否包含gfx94
    return False  # 非HIP平台返回False


if is_fp8_fnuz():  # 如果FP8格式为FNUZ
    fp8_dtype = torch.float8_e4m3fnuz  # 使用E4M3FNUZ数据类型
    fp8_max = 224.0  # FNUZ格式的最大值为224
else:  # 非FNUZ格式
    fp8_dtype = torch.float8_e4m3fn  # 使用E4M3FN数据类型
    fp8_max = torch.finfo(fp8_dtype).max  # FN格式的最大值
fp8_min = -fp8_max  # FP8最小值为负的最大值


@register_custom_op(mutates_args=["C"])  # 注册自定义算子，声明C参数会被修改
def deep_gemm_fp8_fp8_bf16_nt(  # DeepGEMM FP8*FP8->BF16矩阵乘法（非转置布局）
    A: torch.Tensor,  # 矩阵A（激活）
    As: torch.Tensor,  # A的缩放因子
    B: torch.Tensor,  # 矩阵B（权重）
    Bs: torch.Tensor,  # B的缩放因子
    C: torch.Tensor,  # 输出矩阵C
) -> None:  # 无返回值，结果写入C
    """使用DeepGEMM执行FP8*FP8->BF16的矩阵乘法。"""  # 中文函数说明
    deep_gemm_wrapper.gemm_nt_f8f8bf16((A, As), (B, Bs), C)  # 调用DeepGEMM包装器


@triton.jit  # Triton JIT编译装饰器
def _per_token_group_quant_8bit(  # 逐token组8bit量化的Triton内核
    # Pointers to inputs and output  # 输入和输出指针
    y_ptr,  # 输入张量指针
    y_q_ptr,  # 量化输出指针
    y_s_ptr,  # 缩放因子指针
    # Stride of input  # 输入步长
    y_stride,  # 输入行步长
    # Columns of input  # 输入列数
    N,  # 每组元素数
    # Avoid to divide zero  # 避免除零
    eps,  # 最小值阈值
    # Information for float8  # FP8相关信息
    bit8_min,  # 8bit最小值
    bit8_max,  # 8bit最大值
    # Meta-parameters  # 元参数
    BLOCK: tl.constexpr,  # 块大小常量
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """  # Triton加速的逐token组量化函数，将张量值转换为FP8值
    # Map the program id to the row of X and Y it should compute.  # 将程序ID映射到要计算的行
    g_id = tl.program_id(0)  # 获取组ID
    y_ptr += g_id * y_stride  # 移动输入指针
    y_q_ptr += g_id * y_stride  # 移动量化输出指针
    y_s_ptr += g_id  # 移动缩放因子指针

    cols = tl.arange(0, BLOCK)  # N <= BLOCK  # 生成列偏移序列
    mask = cols < N  # 创建掩码

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载输入数据
    # Quant  # 量化操作
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)  # 计算绝对值最大值
    y_s = _absmax / bit8_max  # 计算缩放因子
    y_s_inv = 1.0 / y_s  # 计算缩放因子的倒数
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)  # 量化并裁剪

    tl.store(y_q_ptr + cols, y_q, mask=mask)  # 存储量化结果
    tl.store(y_s_ptr, y_s)  # 存储缩放因子


@triton.jit  # Triton JIT编译装饰器
def _per_token_group_quant_8bit_colmajor(  # 逐token组8bit量化（列主序缩放）的Triton内核
    # Pointers to inputs and output  # 输入和输出指针
    y_ptr,  # 输入张量指针
    y_q_ptr,  # 量化输出指针
    y_s_ptr,  # 缩放因子指针
    group_size,  # 组大小
    # Num columns of y  # y的列数
    y_num_columns,  # 输入列数
    # Stride from one column to the next of y_s  # y_s的列间步长
    y_s_col_stride,  # 缩放因子列步长
    # Avoid to divide zero  # 避免除零
    eps,  # 最小值阈值
    # Information for float8  # FP8相关信息
    bit8_min,  # 8bit最小值
    bit8_max,  # 8bit最大值
    # Meta-parameters  # 元参数
    BLOCK: tl.constexpr,  # 块大小常量
    SCALE_UE8M0: tl.constexpr,  # 是否使用UE8M0缩放格式常量
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """  # Triton加速的逐token组量化函数（列主序），将张量值转换为FP8值
    # Map the program id to the row of X and Y it should compute.  # 将程序ID映射到要计算的行
    g_id = tl.program_id(0)  # 获取组ID
    y_ptr += g_id.to(tl.int64) * group_size  # 移动输入指针
    y_q_ptr += g_id.to(tl.int64) * group_size  # 移动量化输出指针

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix  # 将扁平化的块坐标转换为2D以索引输出缩放矩阵
    blocks_per_row = y_num_columns // group_size  # 每行的块数
    scale_col = g_id % blocks_per_row  # 缩放列索引
    scale_row = g_id // blocks_per_row  # 缩放行索引
    y_s_ptr += scale_col * y_s_col_stride + scale_row  # 移动缩放因子指针

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK  # 生成列偏移序列
    mask = cols < group_size  # 创建掩码

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载输入数据
    # Quant  # 量化操作
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)  # 计算绝对值最大值
    y_s = _absmax / bit8_max  # 计算缩放因子
    if SCALE_UE8M0:  # 如果使用UE8M0缩放格式
        y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))  # 将缩放因子向上取整为2的幂次
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)  # 量化并裁剪

    tl.store(y_q_ptr + cols, y_q, mask=mask)  # 存储量化结果
    tl.store(y_s_ptr, y_s)  # 存储缩放因子


def _per_token_group_quant_8bit_raw(  # 逐token组8bit量化的原始实现
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 组大小
    eps: float = 1e-10,  # 避免除零的最小值
    dtype: torch.dtype = fp8_dtype,  # 输出数据类型
    column_major_scales: bool = False,  # 是否使用列主序缩放
    scale_tma_aligned: bool = False,  # 缩放因子是否TMA对齐
    scale_ue8m0: bool = False,  # 是否使用UE8M0缩放格式
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化张量和缩放因子
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """  # 对输入张量执行逐token组量化，返回FP8量化张量和缩放因子
    assert (  # 断言
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"  # 最后一维必须能被group_size整除
    assert x.is_contiguous(), "`x` is not contiguous"  # 断言输入张量必须是连续的

    if _is_hip:  # HIP平台
        if dtype == torch.int8:  # 如果输出为INT8
            bit8_max = 127.0  # INT8最大值
        else:  # FP8
            bit8_max = 224.0  # FNUZ格式FP8最大值
        bit8_min = -bit8_max  # TODO incorrect for int8  # 最小值（对INT8不准确）
    else:  # 非HIP平台
        if dtype == torch.int8:  # 如果输出为INT8
            info = torch.iinfo(dtype)  # 获取INT8信息
        else:  # FP8
            info = torch.finfo(dtype)  # 获取FP8信息
        bit8_max = info.max  # 最大值
        bit8_min = info.min  # 最小值

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)  # 创建量化输出张量
    x_s = create_per_token_group_quant_fp8_output_scale(  # 创建缩放因子输出张量
        x_shape=x.shape,  # 输入形状
        device=x.device,  # 设备
        group_size=group_size,  # 组大小
        column_major_scales=column_major_scales,  # 列主序缩放
        scale_tma_aligned=scale_tma_aligned,  # TMA对齐
        scale_ue8m0=False,  # 不使用UE8M0
    )

    M = x.numel() // group_size  # 计算总组数
    N = group_size  # 每组元素数

    BLOCK = triton.next_power_of_2(N)  # 计算最小2的幂次块大小
    # heuristics for number of warps  # 启发式计算warp数量
    num_warps = min(max(BLOCK // 256, 1), 8)  # warp数量范围为1-8
    num_stages = 1  # 流水线阶段数为1
    if column_major_scales:  # 如果使用列主序缩放
        _per_token_group_quant_8bit_colmajor[(M,)](  # 调用列主序量化内核
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x_s.stride(1),
            eps,
            bit8_min=bit8_min,
            bit8_max=bit8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
            SCALE_UE8M0=scale_ue8m0,
        )
    else:  # 行主序缩放
        assert not scale_ue8m0  # 不支持UE8M0
        _per_token_group_quant_8bit[(M,)](  # 调用行主序量化内核
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            bit8_min=bit8_min,
            bit8_max=bit8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if scale_ue8m0:  # 如果使用UE8M0缩放格式
        from deep_gemm import transform_sf_into_required_layout  # 导入缩放因子布局转换函数

        assert group_size == 128  # UE8M0仅支持128的组大小
        x_s = transform_sf_into_required_layout(  # 转换缩放因子布局
            x_s,
            num_groups=None,
            mn=x_q.shape[0],
            k=x_q.shape[1],
            recipe=(1, group_size, group_size),
            is_sfa=True,
        )

    return x_q, x_s  # 返回量化结果和缩放因子


def _per_token_group_quant_8bit_fuse_silu_and_mul(  # 融合SiLU和乘法的逐token组8bit量化
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 组大小
    dst_dtype: torch.dtype,  # 目标数据类型
    column_major_scales: bool,  # 是否列主序缩放
    scale_tma_aligned: bool,  # 是否TMA对齐
    scale_ue8m0: bool,  # 是否UE8M0缩放
    masked_m: Optional[torch.Tensor],  # 掩码M张量
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
    """融合SiLU激活和乘法操作的逐token组FP8量化实现。"""  # 中文函数说明
    # Another way to implement (can be used in e.g. comparison tests)
    # from sgl_kernel import silu_and_mul
    # x_after_silu_and_mul = silu_and_mul(x)
    # return per_token_group_quant_fp8(
    #     x_after_silu_and_mul,
    #     group_size=group_size,
    #     eps=eps,
    #     column_major_scales=column_major_scales,
    #     scale_tma_aligned=scale_tma_aligned,
    #     scale_ue8m0=scale_ue8m0,
    # )  # 另一种实现方式（可用于比较测试）

    from deep_gemm import transform_sf_into_required_layout  # 导入布局转换函数

    from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd  # 导入融合内核

    assert column_major_scales  # 必须使用列主序缩放
    assert scale_tma_aligned  # 必须TMA对齐
    assert scale_ue8m0  # 必须使用UE8M0

    needs_unsqueeze = x.dim() == 2  # 如果输入是2维则需要增加维度
    if needs_unsqueeze:  # 需要增加维度
        num_tokens, _ = x.shape  # 获取token数
        x = x.unsqueeze(0)  # 在第0维增加维度
        assert masked_m is None  # 掩码必须为None
        masked_m = torch.tensor([num_tokens], device=x.device, dtype=torch.int32)  # 创建掩码

    # Use `zeros` for easier testing  # 使用zeros便于测试
    output = torch.zeros(  # 创建输出张量
        (*x.shape[:-1], x.shape[-1] // 2),  # 输出维度为输入最后一维的一半
        device=x.device,  # 设备
        dtype=dst_dtype,  # 目标数据类型
    )
    # Use `zeros` for easier testing  # 使用zeros便于测试
    output_scale_for_kernel = torch.zeros(  # 创建缩放因子输出张量
        (*x.shape[:-1], x.shape[-1] // 2 // group_size),  # 缩放因子维度
        device=x.device,  # 设备
        dtype=torch.float32,  # float32类型
    )
    silu_and_mul_masked_post_quant_fwd(  # 调用融合内核
        input=x,  # 输入
        output=output,  # 输出
        output_scale=output_scale_for_kernel,  # 缩放因子
        quant_group_size=group_size,  # 量化组大小
        masked_m=masked_m,  # 掩码
        scale_ue8m0=scale_ue8m0,  # UE8M0缩放
    )

    assert group_size == 128  # UE8M0仅支持128的组大小
    output_scale = transform_sf_into_required_layout(  # 转换缩放因子布局
        output_scale_for_kernel,
        num_groups=output.shape[0],
        mn=output.shape[-2],
        k=output.shape[-1],
        recipe=(1, group_size, group_size),
        is_sfa=True,
    )

    if needs_unsqueeze:  # 如果需要恢复维度
        output = output.squeeze(0)  # 去除第0维
        output_scale = output_scale.squeeze(0)  # 去除缩放因子的第0维

    return output, output_scale  # 返回输出和缩放因子


def per_token_group_quant_8bit(  # 逐token组8bit量化统一入口函数
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 组大小
    dst_dtype: torch.dtype,  # 目标数据类型
    eps: float = 1e-10,  # 避免除零的最小值
    column_major_scales: bool = False,  # 是否列主序缩放
    scale_tma_aligned: bool = False,  # 是否TMA对齐
    scale_ue8m0: bool = False,  # 是否UE8M0缩放
    fuse_silu_and_mul: bool = False,  # 是否融合SiLU和乘法
    masked_m: Optional[torch.Tensor] = None,  # 掩码M
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
    """逐token组8bit量化的统一入口，根据参数选择融合或原始实现。"""  # 中文函数说明
    if fuse_silu_and_mul:  # 如果需要融合SiLU和乘法
        return _per_token_group_quant_8bit_fuse_silu_and_mul(  # 使用融合实现
            x=x,
            group_size=group_size,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
            masked_m=masked_m,
        )
    else:  # 使用原始实现
        return _per_token_group_quant_8bit_raw(  # 调用原始量化实现
            x=x,
            group_size=group_size,
            eps=eps,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
            dtype=dst_dtype,
        )


def create_per_token_group_quant_fp8_output_scale(  # 创建逐token组FP8量化的输出缩放因子张量
    x_shape,  # 输入形状
    device,  # 设备
    group_size,  # 组大小
    column_major_scales: bool,  # 是否列主序缩放
    scale_tma_aligned: bool,  # 是否TMA对齐
    scale_ue8m0: bool,  # 是否UE8M0缩放
):
    """根据参数创建不同布局的FP8量化输出缩放因子张量。"""  # 中文函数说明
    if scale_ue8m0:  # UE8M0缩放格式
        assert column_major_scales and scale_tma_aligned  # 必须列主序且TMA对齐
        *x_batch, x_q_mn, x_q_k = x_shape  # 解包形状
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128  # 计算缩放因子形状
        aligned_mn = ceil_align(x_s_mn, 4)  # M维度4对齐
        aligned_k = ceil_align(x_s_k, 4)  # K维度4对齐
        # TODO(FIXME): Fix cuda kernel and recover here to empty.  # 修复CUDA内核后改回empty
        return torch.empty(  # 创建缩放因子张量
            (*x_batch, aligned_k // 4, aligned_mn),  # UE8M0打包后的形状
            device=device,  # 设备
            dtype=torch.int,  # int32类型（4个uint8打包）
        ).transpose(-1, -2)[..., :x_s_mn, :]  # 转置并裁剪
    elif column_major_scales:  # 列主序缩放
        if scale_tma_aligned:  # TMA对齐
            # TODO extract "align" function  # 提取对齐函数
            # aligned to 4 * sizeof(float)  # 对齐到4*sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4  # 计算对齐后的大小
            return torch.empty(  # 创建缩放因子张量
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),  # 形状
                device=device,  # 设备
                dtype=torch.float32,  # float32类型
            ).transpose(-1, -2)[: x_shape[-2], :]  # 转置并裁剪
        else:  # 列主序但非TMA对齐
            return torch.empty(  # 创建缩放因子张量
                (x_shape[-1] // group_size,) + x_shape[:-1],  # 形状
                device=device,  # 设备
                dtype=torch.float32,  # float32类型
            ).permute(-1, -2)  # 排列维度
    else:  # 行主序缩放
        return torch.empty(  # 创建缩放因子张量
            x_shape[:-1] + (x_shape[-1] // group_size,),  # 形状
            device=device,  # 设备
            dtype=torch.float32,  # float32类型
        )


def sglang_per_token_group_quant_fp8(  # SGLang优化的逐token组FP8量化函数
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 组大小
    eps: float = 1e-10,  # 避免除零
    column_major_scales: bool = False,  # 是否列主序缩放
    scale_tma_aligned: bool = False,  # 是否TMA对齐
    scale_ue8m0: bool = False,  # 是否UE8M0缩放
    fuse_silu_and_mul: bool = False,  # 是否融合SiLU和乘法
    masked_m: Optional[torch.Tensor] = None,  # 掩码M
    enable_v2: Optional[bool] = None,  # 是否启用v2内核
):
    """SGLang优化的逐token组FP8量化，自动选择最优内核（v2或JIT回退）。"""  # 中文函数说明
    assert (  # 断言
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"  # 最后一维必须能被group_size整除
    assert x.is_contiguous(), "`x` is not contiguous"  # 断言连续

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))  # 计算输出形状

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)  # 创建FP8量化输出
    x_s = create_per_token_group_quant_fp8_output_scale(  # 创建缩放因子
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    # Enable v2 kernel by default on supported group sizes  # 默认在支持的组大小上启用v2内核
    _V2_KERNEL_SUPPORTED_GROUP_SIZES = [16, 32, 64, 128]  # v2内核支持的组大小
    if enable_v2 is None:  # 如果未指定
        enable_v2 = group_size in _V2_KERNEL_SUPPORTED_GROUP_SIZES or _is_musa  # 自动选择

    if x.shape[0] > 0:  # 如果输入非空
        # Temporary  # 临时处理
        if enable_sgl_per_token_group_quant_8bit:  # 如果启用v2内核
            if enable_v2:  # 使用v2
                sgl_per_token_group_quant_8bit(  # 调用v2内核
                    x,
                    x_q,
                    x_s,
                    group_size,
                    eps,
                    fp8_min,
                    fp8_max,
                    scale_ue8m0,
                    fuse_silu_and_mul,
                    masked_m,
                    enable_v2=True,
                )
            else:  # 使用JIT回退
                sgl_per_token_group_quant_8bit_jit(  # 调用JIT内核
                    input=x,
                    output_q=x_q,
                    output_s=x_s,
                    group_size=group_size,
                    eps=eps,
                    fp8_min=fp8_min,
                    fp8_max=fp8_max,
                    scale_ue8m0=scale_ue8m0,
                )
        else:  # 旧版本内核
            assert not enable_v2  # 不支持v2
            sgl_per_token_group_quant_fp8(  # 调用旧版FP8组量化内核
                x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
            )

    return x_q, x_s  # 返回量化结果和缩放因子


def sglang_per_token_group_quant_fp8_ue8m0(  # SGLang优化的UE8M0缩放逐token组FP8量化
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 组大小
    eps: float = 1e-10,  # 避免除零
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
    """使用UE8M0缩放格式的SGLang优化逐token组FP8量化。"""  # 中文函数说明
    assert (  # 断言
        x.shape[-1] % group_size == 0
    ), f"hidden ({x.shape[-1]}) must be divisible by group_size ({group_size})"  # 最后一维必须能被group_size整除
    assert x.is_contiguous(), "x must be contiguous"  # 断言连续
    assert enable_sgl_per_token_group_quant_8bit, (  # 断言v2内核可用
        "sgl_per_token_group_quant_8bit is required (v2 kernel supports "
        "group_size in {16, 32, 64, 128})"
    )

    *x_batch, x_q_mn, x_q_k = x.shape  # 解包形状
    x_q = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)  # 创建量化输出

    x_s_mn = x_q_mn  # 缩放因子M维度
    x_s_k = x_q_k // group_size  # 缩放因子K维度
    aligned_mn = ceil_align(x_s_mn, 4)  # M维度4对齐
    aligned_k = ceil_align(x_s_k, 4)  # K维度4对齐
    x_s = torch.empty(  # 创建UE8M0缩放因子
        (*x_batch, aligned_k // 4, aligned_mn),  # 打包后的形状
        device=x.device,  # 设备
        dtype=torch.int,  # int32类型
    ).transpose(-1, -2)[..., :x_s_mn, :]  # 转置并裁剪

    if x.shape[0] > 0:  # 如果输入非空
        sgl_per_token_group_quant_8bit(  # 调用v2内核
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            True,  # scale_ue8m0  # 使用UE8M0缩放
            False,  # fuse_silu_and_mul  # 不融合SiLU
            None,  # masked_m  # 无掩码
            enable_v2=True,  # 启用v2
        )

    return x_q, x_s  # 返回量化结果和缩放因子


# TODO maybe unify int8 and fp8 code later  # 后续可能统一INT8和FP8代码
def sglang_per_token_group_quant_8bit(  # SGLang逐token组8bit量化统一入口
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 组大小
    dst_dtype: torch.dtype,  # 目标数据类型
    eps: float = 1e-10,  # 避免除零
    column_major_scales: bool = False,  # 是否列主序缩放
    scale_tma_aligned: bool = False,  # 是否TMA对齐
    scale_ue8m0: bool = False,  # 是否UE8M0缩放
    fuse_silu_and_mul: bool = False,  # 是否融合SiLU和乘法
    masked_m: Optional[torch.Tensor] = None,  # 掩码M
    enable_v2: Optional[bool] = None,  # 是否启用v2
):
    """根据目标数据类型分发到INT8或FP8量化实现。"""  # 中文函数说明
    from sglang.srt.layers.quantization.int8_kernel import (  # 延迟导入INT8量化
        sglang_per_token_group_quant_int8,
    )

    if dst_dtype == torch.int8:  # 如果目标类型为INT8
        assert not column_major_scales  # INT8不支持列主序缩放
        assert not scale_tma_aligned  # INT8不支持TMA对齐
        assert not fuse_silu_and_mul  # INT8不支持融合SiLU
        assert masked_m is None  # INT8不支持掩码
        return sglang_per_token_group_quant_int8(  # 使用INT8量化
            x=x,
            group_size=group_size,
            eps=eps,
            dtype=dst_dtype,
            enable_v2=enable_v2,
        )

    return sglang_per_token_group_quant_fp8(  # 使用FP8量化
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        enable_v2=enable_v2,
    )


def sglang_per_token_quant_fp8(  # SGLang优化的逐token FP8量化函数
    x: torch.Tensor,  # 输入张量
    dtype: torch.dtype = fp8_dtype,  # 输出数据类型
):
    """对输入张量执行逐token的FP8量化。"""  # 中文函数说明
    assert x.is_contiguous(), "`x` is not contiguous"  # 断言连续

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)  # 创建量化输出
    x_s = torch.empty(  # 创建缩放因子
        x.shape[0],  # 行数
        1,  # 列数为1
        device=x.device,  # 设备
        dtype=torch.float32,  # float32类型
    )

    sgl_per_token_quant_fp8(x, x_q, x_s)  # 调用内核执行量化

    return x_q, x_s  # 返回量化结果和缩放因子


if _is_cuda:  # CUDA平台
    per_token_group_quant_fp8 = sglang_per_token_group_quant_fp8  # 使用SGLang优化版本
else:  # 非CUDA平台
    per_token_group_quant_fp8 = _per_token_group_quant_8bit_raw  # 使用Triton原始版本


@triton.jit  # Triton JIT编译装饰器
def _static_quant_fp8(  # 静态FP8量化的Triton内核
    # Pointers to inputs and output  # 输入和输出指针
    y_ptr,  # 输入指针
    y_q_ptr,  # 量化输出指针
    y_s_ptr,  # 缩放因子指针
    y_s_repeat_ptr,  # 重复缩放因子指针（用于广播）
    # Stride of input  # 输入步长
    y_stride,  # 输入行步长
    # Columns of input  # 输入列数
    N,  # 列数
    # Information for float8  # FP8信息
    fp8_min,  # FP8最小值
    fp8_max,  # FP8最大值
    # Meta-parameters  # 元参数
    BLOCK: tl.constexpr,  # 块大小常量
    REPEAT_SCALE: tl.constexpr,  # 是否重复缩放因子常量
):
    """A Triton-accelerated function to perform quantization using the given scale on a
    tensor

    This function converts the tensor values into float8 values.
    """  # Triton加速的静态量化函数，使用给定缩放因子将张量值转换为FP8
    # Map the program id to the row of X and Y it should compute.  # 将程序ID映射到要计算的行
    g_id = tl.program_id(0)  # 获取行ID
    y_ptr += g_id * y_stride  # 移动输入指针
    y_q_ptr += g_id * y_stride  # 移动量化输出指针
    if REPEAT_SCALE:  # 如果需要重复缩放因子
        y_s_repeat_ptr += g_id  # 移动重复缩放指针

    cols = tl.arange(0, BLOCK)  # N <= BLOCK  # 生成列偏移
    mask = cols < N  # 创建掩码

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载输入
    y_s = tl.load(y_s_ptr).to(tl.float32)  # 加载缩放因子
    y_s_inv = 1.0 / y_s  # 计算缩放因子倒数
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)  # 量化并裁剪

    tl.store(y_q_ptr + cols, y_q, mask=mask)  # 存储量化结果
    if REPEAT_SCALE:  # 如果需要重复缩放因子
        tl.store(y_s_repeat_ptr, y_s)  # 存储重复缩放因子


def static_quant_fp8(  # 静态FP8量化函数
    x: torch.Tensor,  # 输入张量
    x_s: torch.Tensor,  # 量化缩放因子
    repeat_scale: bool = False,  # 是否将逐张量缩放广播为逐通道缩放
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化张量和缩放因子
    """Function to perform static quantization using the given scale on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        x_s: The quantization scale.
        repeat_scale: Whether to broadcast per-tensor scale to per-channel scale.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """  # 使用给定缩放因子对输入张量执行静态FP8量化
    assert x.is_contiguous(), "`x` is not contiguous"  # 断言连续
    assert x_s.numel() == 1, "only supports per-tensor scale"  # 断言仅支持逐张量缩放

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)  # 创建量化输出
    M = x.numel() // x.shape[-1]  # 计算行数
    N = x.shape[-1]  # 获取列数
    if repeat_scale:  # 如果需要重复缩放
        x_s_repeat = torch.empty(  # 创建重复缩放张量
            (M, 1),  # 每行一个缩放值
            device=x.device,  # 设备
            dtype=torch.float32,  # float32类型
        )
    else:  # 不需要重复缩放
        x_s_repeat = None  # 设为None

    BLOCK = triton.next_power_of_2(N)  # 计算块大小
    # heuristics for number of warps  # 启发式计算warp数量
    num_warps = min(max(BLOCK // 256, 1), 8)  # warp数量范围1-8
    num_stages = 1  # 流水线阶段数
    _static_quant_fp8[(M,)](  # 调用静态量化内核
        x,
        x_q,
        x_s,
        x_s_repeat,
        N,
        N,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        REPEAT_SCALE=repeat_scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    x_s = x_s_repeat if repeat_scale else x_s  # 选择缩放因子
    return x_q, x_s  # 返回量化结果和缩放因子


@triton.jit  # Triton JIT编译装饰器
def _w8a8_block_fp8_matmul(  # W8A8分块FP8矩阵乘法的Triton内核
    # Pointers to inputs and output  # 输入和输出指针
    A,  # 矩阵A指针
    B,  # 矩阵B指针
    C,  # 输出矩阵C指针
    As,  # A的缩放因子指针
    Bs,  # B的缩放因子指针
    # Shape for matmul  # 矩阵乘法形状
    M,  # M维度
    N,  # N维度
    K,  # K维度
    # Block size for block-wise quantization  # 分块量化的块大小
    group_n,  # N方向块大小
    group_k,  # K方向块大小
    # Stride for inputs and output  # 输入输出步长
    stride_am,  # A的M步长
    stride_ak,  # A的K步长
    stride_bk,  # B的K步长
    stride_bn,  # B的N步长
    stride_cm,  # C的M步长
    stride_cn,  # C的N步长
    stride_As_m,  # As的M步长
    stride_As_k,  # As的K步长
    stride_Bs_k,  # Bs的K步长
    stride_Bs_n,  # Bs的N步长
    # Meta-parameters  # 元参数
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小
    BLOCK_SIZE_N: tl.constexpr,  # N方向块大小
    BLOCK_SIZE_K: tl.constexpr,  # K方向块大小
    GROUP_SIZE_M: tl.constexpr,  # M方向组大小
    needs_masking: tl.constexpr,  # 是否需要掩码
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """  # Triton加速的分块量化矩阵乘法内核

    pid = tl.program_id(axis=0)  # 获取程序ID
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M方向块数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N方向块数
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每组程序数
    group_id = pid // num_pid_in_group  # 组ID
    first_pid_m = group_id * GROUP_SIZE_M  # 组内第一个M块ID
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 组内实际M块数
    pid_m = first_pid_m + (pid % group_size_m)  # 当前M块ID
    pid_n = (pid % num_pid_in_group) // group_size_m  # 当前N块ID

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  # A的M偏移
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # B的N偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # K偏移
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # A指针
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # B指针

    As_ptrs = As + offs_am * stride_As_m  # A缩放因子指针
    offs_bsn = offs_bn // group_n  # B缩放因子N偏移
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n  # B缩放因子指针
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K  # 每个K组的K块数

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  # 遍历K块
        if needs_masking:  # 需要掩码
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)  # 带掩码加载A
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)  # 带掩码加载B
        else:  # 不需要掩码
            a = tl.load(a_ptrs)  # 加载A
            b = tl.load(b_ptrs)  # 加载B

        a_s = tl.load(As_ptrs)  # 加载A缩放因子
        b_s = tl.load(Bs_ptrs)  # 加载B缩放因子

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)  # 计算缩放步进
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]  # 点积并乘以缩放因子
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 移动A指针
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 移动B指针
        As_ptrs += scale_step_k * stride_As_k  # 移动A缩放指针
        Bs_ptrs += scale_step_k * stride_Bs_k  # 移动B缩放指针

    if C.dtype.element_ty == tl.bfloat16:  # 输出为bfloat16
        c = accumulator.to(tl.bfloat16)  # 转换
    elif C.dtype.element_ty == tl.float16:  # 输出为float16
        c = accumulator.to(tl.float16)  # 转换
    else:  # 其他类型
        c = accumulator.to(tl.float32)  # 转换为float32

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # C的M偏移
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # C的N偏移
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]  # C指针
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  # C掩码
    tl.store(c_ptrs, c, mask=c_mask)  # 存储结果


@triton.jit  # Triton JIT编译装饰器
def _w8a8_block_fp8_matmul_unrolledx4(  # W8A8分块FP8矩阵乘法（4倍展开）的Triton内核
    # Pointers to inputs and output  # 输入和输出指针
    A,  # 矩阵A指针
    B,  # 矩阵B指针
    C,  # 输出矩阵C指针
    As,  # A缩放因子指针
    Bs,  # B缩放因子指针
    # Shape for matmul  # 矩阵乘法形状
    M,  # M维度
    N,  # N维度
    K,  # K维度
    # Block size for block-wise quantization  # 分块量化的块大小
    group_n,  # N方向块大小
    group_k,  # K方向块大小
    # Stride for inputs and output  # 输入输出步长
    stride_am,  # A的M步长
    stride_ak,  # A的K步长
    stride_bk,  # B的K步长
    stride_bn,  # B的N步长
    stride_cm,  # C的M步长
    stride_cn,  # C的N步长
    stride_As_m,  # As的M步长
    stride_As_k,  # As的K步长
    stride_Bs_k,  # Bs的K步长
    stride_Bs_n,  # Bs的N步长
    # Meta-parameters  # 元参数
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小
    BLOCK_SIZE_N: tl.constexpr,  # N方向块大小
    BLOCK_SIZE_K: tl.constexpr,  # K方向块大小
    GROUP_SIZE_M: tl.constexpr,  # M方向组大小
    needs_masking: tl.constexpr,  # 是否需要掩码
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """  # Triton加速的分块量化矩阵乘法内核（4倍循环展开）

    pid = tl.program_id(axis=0)  # 获取程序ID
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M方向块数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N方向块数
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每组程序数
    group_id = pid // num_pid_in_group  # 组ID
    first_pid_m = group_id * GROUP_SIZE_M  # 组内第一个M块ID
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 组内实际M块数
    pid_m = first_pid_m + (pid % group_size_m)  # 当前M块ID
    pid_n = (pid % num_pid_in_group) // group_size_m  # 当前N块ID

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  # A的M偏移
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # B的N偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # K偏移
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # A指针
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # B指针

    As_ptrs = As + offs_am * stride_As_m  # A缩放指针
    offs_bsn = offs_bn // group_n  # B缩放N偏移
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n  # B缩放指针
    scale_step_k = BLOCK_SIZE_K // group_k  # 缩放步进

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器
    # manually unroll to 4 iterations  # 手动展开为4次迭代
    UNROLL_FACTOR = 4  # 展开因子
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * UNROLL_FACTOR)):  # 遍历展开后的K块
        # 1st iteration  # 第1次迭代
        if needs_masking:  # 需要掩码
            a = tl.load(  # 加载A
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(  # 加载B
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:  # 不需要掩码
            a = tl.load(a_ptrs)  # 加载A
            b = tl.load(b_ptrs)  # 加载B

        a_s = tl.load(As_ptrs)  # 加载A缩放
        b_s = tl.load(Bs_ptrs)  # 加载B缩放

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]  # 累加
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 移动A指针
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 移动B指针
        As_ptrs += scale_step_k * stride_As_k  # 移动A缩放指针
        Bs_ptrs += scale_step_k * stride_Bs_k  # 移动B缩放指针

        # 2nd iteration  # 第2次迭代
        if needs_masking:  # 需要掩码
            a = tl.load(  # 加载A
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 1) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(  # 加载B
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 1) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:  # 不需要掩码
            a = tl.load(a_ptrs)  # 加载A
            b = tl.load(b_ptrs)  # 加载B

        a_s = tl.load(As_ptrs)  # 加载A缩放
        b_s = tl.load(Bs_ptrs)  # 加载B缩放

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]  # 累加
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 移动A指针
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 移动B指针
        As_ptrs += scale_step_k * stride_As_k  # 移动A缩放指针
        Bs_ptrs += scale_step_k * stride_Bs_k  # 移动B缩放指针

        # 3rd iteration  # 第3次迭代
        if needs_masking:  # 需要掩码
            a = tl.load(  # 加载A
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 2) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(  # 加载B
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 2) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:  # 不需要掩码
            a = tl.load(a_ptrs)  # 加载A
            b = tl.load(b_ptrs)  # 加载B

        a_s = tl.load(As_ptrs)  # 加载A缩放
        b_s = tl.load(Bs_ptrs)  # 加载B缩放

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]  # 累加
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 移动A指针
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 移动B指针
        As_ptrs += scale_step_k * stride_As_k  # 移动A缩放指针
        Bs_ptrs += scale_step_k * stride_Bs_k  # 移动B缩放指针

        # 4th iteration  # 第4次迭代
        if needs_masking:  # 需要掩码
            a = tl.load(  # 加载A
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 3) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(  # 加载B
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 3) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:  # 不需要掩码
            a = tl.load(a_ptrs)  # 加载A
            b = tl.load(b_ptrs)  # 加载B

        a_s = tl.load(As_ptrs)  # 加载A缩放
        b_s = tl.load(Bs_ptrs)  # 加载B缩放

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]  # 累加
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 移动A指针
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 移动B指针
        As_ptrs += scale_step_k * stride_As_k  # 移动A缩放指针
        Bs_ptrs += scale_step_k * stride_Bs_k  # 移动B缩放指针

    if C.dtype.element_ty == tl.bfloat16:  # 输出为bfloat16
        c = accumulator.to(tl.bfloat16)  # 转换
    elif C.dtype.element_ty == tl.float16:  # 输出为float16
        c = accumulator.to(tl.float16)  # 转换
    else:  # 其他
        c = accumulator.to(tl.float32)  # 转换

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # C的M偏移
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # C的N偏移
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]  # C指针
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  # C掩码
    tl.store(c_ptrs, c, mask=c_mask)  # 存储结果


@functools.lru_cache  # LRU缓存装饰器
def get_w8a8_block_fp8_configs(  # 获取W8A8分块FP8内核的优化配置
    N: int, K: int, block_n: int, block_k: int  # 矩阵维度和块大小
) -> Optional[Dict[int, Any]]:  # 返回配置字典或None
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """  # 返回W8A8分块FP8内核的优化配置

    # Skip config lookup during torch.compile to avoid non-Tensor ops (e.g., device name).
    # Returning None forces the caller to use the default config path during compile.  # torch.compile时跳过配置查找
    if torch._dynamo.is_compiling():  # 如果正在编译
        return None  # 返回None使用默认配置

    # First look up if an optimized configuration is available in the configs
    # directory  # 首先查找配置目录中的优化配置
    device_name = get_device_name().replace(" ", "_")  # 获取设备名称
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json"  # 构建配置文件名

    config_file_path = os.path.join(  # 构建完整路径
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):  # 如果配置文件存在
        with open(config_file_path) as f:  # 打开文件
            log_info_on_rank0(  # 在rank0记录信息
                logger,
                f"Using configuration from {config_file_path} for W8A8 Block FP8 kernel.",
            )
            raw = {int(key): val for key, val in json.load(f).items()}  # 解析JSON

        sanitized = {}  # 清理后的配置
        clamped_ms = []  # 被钳制的M值列表
        for m_key, cfg in raw.items():  # 遍历配置
            if cfg["BLOCK_SIZE_K"] < block_k:  # 如果BLOCK_SIZE_K小于block_k
                clamped_ms.append((m_key, cfg["BLOCK_SIZE_K"]))  # 记录
                cfg = {**cfg, "BLOCK_SIZE_K": block_k}  # 钳制到block_k
            sanitized[m_key] = cfg  # 保存
        if clamped_ms:  # 如果有被钳制的值
            logger.warning(  # 记录警告
                "Clamped BLOCK_SIZE_K up to %d in tuned config %s for entries %s "
                "(scale stepping requires BLOCK_SIZE_K >= block_k).",
                block_k,
                json_file_name,
                clamped_ms,
            )

        return sanitized  # 返回清理后的配置

    # If no optimized configuration is available, we will use the default
    # configuration  # 没有优化配置时使用默认配置
    logger.warning(  # 记录警告
        (
            "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! "
            "Config file not found at %s"
        ),
        config_file_path,
    )
    return None  # 返回None使用默认配置


def select_w8a8_block_fp8_matmul_kernel(M, N, META):  # 选择W8A8分块FP8矩阵乘法内核
    """选择标准的W8A8分块FP8矩阵乘法内核。"""  # 中文函数说明
    return _w8a8_block_fp8_matmul  # 返回标准内核


if _is_hip:  # HIP平台

    def use_w8a8_block_fp8_matmul_unrolledx4(M, N, META):  # 判断是否使用4倍展开内核
        # Use manually unrolledx4 kernel on AMD GPU when the grid size is small.
        # Empirical testing shows the sweet spot lies when it's less than the # of
        # compute units available on the device.  # 当网格大小小于计算单元数时使用展开内核
        num_workgroups = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(  # 计算工作组数
            N, META["BLOCK_SIZE_N"]
        )
        num_workgroups <= get_device_core_count()  # 比较与计算单元数

    def select_w8a8_block_fp8_matmul_kernel(M, N, META):  # HIP平台选择内核
        """在HIP平台上根据网格大小选择标准或4倍展开内核。"""  # 中文函数说明
        if use_w8a8_block_fp8_matmul_unrolledx4(M, N, META):  # 如果适合使用展开内核
            return _w8a8_block_fp8_matmul_unrolledx4  # 返回展开内核
        else:  # 否则
            return _w8a8_block_fp8_matmul  # 返回标准内核


def prepare_block_fp8_matmul_inputs(  # 准备分块FP8矩阵乘法的输入
    A: torch.Tensor,  # 矩阵A
    B: torch.Tensor,  # 矩阵B
    As: torch.Tensor,  # A缩放因子
    Bs: torch.Tensor,  # B缩放因子
    block_size: List[int],  # 块大小
    output_dtype: torch.dtype = torch.float16,  # 输出数据类型
) -> Tuple[int, int, int]:  # 返回M, N, K和输出张量C
    """验证并准备分块FP8矩阵乘法的输入，创建输出张量。"""  # 中文函数说明
    assert len(block_size) == 2  # 块大小必须为2维
    block_n, block_k = block_size[0], block_size[1]  # 解包块大小

    assert A.shape[-1] == B.shape[-1]  # A和B的内积维度必须一致
    assert A.shape[:-1] == As.shape[:-1]  # A和As的前缀形状必须一致
    assert A.is_contiguous()  # A必须连续

    if As.dtype == torch.float:  # 如果A缩放因子为float32
        assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]  # 验证As列数
    elif As.dtype == torch.int:  # 如果A缩放因子为int32（UE8M0打包）
        assert (
            triton.cdiv(triton.cdiv(A.shape[-1], block_k), 4) == As.shape[-1]
        ), f"{A.shape=} {As.shape=} {block_size=}"  # 验证打包后的列数
    else:  # 不支持的类型
        raise NotImplementedError  # 抛出异常

    M = A.numel() // A.shape[-1]  # 计算A的行数

    assert B.ndim == 2  # B必须为2维
    assert B.is_contiguous()  # B必须连续
    assert Bs.ndim == 2  # Bs必须为2维
    N, K = B.shape  # 获取B的形状

    if Bs.dtype == torch.float:  # B缩放因子为float32
        assert triton.cdiv(N, block_n) == Bs.shape[0]  # 验证Bs行数
        assert triton.cdiv(K, block_k) == Bs.shape[1]  # 验证Bs列数
    elif Bs.dtype == torch.int:  # B缩放因子为int32
        assert N == Bs.shape[0], f"{B.shape=} {Bs.shape=} {block_size=}"  # 验证Bs行数
        assert (
            triton.cdiv(triton.cdiv(K, block_k), 4) == Bs.shape[1]
        ), f"{B.shape=} {Bs.shape=} {block_size=}"  # 验证Bs列数
    else:  # 不支持的类型
        raise NotImplementedError  # 抛出异常

    C_shape = A.shape[:-1] + (N,)  # 计算输出形状
    C = A.new_empty(C_shape, dtype=output_dtype)  # 创建输出张量

    return M, N, K, C  # 返回维度和输出张量


def w8a8_block_fp8_matmul_deepgemm(  # 使用DeepGEMM的W8A8分块FP8矩阵乘法
    A: torch.Tensor,  # 矩阵A
    B: torch.Tensor,  # 矩阵B
    As: torch.Tensor,  # A缩放因子
    Bs: torch.Tensor,  # B缩放因子
    block_size: List[int],  # 块大小
    output_dtype: torch.dtype,  # 输出数据类型
) -> torch.Tensor:  # 返回矩阵乘法结果
    """使用DeepGEMM后端执行W8A8分块FP8矩阵乘法。"""  # 中文函数说明
    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)  # 准备输入

    # Deepgemm only supports output tensor type as bfloat16  # DeepGEMM仅支持bfloat16输出
    assert C.dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM  # 断言

    deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, C)  # 调用DeepGEMM内核

    return C  # 返回结果


def w8a8_block_fp8_matmul_triton(  # 使用Triton的W8A8分块FP8矩阵乘法
    A: torch.Tensor,  # 矩阵A
    B: torch.Tensor,  # 矩阵B
    As: torch.Tensor,  # A缩放因子
    Bs: torch.Tensor,  # B缩放因子
    block_size: List[int],  # 块大小
    output_dtype: torch.dtype = torch.float16,  # 输出数据类型
) -> torch.Tensor:  # 返回矩阵乘法结果
    """This function performs matrix multiplication with block-wise quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of matmul.
    """  # 使用Triton执行分块量化矩阵乘法

    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)  # 准备输入

    block_n, block_k = block_size  # 解包块大小

    configs = get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])  # 获取优化配置
    if configs:  # 有优化配置
        # If an optimal configuration map has been found, look up the
        # optimal config  # 查找最优配置
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]  # 选择最接近M的配置
    else:  # 无优化配置
        # Default config  # 默认配置
        # Block-wise quant: BLOCK_SIZE_K must be divisible by block_size[1]  # BLOCK_SIZE_K必须能被block_k整除
        config = {
            "BLOCK_SIZE_M": 64,  # M方向块大小64
            "BLOCK_SIZE_N": block_size[0],  # N方向块大小
            "BLOCK_SIZE_K": block_size[1],  # K方向块大小
            "GROUP_SIZE_M": 32,  # M方向组大小32
            "num_warps": 4,  # 4个warp
            "num_stages": 3,  # 3个流水线阶段
        }

    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)  # 是否需要K方向掩码

    def grid(META):  # 定义网格大小函数
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    kernel = select_w8a8_block_fp8_matmul_kernel(M, N, config)  # 选择内核

    kernel[grid](  # 调用内核
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=needs_masking,
    )

    return C  # 返回结果


# universal entry point, for testing purposes  # 通用入口点，用于测试
def w8a8_block_fp8_matmul(  # W8A8分块FP8矩阵乘法统一入口
    A: torch.Tensor,  # 矩阵A
    B: torch.Tensor,  # 矩阵B
    As: torch.Tensor,  # A缩放因子
    Bs: torch.Tensor,  # B缩放因子
    block_size: List[int],  # 块大小
    output_dtype: torch.dtype = torch.float16,  # 输出数据类型
) -> torch.Tensor:  # 返回矩阵乘法结果
    """W8A8分块FP8矩阵乘法的统一入口，自动选择DeepGEMM或Triton后端。"""  # 中文函数说明
    if output_dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:  # 如果支持DeepGEMM
        return w8a8_block_fp8_matmul_deepgemm(  # 使用DeepGEMM
            A, B, As, Bs, block_size, output_dtype=output_dtype
        )

    return w8a8_block_fp8_matmul_triton(  # 使用Triton
        A, B, As, Bs, block_size, output_dtype=output_dtype
    )


# Copied and adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py  # 参考自Triton教程
@triton.jit  # Triton JIT编译装饰器
def _mxfp8_block_scaled_matmul_kernel(  # MXFP8分块缩放矩阵乘法的Triton内核
    a_desc,  # a  # A张量描述符
    a_scale_desc,  # a_scale  # A缩放因子描述符
    b_desc,  # b  # B张量描述符
    b_scale_desc,  # b_scale  # B缩放因子描述符
    c_desc,  # c  # C输出描述符
    M: tl.constexpr,  # M维度常量
    N: tl.constexpr,  # N维度常量
    K: tl.constexpr,  # K维度常量
    output_type: tl.constexpr,  # 输出类型常量
    BLOCK_M: tl.constexpr,  # M方向块大小
    BLOCK_N: tl.constexpr,  # N方向块大小
    BLOCK_K: tl.constexpr,  # K方向块大小
    rep_m: tl.constexpr,  # M方向重复数
    rep_n: tl.constexpr,  # N方向重复数
    rep_k: tl.constexpr,  # K方向重复数
    NUM_STAGES: tl.constexpr,  # 流水线阶段数
):  #
    if output_type == 0:  # 输出类型0
        output_dtype = tl.float32  # float32
    elif output_type == 1:  # 输出类型1
        output_dtype = tl.float16  # float16
    elif output_type == 2:  # 输出类型2
        output_dtype = tl.bfloat16  # bfloat16

    pid = tl.program_id(axis=0)  # 获取程序ID
    num_pid_m = tl.cdiv(M, BLOCK_M)  # M方向块数
    pid_m = pid % num_pid_m  # M方向块ID
    pid_n = pid // num_pid_m  # N方向块ID
    offs_am = pid_m * BLOCK_M  # A的M偏移
    offs_bn = pid_n * BLOCK_N  # B的N偏移
    offs_k_a = 0  # A的K偏移初始化
    offs_k_b = 0  # B的K偏移初始化
    offs_scale_m = pid_m * rep_m  # 缩放M偏移
    offs_scale_n = pid_n * rep_n  # 缩放N偏移
    offs_scale_k = 0  # 缩放K偏移

    VEC_SIZE: tl.constexpr = 32  # 向量大小常量

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # 初始化累加器
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):  # 遍历K块
        a = a_desc.load([offs_am, offs_k_a])  # 加载A
        b = b_desc.load([offs_bn, offs_k_b])  # 加载B
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])  # 加载A缩放
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])  # 加载B缩放

        scale_a = (  # 重排A缩放因子
            scale_a.reshape(rep_m, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        )
        scale_b = (  # 重排B缩放因子
            scale_b.reshape(rep_n, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
        )

        accumulator = tl.dot_scaled(  # 执行缩放点积
            a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator
        )

        offs_k_a += BLOCK_K  # 更新A的K偏移
        offs_k_b += BLOCK_K  # 更新B的K偏移
        offs_scale_k += rep_k  # 更新缩放K偏移

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))  # 存储输出


# Copied and adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py  # 参考自Triton教程
def mxfp8_block_scaled_matmul_triton(  # MXFP8分块缩放矩阵乘法的Triton实现
    a: torch.Tensor,  # 矩阵A
    a_scale: torch.Tensor,  # A缩放因子
    b: torch.Tensor,  # 矩阵B
    b_scale: torch.Tensor,  # B缩放因子
    output_dtype: torch.dtype,  # 输出数据类型
    *,  # 以下为关键字参数
    block_m: int = 128,  # M方向块大小
    block_n: int = 256,  # N方向块大小
    block_k: int = 128,  # K方向块大小
    num_stages: Optional[int] = None,  # 流水线阶段数
) -> torch.Tensor:  # 返回矩阵乘法结果
    """Block-scaled matmul for MXFP8 using Triton dot_scaled.

    Args:
        num_stages: Number of pipeline stages. If None, auto-selects based on GPU:
            SM120: 1, SM100: 4.
    """  # 使用Triton dot_scaled的MXFP8分块缩放矩阵乘法
    if num_stages is None:  # 未指定阶段数
        num_stages = 1 if _is_sm120_supported else (4 if _is_sm100_supported else 1)  # 自动选择
    M, K = a.shape  # 获取A的形状
    N, K_b = b.shape  # 获取B的形状
    assert K == K_b  # K维度必须一致

    if output_dtype == torch.float32:  # 输出类型映射
        output_type = 0  # 0对应float32
    elif output_dtype == torch.float16:  # float16
        output_type = 1  # 1对应float16
    elif output_dtype == torch.bfloat16:  # bfloat16
        output_type = 2  # 2对应bfloat16
    else:  # 不支持的类型
        raise ValueError(f"Unsupported output dtype: {output_dtype}")  # 抛出错误

    rep_m = block_m // 128  # M方向重复数
    rep_n = block_n // 128  # N方向重复数
    rep_k = block_k // 32 // 4  # K方向重复数

    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])  # 创建A描述符
    b_desc = TensorDescriptor.from_tensor(b, [block_n, block_k])  # 创建B描述符

    scale_block_shape = [1, rep_m, rep_k, 2, 256]  # 缩放块形状
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=scale_block_shape)  # A缩放描述符
    scale_block_shape = [1, rep_n, rep_k, 2, 256]  # 缩放块形状
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=scale_block_shape)  # B缩放描述符

    output = torch.empty((M, N), dtype=output_dtype, device=a.device)  # 创建输出张量
    c_desc = TensorDescriptor.from_tensor(output, [block_m, block_n])  # C描述符

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), 1)  # 网格大小
    _mxfp8_block_scaled_matmul_kernel[grid](  # 调用MXFP8内核
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        output_type,
        block_m,
        block_n,
        block_k,
        rep_m,
        rep_n,
        rep_k,
        num_stages,
    )
    return output  # 返回结果


@triton.jit  # Triton JIT编译装饰器
def _per_tensor_quant_mla_fp8_stage1(  # 逐张量MLA FP8量化的第一阶段内核（计算最大值）
    x_ptr,  # 输入指针
    x_s_ptr,  # 缩放因子指针
    head_size,  # 头大小
    x_stride_h,  # 头步长
    x_stride_s,  # 序列步长
    eps,  # 最小值阈值
    fp8_max,  # FP8最大值
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """MLA FP8逐张量量化的第一阶段：计算每个头的绝对值最大值。"""  # 中文函数说明
    seq_id = tl.program_id(0)  # 序列ID
    head_id = tl.program_id(1)  # 头ID
    offset = tl.arange(0, BLOCK_SIZE)  # 偏移量
    mask = offset < head_size  # 掩码

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s  # 移动输入指针
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)  # 加载数据
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)  # 计算绝对值最大值

    tl.atomic_max(x_s_ptr, _absmax / fp8_max)  # 原子最大操作更新缩放因子


@triton.jit  # Triton JIT编译装饰器
def _per_tensor_quant_mla_fp8_stage2(  # 逐张量MLA FP8量化的第二阶段内核（执行量化）
    x_ptr,  # 输入指针
    x_s_ptr,  # 缩放因子指针
    x_q_ptr,  # 量化输出指针
    num_seq,  # 序列数
    head_size,  # 头大小
    x_stride_h,  # 头步长
    x_stride_s,  # 序列步长
    fp8_min,  # FP8最小值
    fp8_max,  # FP8最大值
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """MLA FP8逐张量量化的第二阶段：使用全局缩放因子执行量化。"""  # 中文函数说明
    seq_id = tl.program_id(0)  # 序列ID
    head_id = tl.program_id(1)  # 头ID
    offset = tl.arange(0, BLOCK_SIZE)  # 偏移量
    mask = offset < head_size  # 掩码

    x_s = tl.load(x_s_ptr)  # 加载全局缩放因子
    x_s_inv = 1.0 / x_s  # 计算倒数

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s  # 移动输入指针
    x_q_ptr += head_id * num_seq * head_size + seq_id * head_size  # 移动输出指针

    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)  # 加载数据
    x_q = tl.clamp(x * x_s_inv, fp8_min, fp8_max).to(x_q_ptr.dtype.element_ty)  # 量化
    tl.store(x_q_ptr + offset, x_q, mask=mask)  # 存储结果


def per_tensor_quant_mla_fp8(  # MLA专用的逐张量FP8量化函数
    x: torch.Tensor, x_s_out: torch.Tensor, eps: float = 1e-12  # 输入张量、缩放因子输出、最小值
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
    """
    This function quantizes input values to float8 values with tensor-wise quantization
    and specialized for mla absorbed case.
    """  # 将输入值量化为FP8，使用逐张量量化，专为MLA吸收情况优化
    assert x.dim() == 3, "`x` is not a 3d-tensor"  # 断言3维
    assert (  # 断言缩放因子形状
        x_s_out.shape == (1,)
        and x_s_out.dtype == torch.float32
        and x_s_out.device == x.device
    )

    x_q = x.new_empty(x.size(), dtype=fp8_dtype)  # 创建量化输出

    num_head, num_seq, head_size = x.shape  # 解包形状
    BLOCK_SIZE = triton.next_power_of_2(head_size)  # 计算块大小
    grid = (num_seq, num_head)  # 网格大小

    _per_tensor_quant_mla_fp8_stage1[grid](  # 第一阶段：计算最大值
        x,
        x_s_out,
        head_size,
        x.stride(0),
        x.stride(1),
        eps,
        fp8_max,
        BLOCK_SIZE,
    )
    _per_tensor_quant_mla_fp8_stage2[grid](  # 第二阶段：执行量化
        x,
        x_s_out,
        x_q,
        num_seq,
        head_size,
        x.stride(0),
        x.stride(1),
        fp8_min,
        fp8_max,
        BLOCK_SIZE,
    )

    return x_q, x_s_out  # 返回结果


@triton.jit  # Triton JIT编译装饰器
def _per_token_group_quant_mla_deep_gemm_masked_fp8(  # MLA DeepGEMM掩码逐token组FP8量化的Triton内核
    y_ptr,  # 输入指针
    y_q_ptr,  # 量化输出指针
    y_s_ptr,  # 缩放因子指针
    masked_m_ptr,  # 掩码M指针
    group_size,  # 组大小
    y_stride_b,  # batch步长
    y_stride_t,  # token步长
    y_q_stride_b,  # 量化输出batch步长
    y_q_stride_t,  # 量化输出token步长
    y_s_stride_b,  # 缩放batch步长
    y_s_stride_g,  # 缩放组步长
    eps,  # 最小值
    fp8_min,  # FP8最小值
    fp8_max,  # FP8最大值
    NUM_GROUP: tl.constexpr,  # 组数量常量
    BLOCK: tl.constexpr,  # 块大小常量
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor for deep_gemm grouped_gemm_masked.
    This function converts the tensor values into float8 values.
    y and y_q: (b, t, k)
    y_s: (b, k//group_size, t)
    """  # 为DeepGEMM grouped_gemm_masked优化的逐token组量化内核
    t_id = tl.program_id(0)  # token ID
    b_id = tl.program_id(1)  # batch ID

    y_ptr += b_id * y_stride_b + t_id * y_stride_t  # 移动输入指针
    y_q_ptr += b_id * y_q_stride_b + t_id * y_q_stride_t  # 移动量化输出指针
    y_s_ptr += b_id * y_s_stride_b + t_id  # 移动缩放指针

    if t_id == 0:  # 第一个token
        tl.store(masked_m_ptr + b_id, tl.num_programs(0))  # 存储token数

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK  # 列偏移
    mask = cols < group_size  # 掩码

    for gid in range(NUM_GROUP):  # 遍历每个组
        y = tl.load(y_ptr + gid * group_size + cols, mask=mask, other=0.0).to(  # 加载数据
            tl.float32
        )
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)  # 计算绝对值最大值
        y_s = _absmax / fp8_max  # 计算缩放因子
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)  # 量化

        tl.store(y_q_ptr + gid * group_size + cols, y_q, mask=mask)  # 存储量化结果
        tl.store(y_s_ptr + gid * y_s_stride_g, y_s)  # 存储缩放因子


def per_token_group_quant_mla_deep_gemm_masked_fp8(  # MLA DeepGEMM掩码逐token组FP8量化
    x: torch.Tensor,  # 输入张量
    group_size: int = 128,  # 组大小，默认128
    eps: float = 1e-12,  # 最小值
    dtype: torch.dtype = fp8_dtype,  # 输出数据类型
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
    """
    This function quantizes input values to float8 values with per-token-group-quantization
    for deep_gemm grouped_gemm_masked and specialized for mla absorbed case.
    """  # 为DeepGEMM grouped_gemm_masked优化的逐token组FP8量化
    assert x.dim() == 3, "`x` is not a 3d-tensor"  # 断言3维

    b, m, k = x.shape  # 解包形状
    aligned_m = (m + 255) // 256 * 256  # 256 is the max block_m of the gemm kernel  # 对齐到256的倍数
    num_tiles_k = k // group_size  # K方向的组数
    assert num_tiles_k * group_size == k, f"k % {group_size} must be zero"  # K必须能被group_size整除

    x_q = x.new_empty((b, aligned_m, k), dtype=dtype)  # 创建量化输出
    x_s = x.new_empty((b, num_tiles_k, aligned_m), dtype=torch.float32)  # 创建缩放因子
    masked_m = x.new_empty((b,), dtype=torch.int32)  # 创建掩码M

    BLOCK_SIZE = triton.next_power_of_2(group_size)  # 计算块大小
    grid = (m, b)  # 网格大小

    _per_token_group_quant_mla_deep_gemm_masked_fp8[grid](  # 调用内核
        x,
        x_q,
        x_s,
        masked_m,
        group_size,
        x.stride(0),
        x.stride(1),
        x_q.stride(0),
        x_q.stride(1),
        x_s.stride(0),
        x_s.stride(1),
        eps,
        -fp8_max,
        fp8_max,
        num_tiles_k,
        BLOCK_SIZE,
    )

    return x_q, x_s.transpose(1, 2), masked_m, m, aligned_m  # 返回结果


"""
Quantize input tensor to FP8 (8-bit floating point) format.

Args:
    input (torch.Tensor): Input tensor to be quantized
    scale (Optional[torch.Tensor]): Pre-computed scaling factor for static quantization.
        If None, scales will be computed dynamically.
    num_token_padding (Optional[int]): If specified, pad the first dimension
        of the output to at least this value.
    use_per_token_if_dynamic (bool): When using dynamic scaling (scale=None),
        determines the quantization granularity:
        - True: compute scale per token
        - False: compute single scale per tensor

Returns:
    Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - quantized_tensor: The FP8 quantized version of input
        - scale_tensor: The scaling factors used for quantization

Raises:
    AssertionError: If input is not 2D or if static scale's numel != 1
"""  # 将输入张量量化为FP8格式的文档字符串
if _is_hip:  # HIP平台

    def _native_dynamic_per_token_quant_fp8(output, input, scale):  # HIP平台原生逐token动态FP8量化
        """Native PyTorch fallback for dynamic per-token FP8 quantization when vLLM is unavailable."""  # vLLM不可用时的PyTorch原生回退实现
        M, N = input.shape  # 获取形状
        eps = 1e-12  # 最小值
        # Compute per-token scale  # 计算逐token缩放因子
        absmax = input.abs().max(dim=1, keepdim=True).values  # 逐token绝对值最大值
        absmax = torch.clamp(absmax, min=eps)  # 钳制最小值
        scale_val = absmax / fp8_max  # 计算缩放因子
        scale.copy_(scale_val)  # 复制到输出
        # Quantize  # 量化
        output_data = torch.clamp(input / scale_val, fp8_min, fp8_max).to(fp8_dtype)  # 量化并裁剪
        output.copy_(output_data)  # 复制到输出

    def _native_dynamic_per_tensor_quant_fp8(output, input, scale):  # HIP平台原生逐张量动态FP8量化
        """Native PyTorch fallback for dynamic per-tensor FP8 quantization when vLLM is unavailable."""  # vLLM不可用时的PyTorch原生回退实现
        eps = 1e-12  # 最小值
        absmax = input.abs().max()  # 计算全局绝对值最大值
        absmax = torch.clamp(absmax, min=eps)  # 钳制最小值
        scale_val = absmax / fp8_max  # 计算缩放因子
        # Use copy_ instead of fill_ with .item() to avoid CPU-GPU sync  # 使用copy_而非fill_避免CPU-GPU同步
        scale.view(-1).copy_(scale_val.view(-1))  # 复制缩放因子
        # Quantize  # 量化
        output_data = torch.clamp(input / scale_val, fp8_min, fp8_max).to(fp8_dtype)  # 量化
        output.copy_(output_data)  # 复制到输出

    def _native_static_quant_fp8(output, input, scale):  # HIP平台原生静态FP8量化
        """Native PyTorch fallback for static FP8 quantization when vLLM is unavailable."""  # vLLM不可用时的PyTorch原生回退实现
        # Use tensor directly instead of .item() to avoid CPU-GPU sync  # 直接使用张量而非.item()避免CPU-GPU同步
        output_data = torch.clamp(input / scale, fp8_min, fp8_max).to(fp8_dtype)  # 量化
        output.copy_(output_data)  # 复制到输出

    def scaled_fp8_quant(  # HIP平台FP8量化统一入口
        input: torch.Tensor,  # 输入张量
        scale: Optional[torch.Tensor] = None,  # 预计算缩放因子
        num_token_padding: Optional[int] = None,  # token填充数
        use_per_token_if_dynamic: bool = False,  # 动态量化时是否逐token
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
        """HIP平台的FP8量化统一入口，支持动态和静态量化。"""  # 中文函数说明
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"  # 断言2维
        shape = input.shape  # 获取形状
        if num_token_padding:  # 如果需要填充
            shape = (max(num_token_padding, input.shape[0]), shape[1])  # 填充第一维
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)  # 创建输出

        if scale is None:  # 动态量化
            # Dynamic scaling  # 动态缩放
            if use_per_token_if_dynamic:  # 逐token量化
                scale = torch.empty(  # 创建缩放因子
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                if _use_aiter:  # 使用AITER
                    dynamic_per_token_scaled_quant(output, input, scale)  # AITER逐token量化
                elif _has_vllm:  # 使用vLLM
                    torch.ops._C.dynamic_per_token_scaled_fp8_quant(  # vLLM逐token量化
                        output, input.contiguous(), scale, None
                    )
                else:  # PyTorch原生回退
                    _native_dynamic_per_token_quant_fp8(output, input, scale)  # 原生逐token量化
            else:  # 逐张量量化
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)  # 创建缩放因子
                if _use_aiter:  # 使用AITER
                    dynamic_per_tensor_quant(output, input, scale)  # AITER逐张量量化
                elif _has_vllm:  # 使用vLLM
                    torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)  # vLLM逐张量量化
                else:  # PyTorch原生回退
                    _native_dynamic_per_tensor_quant_fp8(output, input, scale)  # 原生逐张量量化
        else:  # 静态量化
            # Static scaling  # 静态缩放
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"  # 断言标量缩放因子
            if _use_aiter:  # 使用AITER
                static_per_tensor_quant(output, input, scale)  # AITER静态量化
            elif _has_vllm:  # 使用vLLM
                torch.ops._C.static_scaled_fp8_quant(output, input, scale)  # vLLM静态量化
            else:  # PyTorch原生回退
                _native_static_quant_fp8(output, input, scale)  # 原生静态量化

        return output, scale  # 返回结果

else:  # 非HIP平台

    def scaled_fp8_quant(  # 非HIP平台FP8量化统一入口
        input: torch.Tensor,  # 输入张量
        scale: Optional[torch.Tensor] = None,  # 预计算缩放因子
        num_token_padding: Optional[int] = None,  # token填充数
        use_per_token_if_dynamic: bool = False,  # 动态量化时是否逐token
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
        """非HIP平台的FP8量化统一入口，使用sgl_kernel实现。"""  # 中文函数说明

        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"  # 断言2维
        shape = input.shape  # 获取形状
        if num_token_padding:  # 如果需要填充
            shape = (max(num_token_padding, input.shape[0]), shape[1])  # 填充第一维
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)  # 创建输出

        if scale is None:  # 动态量化
            # Dynamic scaling  # 动态缩放
            if use_per_token_if_dynamic:  # 逐token量化
                scale = torch.empty(  # 创建缩放因子
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                sgl_per_token_quant_fp8(input, output, scale)  # 调用逐token量化内核
            else:  # 逐张量量化
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)  # 创建缩放因子
                sgl_per_tensor_quant_fp8(  # 调用逐张量量化内核
                    input, output, scale, is_static=False
                )  # False for dynamic  # False表示动态量化
        else:  # 静态量化
            # Static scaling  # 静态缩放
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"  # 断言标量缩放因子
            sgl_per_tensor_quant_fp8(  # 调用静态量化内核
                input, output, scale, is_static=True
            )  # True for static  # True表示静态量化

        return output, scale  # 返回结果


fp8_autotune = triton.autotune(  # FP8自动调优装饰器
    configs=[  # 配置列表
        triton.Config({"BLOCK_M": block_m}, num_warps=num_warps)  # 每种block_m和num_warps的组合
        for block_m in [16, 32, 64, 128]  # M方向块大小候选
        for num_warps in [2, 4, 8]  # warp数量候选
    ],
    key=["K", "BLOCK_K", "M_ALIGNMENT"],  # 调优键
)


@triton.jit  # Triton JIT编译装饰器
def _per_token_group_quant_fp8_hopper_moe_mn_major(  # Hopper MoE MN主序逐token组FP8量化的Triton内核
    a,  # (M, K):(K, 1)  # 输入激活
    expert_offsets,  # (num_experts,)  # 专家偏移
    problem_sizes,  # (num_experts, 3)  # 问题大小
    a_fp8,  # (M, K):(K, 1)  # FP8输出
    sfa,  # (M, k)  # 缩放因子输出
    K: tl.constexpr,  # K维度常量
    BLOCK_K: tl.constexpr,  # K方向块大小常量
    M_ALIGNMENT: tl.constexpr,  # M对齐常量
    BLOCK_M: tl.constexpr,  # tune  # M方向块大小（可调优）
):
    """Hopper MoE专用的MN主序逐token组FP8量化内核。"""  # 中文函数说明
    k_offset = tl.program_id(0)  # K偏移ID
    expert_id = tl.program_id(1)  # 专家ID

    m = tl.load(problem_sizes + expert_id * 3)  # 加载M维度
    current_expert_offset = tl.load(expert_offsets + expert_id).to(tl.int64)  # 加载专家偏移
    tl.multiple_of(m, M_ALIGNMENT)  # 对齐提示
    tl.multiple_of(current_expert_offset, M_ALIGNMENT)  # 对齐提示

    coord_k = k_offset * BLOCK_K + tl.arange(0, BLOCK_K)  # K坐标
    for i in tl.range(tl.cdiv(m, BLOCK_M)):  # 遍历M块
        coord_m = i * BLOCK_M + tl.arange(0, BLOCK_M)  # M坐标
        a_ptrs = a + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]  # A指针
        a_mask = (coord_m < m)[:, None] & (coord_k < K)[None, :]  # 掩码

        inp = tl.load(a_ptrs, mask=a_mask).to(tl.float32)  # [BLOCK_M, BLOCK_K]  # 加载并转换为float32
        inp_amax = tl.max(tl.abs(inp), axis=1)  # [BLOCK_M,]  # 计算绝对值最大值
        inp_amax = tl.clamp(inp_amax, min=1e-4, max=float("inf"))  # 钳制范围
        inp_fp8 = (inp * (448.0 / inp_amax[:, None])).to(tl.float8e4nv)  # 量化为FP8

        # Store fp8  # 存储FP8结果
        a_fp8_ptrs = (
            a_fp8 + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]
        )
        tl.store(a_fp8_ptrs, inp_fp8, mask=a_mask)  # 存储

        # Store sfa  # 存储缩放因子
        k = tl.cdiv(K, BLOCK_K)  # K方向块数
        sfa_ptrs = (
            sfa + current_expert_offset * k + k_offset * m + coord_m
        )  # MN-Major with sfa  # MN主序布局
        tl.store(sfa_ptrs, inp_amax / 448.0, mask=coord_m < m)  # 存储缩放因子


if not _is_cpu:  # 非CPU平台
    _per_token_group_quant_fp8_hopper_moe_mn_major = fp8_autotune(  # 应用自动调优
        _per_token_group_quant_fp8_hopper_moe_mn_major
    )


def per_token_group_quant_fp8_hopper_moe_mn_major(  # Hopper MoE MN主序逐token组FP8量化
    A: torch.Tensor,  # 输入激活
    expert_offsets: torch.Tensor,  # 专家偏移
    problem_sizes: torch.Tensor,  # 问题大小
    group_size: int,  # 组大小
    expert_tokens_alignment: int = 1,  # 专家token对齐
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
    """Hopper MoE专用的MN主序逐token组FP8量化函数。"""  # 中文函数说明
    assert A.dim() == 2  # 断言2维
    assert A.is_contiguous(), "`A` is not contiguous"  # 断言连续
    assert (  # 断言
        A.shape[-1] % group_size == 0
    ), "the last dimension of `A` cannot be divisible by `group_size`"  # 最后一维必须能被group_size整除

    a_q = torch.empty_like(A, device=A.device, dtype=fp8_dtype)  # 创建量化输出
    M, K = A.shape[0], A.shape[1]  # 获取形状
    k = K // group_size  # K方向组数
    sfa = torch.empty((M, k), device=A.device, dtype=torch.float32)  # 创建缩放因子
    num_experts = problem_sizes.shape[0]  # 获取专家数
    grid = (k, num_experts)  # 网格大小
    _per_token_group_quant_fp8_hopper_moe_mn_major[grid](  # 调用内核
        A,
        expert_offsets,
        problem_sizes,
        a_q,
        sfa,
        K,
        group_size,
        expert_tokens_alignment,
    )
    return a_q, sfa  # 返回结果


@triton.jit  # Triton JIT编译装饰器
def _per_group_transpose(  # 逐组转置的Triton内核
    data_ptr: torch.Tensor,  # 输入数据指针
    trans_data_ptr: torch.Tensor,  # 转置输出指针
    expert_offsets: torch.Tensor,  # 专家偏移
    k: int,  # K维度
    M_ALIGNMENT: tl.constexpr,  # M对齐常量
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小
    BLOCK_SIZE_K: tl.constexpr,  # K方向块大小
):
    """在MoE场景中按专家组进行数据转置。"""  # 中文函数说明
    expert_id = tl.program_id(0)  # 专家ID
    m_id = tl.program_id(1)  # M方向ID
    k_id = tl.program_id(2)  # K方向ID

    curr_expert_offset = tl.load(expert_offsets + expert_id)  # 当前专家偏移
    next_expert_offset = tl.load(expert_offsets + expert_id + 1)  # 下一个专家偏移
    num_tokens_of_expert = next_expert_offset - curr_expert_offset  # 当前专家token数
    tl.multiple_of(curr_expert_offset, M_ALIGNMENT)  # 对齐提示
    tl.multiple_of(next_expert_offset, M_ALIGNMENT)  # 对齐提示

    data_start_ptr = data_ptr + curr_expert_offset * k  # 数据起始指针
    trans_data_start_ptr = trans_data_ptr + curr_expert_offset * k  # 转置起始指针

    k_coord = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # K坐标
    k_mask = k_coord < k  # K掩码
    for start_m in tl.range(0, num_tokens_of_expert, BLOCK_SIZE_M * tl.num_programs(1)):  # 遍历M块
        m_coord = start_m + m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # M坐标
        m_mask = m_coord < num_tokens_of_expert  # M掩码
        off = m_coord[:, None] * k + k_coord[None, :]  # 原始偏移
        trans_off = m_coord[:, None] + k_coord[None, :] * num_tokens_of_expert  # 转置偏移
        mask = m_mask[:, None] & k_mask[None, :]  # 组合掩码

        data = tl.load(data_start_ptr + off, mask=mask)  # 加载数据
        tl.store(trans_data_start_ptr + trans_off, data, mask=mask)  # 存储转置结果


def per_group_transpose(  # 逐组转置函数
    a: torch.Tensor,  # 输入张量
    expert_offsets: torch.Tensor,  # 专家偏移
    M_ALIGNMENT: int = 1,  # M对齐
) -> torch.Tensor:  # 返回转置结果
    """在MoE场景中按专家组对数据进行转置操作。"""  # 中文函数说明
    assert a.dim() == 2  # 断言2维
    assert a.is_contiguous(), "`a` is not contiguous"  # 断言连续

    m, k = a.size()  # 获取形状
    trans_a = torch.empty_like(a)  # 创建转置输出
    num_experts = expert_offsets.size(0) - 1  # 专家数量

    grid = lambda META: (  # 网格大小lambda
        num_experts,
        triton.cdiv((m + num_experts - 1) // num_experts, META["BLOCK_SIZE_M"]),
        triton.cdiv(k, META["BLOCK_SIZE_K"]),
    )
    _per_group_transpose[grid](  # 调用转置内核
        a, trans_a, expert_offsets, k, M_ALIGNMENT, BLOCK_SIZE_M=16, BLOCK_SIZE_K=8
    )
    return trans_a  # 返回转置结果


def is_weak_contiguous(x: torch.Tensor):  # 检查张量是否弱连续（包括转置情况）
    """检查张量是否弱连续，允许转置布局。"""  # 中文函数说明
    strides = x.stride()  # 获取步长
    sizes = x.shape  # 获取形状
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))  # 未转置
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))  # 已转置
    return is_transpose or is_not_transpose  # 任一情况返回True


def _as_column_scale(scale: torch.Tensor, expected_len: int) -> torch.Tensor:  # 将缩放因子转换为列格式
    """将缩放因子张量转换为列格式（M,1）以便后续使用。"""  # 中文函数说明
    if scale.dim() <= 1:  # 1维或0维
        return scale.reshape(-1, 1)  # 重塑为列格式
    if scale.dim() == 2:  # 2维
        if scale.shape[1] == 1:  # 已经是列格式
            return scale  # 直接返回
        if scale.shape[0] == 1 and scale.shape[1] == expected_len:  # 行格式需要转置
            return scale.t()  # 转置
    return scale  # 其他情况原样返回


@triton.jit  # Triton JIT编译装饰器
def scaled_mm_kernel(  # 缩放矩阵乘法Triton内核
    a_ptr,  # A指针
    b_ptr,  # B指针
    scale_a_ptr,  # A缩放指针
    scale_b_ptr,  # B缩放指针
    c_ptr,  # C输出指针
    bias_ptr,  # 偏置指针
    M,  # M维度
    N,  # N维度
    K,  # K维度
    stride_am,  # A的M步长
    stride_ak,  # A的K步长
    stride_bk,  # B的K步长
    stride_bn,  # B的N步长
    stride_cm,  # C的M步长
    stride_cn,  # C的N步长
    ACCUMULATOR_DTYPE: tl.constexpr,  # 累加器数据类型
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小
    BLOCK_SIZE_N: tl.constexpr,  # N方向块大小
    BLOCK_SIZE_K: tl.constexpr,  # K方向块大小
    BLOCK_SIZE_SCALE_A: tl.constexpr,  # A缩放块大小
    BLOCK_SIZE_SCALE_B: tl.constexpr,  # B缩放块大小
):
    """缩放矩阵乘法的Triton内核，支持per-tensor和per-token缩放。"""  # 中文函数说明
    pid = tl.program_id(axis=0)  # 获取程序ID

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N方向块数

    pid_m = pid // num_pid_n  # M方向块ID
    pid_n = pid % num_pid_n  # N方向块ID

    accumulator_dtype = ACCUMULATOR_DTYPE  # 累加器数据类型
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)  # 初始化累加器

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.  # 某些张量输入很大，会导致int32溢出，需要使用int64

    # Offsets and masks.  # 偏移和掩码
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)  # A的M偏移
    masks_am = offsets_am < M  # A的M掩码

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)  # B的N偏移
    masks_bn = offsets_bn < N  # B的N掩码

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)  # K偏移
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]  # A偏移
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]  # B偏移

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.  # 缩放块大小可能是1或BLOCK_SIZE_M，需要创建适当的偏移和掩码
    offsets_scale_am = (  # A缩放偏移
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M  # A缩放掩码

    offsets_scale_bn = (  # B缩放偏移
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N  # B缩放掩码

    a_ptrs = a_ptr + offsets_a  # A指针
    b_ptrs = b_ptr + offsets_b  # B指针

    scale_a_ptrs = scale_a_ptr + offsets_scale_am  # A缩放指针
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn  # B缩放指针

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  # 遍历K块
        masks_k = offsets_k < K  # K掩码
        masks_a = masks_am[:, None] & masks_k[None, :]  # A掩码
        a = tl.load(a_ptrs, mask=masks_a)  # 加载A

        masks_b = masks_k[:, None] & masks_bn[None, :]  # B掩码
        b = tl.load(b_ptrs, mask=masks_b)  # 加载B

        # Accumulate results.  # 累加结果
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)  # 执行点积

        offsets_k += BLOCK_SIZE_K  # 更新K偏移
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 更新A指针
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 更新B指针

    # Apply scale at end.  # 最后应用缩放因子
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]  # A缩放掩码
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)  # 加载A缩放
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.  # 需要广播到适当大小
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))  # 广播A缩放
    accumulator = scale_a * accumulator.to(tl.float32)  # 应用A缩放

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]  # B缩放掩码
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)  # 加载B缩放
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))  # 广播B缩放
    accumulator = scale_b.T * accumulator.to(tl.float32)  # 应用B缩放

    # Convert to output format.  # 转换为输出格式
    c = accumulator.to(c_ptr.type.element_ty)  # 类型转换

    # Add bias, it's already in output format, so add it after conversion.  # 添加偏置
    if bias_ptr:  # 如果有偏置
        offsets_bias = offsets_bn  # 偏置偏移
        bias_ptrs = bias_ptr + offsets_bias  # 偏置指针
        bias_mask = offsets_bias < N  # 偏置掩码
        bias = tl.load(bias_ptrs, bias_mask)  # 加载偏置
        c += bias  # 添加偏置

    # Save output  # 保存输出
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)  # C的M偏移
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)  # C的N偏移
    offs_cm = offs_cm.to(tl.int64)  # 转换为int64
    offs_cn = offs_cn.to(tl.int64)  # 转换为int64
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]  # C指针
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  # C掩码

    tl.store(c_ptrs, c, mask=c_mask)  # 存储输出


# input  - [M, K]  # 输入张量形状
# weight - [K, N]  # 权重张量形状
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py  # 参考自vLLM项目
def triton_scaled_mm(  # Triton缩放矩阵乘法函数
    input: torch.Tensor,  # 输入张量 [M, K]
    weight: torch.Tensor,  # 权重张量 [K, N]
    scale_a: torch.Tensor,  # A缩放因子
    scale_b: torch.Tensor,  # B缩放因子
    out_dtype: type[torch.dtype],  # 输出数据类型
    bias: Optional[torch.Tensor] = None,  # 偏置
    block_size_m: int = 32,  # M方向块大小
    block_size_n: int = 32,  # N方向块大小
    block_size_k: int = 32,  # K方向块大小
    use_heuristic=True,  # 是否使用启发式
) -> torch.Tensor:  # 返回矩阵乘法结果
    """使用Triton实现的缩放矩阵乘法，支持per-tensor和per-token缩放。"""  # 中文函数说明
    M, K = input.shape  # 获取输入形状
    N = weight.shape[1]  # 获取N维度

    assert N > 0 and K > 0 and M > 0  # 断言维度非零
    assert weight.shape[0] == K  # 断言权重形状
    assert input.dtype == weight.dtype  # 断言类型一致

    scale_a = _as_column_scale(scale_a, M)  # 转换A缩放格式
    scale_b = _as_column_scale(scale_b, N)  # 转换B缩放格式

    assert scale_a.dim() == 2 and scale_b.dim() == 2  # 断言2维
    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()  # 断言类型
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)  # 断言形状
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)  # 断言形状
    assert out_dtype.is_floating_point  # 断言浮点输出
    assert bias is None or bias.is_floating_point()  # 断言偏置为浮点
    assert is_weak_contiguous(input)  # 断言弱连续
    assert is_weak_contiguous(weight)  # 断言弱连续

    grid = lambda META: (  # 网格大小
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)  # 创建结果张量

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1  # 检查是否为标量

    if use_heuristic:  # 使用启发式选择块大小
        is_small_N = N < 8192  # N是否较小
        next_power_of_2_M = max(32, triton.next_power_of_2(M))  # M的2的幂次
        if next_power_of_2_M <= 32:  # M较小时
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)  # 选择块大小
        elif next_power_of_2_M <= 64:  # M中等
            tile_shape = (64, 64, 256)  # 块大小
        elif next_power_of_2_M <= 128:  # M较大
            tile_shape = (64, 128, 128)  # 块大小
        else:  # M很大
            tile_shape = (128, 128, 128)  # 块大小

    block_size_m, block_size_n, block_size_k = tile_shape  # 解包块大小

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m  # A缩放块大小
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n  # B缩放块大小

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32  # 累加器类型

    # A = input, B = weight, C = result  # A=输入, B=权重, C=结果
    # A = M x K, B = K x N, C = M x N  # 矩阵形状说明
    scaled_mm_kernel[grid](  # 调用缩放矩阵乘法内核
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
    )

    return result.to(out_dtype)  # 返回结果


if _is_cuda:  # CUDA平台
    if enable_sgl_per_token_group_quant_8bit:  # 如果启用v2内核

        @register_fake_if_exists("sgl_kernel::sgl_per_token_group_quant_8bit")  # 注册fake实现
        def _(  # v2版本的fake实现
            input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
        ):
            return  # 返回空

    else:  # 旧版本

        @register_fake_if_exists("sgl_kernel::sgl_per_token_group_quant_fp8")  # 注册fake实现
        def _(  # 旧版本的fake实现
            input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
        ):
            return  # 返回空

    @register_fake_if_exists("sgl_kernel::sgl_per_token_quant_fp8")  # 注册fake实现
    def _(input, output_q, output_s):  # 逐token量化的fake实现
        return  # 返回空
