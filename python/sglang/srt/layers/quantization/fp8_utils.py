# FP8量化工具模块 - 提供FP8量化的高层工具函数和多种GEMM后端的线性层实现
# 支持DeepGEMM、FlashInfer、CUTLASS、AITER、Triton等多种后端
# 包含MXFP4反量化、UE8M0缩放变换、块量化和通道量化等工具函数

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from enum import Enum  # 导入枚举类
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union  # 导入类型注解

import torch  # 导入PyTorch

from sglang.srt.layers import deep_gemm_wrapper  # 导入DeepGEMM包装器
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8  # 导入SGLang逐token组FP8量化
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil  # 导入MXFP4量化工具
from sglang.srt.utils.common import torch_release  # 导入torch版本号

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.server_args import ServerArgs  # 导入服务器参数类型

from sglang.srt.layers.quantization.fp8_kernel import (  # 从FP8内核模块导入
    fp8_dtype,  # FP8数据类型
    fp8_max,  # FP8最大值
    fp8_min,  # FP8最小值
    is_fp8_fnuz,  # FP8 FNUZ格式检测
    mxfp8_block_scaled_matmul_triton,  # MXFP8分块缩放矩阵乘法
    per_token_group_quant_fp8,  # 逐token组FP8量化
    scaled_fp8_quant,  # FP8量化统一入口
    sglang_per_token_quant_fp8,  # SGLang逐token FP8量化
    static_quant_fp8,  # 静态FP8量化
    triton_scaled_mm,  # Triton缩放矩阵乘法
    w8a8_block_fp8_matmul_deepgemm,  # DeepGEMM分块FP8矩阵乘法
    w8a8_block_fp8_matmul_triton,  # Triton分块FP8矩阵乘法
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取
from sglang.srt.utils import (  # 从工具模块导入
    ceil_align,  # 向上对齐
    ceil_div,  # 向上整除
    get_bool_env_var,  # 获取布尔环境变量
    get_cuda_version,  # 获取CUDA版本
    get_device_capability,  # 获取设备计算能力
    get_hip_version,  # 获取HIP版本
    is_blackwell_supported,  # Blackwell架构支持检测
    is_cuda,  # CUDA检测
    is_flashinfer_available,  # FlashInfer可用性检测
    is_gfx95_supported,  # gfx95架构支持检测
    is_hip,  # HIP检测
    is_musa,  # MUSA检测
    is_sm90_supported,  # SM90支持检测
    is_sm100_supported,  # SM100支持检测
    is_sm120_supported,  # SM120支持检测
    offloader,  # 卸载器
)
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

logger = logging.getLogger(__name__)  # 创建日志记录器

_is_hip = is_hip()  # 检测是否为HIP环境
_is_cuda = is_cuda()  # 检测是否为CUDA环境
_is_fp8_fnuz = is_fp8_fnuz()  # 检测FP8是否为FNUZ格式
_is_sm100_supported = is_sm100_supported()  # 检测是否支持SM100
_is_sm120_supported = is_sm120_supported()  # 检测是否支持SM120
_is_gfx95_supported = is_gfx95_supported()  # 检测是否支持gfx95
_is_musa = is_musa()  # 检测是否为MUSA环境

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported  # 是否在gfx95上使用AITER
# ROCm 7.0 hipcc miscompiles gemm_a8w8_blockscale_bpreshuffle on gfx95 (#23319).  # ROCm 7.0在gfx95上错误编译blockscale内核
_use_aiter_bpreshuffle_gfx95 = _use_aiter_gfx95 and get_hip_version() >= (7, 2, 0)  # HIP 7.2+才启用


def use_aiter_triton_gemm_w8a8_tuned_gfx950(n: int, k: int) -> bool:  # 判断是否使用AITER Triton GEMM（gfx950调优版本）
    """检查给定的(N,K)维度是否在gfx950调优配置列表中。"""  # 中文函数说明
    return (n, k) in [  # 返回是否在调优列表中
        (1024, 8192),
        (16384, 1536),
        (2112, 7168),
        (3072, 1536),
        (32768, 8192),
        (4096, 7168),
        (4608, 7168),
        (512, 7168),
        (7168, 2048),
        (7168, 2304),
        (7168, 16384),
        (7168, 256),
        (8192, 1024),
        (8192, 32768),
    ]


if _use_aiter:  # 如果使用AITER
    import aiter  # 导入AITER库
    from aiter import (  # 从AITER导入
        gemm_a8w8_blockscale_bpreshuffle,  # 分块缩放bpreshuffle GEMM
        gemm_a8w8_bpreshuffle,  # bpreshuffle GEMM
        get_hip_quant,  # HIP量化函数获取
    )
    from aiter.ops.triton.gemm_a8w8_blockscale import (  # 从AITER Triton导入
        gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,  # Triton分块缩放GEMM
    )

    aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)  # 获取1x128量化函数


if _is_cuda:  # CUDA平台
    from sgl_kernel import fp8_blockwise_scaled_mm, fp8_scaled_mm  # 导入FP8缩放矩阵乘法内核

    from sglang.srt.utils.patch_torch import register_fake_if_exists  # 导入torch补丁注册

    @register_fake_if_exists("sgl_kernel::fp8_scaled_mm")  # 注册fp8_scaled_mm的fake实现
    def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):  # fake实现
        # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].  # 输出形状说明
        M = mat_a.shape[-2]  # M维度
        N = mat_b.shape[-1]  # N维度
        return mat_a.new_empty((M, N), dtype=out_dtype)  # 返回空张量

    @register_fake_if_exists("sgl_kernel::fp8_blockwise_scaled_mm")  # 注册fp8_blockwise_scaled_mm的fake实现
    def _fp8_blockwise_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype):  # fake实现
        # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].  # 输出形状说明
        M = mat_a.shape[-2]  # M维度
        N = mat_b.shape[-1]  # N维度
        return mat_a.new_empty((M, N), dtype=out_dtype)  # 返回空张量


use_triton_w8a8_fp8_kernel = get_bool_env_var("USE_TRITON_W8A8_FP8_KERNEL")  # 是否使用Triton W8A8 FP8内核

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale  # PyTorch 2.5+中_scaled_mm需要缩放因子，分配虚拟张量
TORCH_DEVICE_IDENTITY = None  # 虚拟身份张量


def use_rowwise_torch_scaled_mm():  # 检测是否使用行方向torch._scaled_mm
    """检测当前平台是否支持torch._scaled_mm的行方向缩放功能。"""  # 中文函数说明
    if _is_hip:  # HIP平台
        # The condition to determine if it is on a platform that supports
        # torch._scaled_mm rowwise feature.
        # The condition is determined once as the operations
        # are time consuming.  # 条件只判断一次，因为操作耗时
        return get_device_capability() >= (9, 4) and torch_release >= (2, 7)  # gfx94+和torch 2.7+
    return False  # 非HIP平台返回False


USE_ROWWISE_TORCH_SCALED_MM = use_rowwise_torch_scaled_mm()  # 缓存结果


@lru_cache(maxsize=1)  # LRU缓存
def cutlass_fp8_supported():  # 检测CUTLASS FP8是否受支持
    """检测当前硬件和CUDA版本是否支持CUTLASS FP8。"""  # 中文函数说明
    if not _is_cuda:  # 非CUDA平台不支持
        return False
    major, minor = get_device_capability()  # 获取计算能力
    cuda_version = get_cuda_version()  # 获取CUDA版本
    if major >= 9:  # SM90+需要CUDA 12.0+
        return cuda_version >= (12, 0)
    elif major == 8 and minor == 9:  # SM89需要CUDA 12.4+
        return cuda_version >= (12, 4)
    return False  # 其他情况不支持


def normalize_e4m3fn_to_e4m3fnuz(  # 将E4M3FN格式归一化为E4M3FNUZ
    weight: torch.Tensor,  # 权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:  # 返回转换后的权重、缩放因子和输入缩放
    """将FP8 E4M3FN格式的权重和缩放因子转换为E4M3FNUZ格式。"""  # 中文函数说明
    assert weight.dtype == torch.float8_e4m3fn  # 断言输入格式
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html  # 10000000在FN中为零，在FNUZ中为NaN，设为0
    weight_as_int8 = weight.view(torch.int8)  # 视图转换为int8
    ROCM_FP8_NAN_AS_INT = -128  # FP8 NaN的int8表示
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0  # 将NaN设为0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)  # 视图转换为FNUZ格式

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html  # 相同位模式下FNUZ值是FN值的一半，缩放因子需翻倍
    weight_scale = weight_scale * 2.0  # 缩放因子翻倍
    if input_scale is not None:  # 如果有输入缩放因子
        input_scale = input_scale * 2.0  # 同样翻倍
    return weight, weight_scale, input_scale  # 返回转换结果


class Fp8GemmRunnerBackend(Enum):  # FP8 GEMM运行器后端枚举
    """Enum for FP8 GEMM runner backend selection."""  # FP8 GEMM运行器后端选择枚举

    AUTO = "auto"  # 自动选择
    FLASHINFER_TRTLLM = "flashinfer_trtllm"  # FlashInfer TRTLLM后端
    FLASHINFER_CUTLASS = "flashinfer_cutlass"  # FlashInfer CUTLASS后端
    FLASHINFER_DEEPGEMM = "flashinfer_deepgemm"  # FlashInfer DeepGEMM后端
    CUTLASS = "cutlass"  # CUTLASS后端
    DEEP_GEMM = "deep_gemm"  # DeepGEMM后端
    TRITON = "triton"  # Triton后端
    AITER = "aiter"  # AITER后端

    def is_auto(self) -> bool:  # 是否为自动选择
        return self == Fp8GemmRunnerBackend.AUTO

    def is_flashinfer_trtllm(self) -> bool:  # 是否为FlashInfer TRTLLM
        return self == Fp8GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self) -> bool:  # 是否为FlashInfer CUTLASS
        return self == Fp8GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_deepgemm(self) -> bool:  # 是否为FlashInfer DeepGEMM
        return self == Fp8GemmRunnerBackend.FLASHINFER_DEEPGEMM

    def is_cutlass(self) -> bool:  # 是否为CUTLASS
        return self == Fp8GemmRunnerBackend.CUTLASS

    def is_deep_gemm(self) -> bool:  # 是否为DeepGEMM
        return self == Fp8GemmRunnerBackend.DEEP_GEMM

    def is_triton(self) -> bool:  # 是否为Triton
        return self == Fp8GemmRunnerBackend.TRITON

    def is_aiter(self) -> bool:  # 是否为AITER
        return self == Fp8GemmRunnerBackend.AITER


FP8_GEMM_RUNNER_BACKEND: Fp8GemmRunnerBackend | None = None  # 全局FP8 GEMM后端配置


def _check_cutlass_block_fp8_hardware_support() -> bool:  # 检查CUTLASS分块FP8硬件支持
    """Return True if CUTLASS block FP8 is supported (Hopper or newer with CUDA 12.0+)."""  # Hopper或更新架构且CUDA 12.0+返回True
    return is_sm90_supported() or is_blackwell_supported()  # SM90或Blackwell支持


if is_blackwell_supported() and is_flashinfer_available():  # Blackwell且有FlashInfer
    from flashinfer import SfLayout  # 导入缩放布局
    from flashinfer import mm_mxfp8 as _raw_flashinfer_mm_mxfp8  # 导入MXFP8矩阵乘法
    from flashinfer import mxfp8_quantize as _raw_flashinfer_mxfp8_quantize  # 导入MXFP8量化
    from flashinfer.gemm import gemm_fp8_nt_groupwise as _raw_gemm_fp8_nt_groupwise  # 导入分组FP8 GEMM

    from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

    @lru_cache(maxsize=1)  # 缓存结果
    def _get_flashinfer_groupwise_backend() -> str:  # 获取FlashInfer分组后端
        """根据硬件和用户配置选择FlashInfer分组GEMM的后端。"""  # 中文函数说明
        if get_fp8_gemm_runner_backend().is_flashinfer_cutlass():  # 用户指定CUTLASS
            return "cutlass"
        if get_fp8_gemm_runner_backend().is_flashinfer_trtllm():  # 用户指定TRTLLM
            return "trtllm"

        major, minor = get_device_capability()  # 获取计算能力
        # SM120/121: CUTLASS only.
        # SM100/103: TRTLLM only.  # SM120/121仅CUTLASS，SM100/103仅TRTLLM
        if major >= 12:  # SM120+
            return "cutlass"
        return "trtllm"  # SM100/103使用TRTLLM

    # Wrap gemm_fp8_nt_groupwise as a custom op so torch.compile does not trace
    # into flashinfer's JIT compilation code (pathlib/cubin_loader ops).  # 包装为自定义算子避免torch.compile追踪FlashInfer的JIT编译
    @register_custom_op(
        op_name="flashinfer_gemm_fp8_nt_groupwise",
        mutates_args=[],
        fake_impl=lambda q_input, weight, x_scale, weight_scale, out_dtype: (
            q_input.new_empty((q_input.shape[0], weight.shape[0]), dtype=out_dtype)
        ),
    )
    def gemm_fp8_nt_groupwise(  # FlashInfer分组FP8矩阵乘法
        q_input: torch.Tensor,  # 量化输入
        weight: torch.Tensor,  # 权重
        x_scale: torch.Tensor,  # 输入缩放因子
        weight_scale: torch.Tensor,  # 权重缩放因子
        out_dtype: torch.dtype,  # 输出数据类型
    ) -> torch.Tensor:  # 返回矩阵乘法结果
        """FlashInfer分组FP8矩阵乘法的自定义算子包装。"""  # 中文函数说明
        backend = _get_flashinfer_groupwise_backend()  # 获取后端
        if backend == "cutlass":  # CUTLASS后端
            # FlashInfer CUTLASS groupwise kernel requires contiguous scale tensors  # CUTLASS内核需要连续缩放张量
            x_scale = x_scale.contiguous()  # 确保连续
            weight_scale = weight_scale.contiguous()  # 确保连续
            return _raw_gemm_fp8_nt_groupwise(  # 调用原始函数
                q_input,
                weight,
                x_scale,
                weight_scale,
                out_dtype=out_dtype,
                backend="cutlass",
                scale_major_mode="MN",
            )
        return _raw_gemm_fp8_nt_groupwise(  # TRTLLM后端
            q_input,
            weight,
            x_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend=backend,
        )

    # Wrap MXFP8 ops as custom ops so torch.compile does not trace into
    # flashinfer's JIT compilation path (filesystem checks/cubin loader).  # 包装MXFP8操作避免torch.compile追踪FlashInfer JIT
    def _fake_flashinfer_mxfp8_quantize(  # MXFP8量化fake实现
        input: torch.Tensor,  # 输入张量
        _is_sf_swizzled_layout: bool = True,  # 是否使用swizzle布局
        alignment: int = 32,  # 对齐大小
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
        # Fake mode only needs dtypes and output rank to propagate compile graph.
        # The scale tensor shape is not consumed before the following fake mm op.  # fake模式只需传播数据类型和秩
        k_aligned = ((input.shape[1] + alignment - 1) // alignment) * alignment  # K对齐
        q_input = input.new_empty(  # 创建量化输出
            (input.shape[0], k_aligned), dtype=torch.float8_e4m3fn
        )
        scale = input.new_empty((1,), dtype=torch.uint8)  # 创建缩放因子
        return q_input, scale  # 返回

    @register_custom_op(
        op_name="flashinfer_mxfp8_quantize",
        mutates_args=[],
        fake_impl=_fake_flashinfer_mxfp8_quantize,
    )
    def flashinfer_mxfp8_quantize(  # FlashInfer MXFP8量化
        input: torch.Tensor,  # 输入张量
        is_sf_swizzled_layout: bool = True,  # 是否swizzle布局
        alignment: int = 32,  # 对齐大小
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
        """FlashInfer MXFP8量化的自定义算子包装。"""  # 中文函数说明
        return _raw_flashinfer_mxfp8_quantize(  # 调用原始函数
            input,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment,
            sf_swizzle_layout=SfLayout.layout_128x4,
        )

    @register_custom_op(
        op_name="flashinfer_mm_mxfp8",
        mutates_args=[],
        fake_impl=lambda q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, use_8x4_sf_layout=False, backend="auto": (
            q_input.new_empty((q_input.shape[0], weight_t.shape[1]), dtype=out_dtype)
        ),
    )
    def flashinfer_mm_mxfp8(  # FlashInfer MXFP8矩阵乘法
        q_input: torch.Tensor,  # 量化输入
        weight_t: torch.Tensor,  # 转置权重
        x_scale_u8: torch.Tensor,  # 输入缩放因子(uint8)
        weight_scale_t: torch.Tensor,  # 权重缩放因子(uint8)
        out_dtype: torch.dtype,  # 输出数据类型
        use_8x4_sf_layout: bool = False,  # 是否使用8x4缩放布局
        backend: str = "auto",  # 后端选择
    ) -> torch.Tensor:  # 返回矩阵乘法结果
        """FlashInfer MXFP8矩阵乘法的自定义算子包装。"""  # 中文函数说明
        return _raw_flashinfer_mm_mxfp8(  # 调用原始函数
            q_input,
            weight_t,
            x_scale_u8,
            weight_scale_t,
            out_dtype=out_dtype,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )


if is_sm90_supported() and is_flashinfer_available():  # SM90且有FlashInfer
    # FlashInfer SM90 DeepGEMM with automatic swapAB optimization for small M  # FlashInfer SM90 DeepGEMM，小M时自动swapAB
    from flashinfer.gemm import fp8_blockscale_gemm_sm90  # 导入SM90分块缩放GEMM


def dispatch_w8a8_block_fp8_linear() -> Callable:  # 分发W8A8分块FP8线性层实现
    """
    Dispatch to the appropriate FP8 block linear implementation.

    This function selects the backend based on:
    1. The --fp8-gemm-backend server argument (preferred)
    2. Auto-detection based on hardware capabilities
    """  # 根据服务器参数或硬件能力选择FP8分块线性层后端
    backend = get_fp8_gemm_runner_backend()  # 获取后端配置

    # Handle explicit backend selection via --fp8-gemm-backend  # 处理显式后端选择
    if not backend.is_auto():  # 非自动模式
        return _dispatch_explicit_backend(backend)  # 使用显式指定的后端

    # Auto mode: Select based purely on hardware/backend availability  # 自动模式：根据硬件/后端可用性选择
    return _dispatch_auto_backend()  # 自动选择后端


def dispatch_w8a8_mxfp8_linear() -> Callable:  # 分发MXFP8线性层实现
    """Dispatch MXFP8 linear kernel by --fp8-gemm-backend.

    For MXFP8, Triton remains the default path. We only route to FlashInfer
    when backend is explicitly set to flashinfer_cutlass or flashinfer_trtllm.
    """  # 根据fp8-gemm-backend参数分发MXFP8线性内核，默认Triton，显式指定时使用FlashInfer
    backend = get_fp8_gemm_runner_backend()  # 获取后端
    if backend.is_flashinfer_trtllm():  # FlashInfer TRTLLM
        return flashinfer_mxfp8_blockscaled_linear  # 返回FlashInfer实现
    elif backend.is_flashinfer_cutlass():  # FlashInfer CUTLASS
        return flashinfer_mxfp8_blockscaled_linear  # 返回FlashInfer实现
    return triton_mxfp8_blockscaled_linear  # 默认返回Triton实现


def _dispatch_explicit_backend(backend: Fp8GemmRunnerBackend) -> Callable:  # 显式后端分发
    """Dispatch based on explicitly selected backend."""  # 根据显式选择的后端分发
    if backend.is_flashinfer_trtllm():  # FlashInfer TRTLLM
        if not (is_sm100_supported() and is_flashinfer_available()):  # 检查支持
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_trtllm, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer TRTLLM FP8 GEMM requires SM100/SM103 GPUs and FlashInfer."
            )
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback  # 返回FlashInfer实现

    elif backend.is_flashinfer_cutlass():  # FlashInfer CUTLASS
        if not (is_blackwell_supported() and is_flashinfer_available()):  # 检查支持
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_cutlass, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer CUTLASS FP8 GEMM requires Blackwell GPUs and FlashInfer."
            )
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback  # 返回FlashInfer实现

    elif backend.is_flashinfer_deepgemm():  # FlashInfer DeepGEMM
        if not (is_sm90_supported() and is_flashinfer_available()):  # 检查支持
            raise RuntimeError(
                "FlashInfer DeepGEMM with swapAB requested via --fp8-gemm-backend=flashinfer_deepgemm, "
                "but it's not available. This backend requires Hopper (SM90) GPUs and FlashInfer "
                "to be installed."
            )
        return flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback  # 返回DeepGEMM实现

    elif backend.is_cutlass():  # CUTLASS
        if not _check_cutlass_block_fp8_hardware_support():  # 检查支持
            raise RuntimeError(
                "CUTLASS block FP8 requested via --fp8-gemm-backend=cutlass, "
                "but hardware does not support it. CUTLASS block FP8 requires "
                "Hopper (SM90+) GPUs with CUDA 12.0+."
            )
        return cutlass_w8a8_block_fp8_linear_with_fallback  # 返回CUTLASS实现

    elif backend.is_aiter():  # AITER
        if not _use_aiter:  # 检查支持
            raise RuntimeError(
                "AITER backend requested via --fp8-gemm-backend=aiter, "
                "but AITER is not available. AITER requires AMD GPUs with "
                "SGLANG_USE_AITER=1 environment variable set."
            )
        return aiter_w8a8_block_fp8_linear  # 返回AITER实现

    elif backend.is_deep_gemm():  # DeepGEMM
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:  # 检查支持
            raise RuntimeError(
                "DeepGEMM backend requested via --fp8-gemm-backend=deep_gemm, "
                "but DeepGEMM is not available. This usually means the deep_gemm package "
                "is not installed or has been disabled via SGLANG_ENABLE_JIT_DEEPGEMM=0."
            )
        return deepgemm_w8a8_block_fp8_linear_with_fallback  # 返回DeepGEMM实现

    elif backend.is_triton():  # Triton
        return triton_w8a8_block_fp8_linear  # 返回Triton实现

    else:  # 未知后端
        raise ValueError(f"Unknown FP8 GEMM backend: {backend}")  # 抛出错误


def _dispatch_auto_backend() -> Callable:  # 自动后端分发
    """Auto-select the best backend based on hardware capabilities."""  # 根据硬件能力自动选择最佳后端
    # Priority order for auto selection:
    # 1. DeepGEMM (if enabled and available)
    # 2. FlashInfer TRTLLM (if Blackwell GPU and FlashInfer available)
    # 3. CUTLASS (if Hopper+ GPU and CUDA 12.0+)
    # 4. AITER (if AMD GPU with AITER enabled)
    # 5. Triton (fallback)  # 自动选择优先级顺序

    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:  # DeepGEMM可用
        return deepgemm_w8a8_block_fp8_linear_with_fallback  # 返回DeepGEMM
    elif is_blackwell_supported() and is_flashinfer_available():  # Blackwell+FlashInfer
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback  # 返回FlashInfer
    elif _check_cutlass_block_fp8_hardware_support():  # CUTLASS受支持
        return cutlass_w8a8_block_fp8_linear_with_fallback  # 返回CUTLASS
    elif _use_aiter:  # AITER可用
        return aiter_w8a8_block_fp8_linear  # 返回AITER
    else:  # 回退
        return triton_w8a8_block_fp8_linear  # 返回Triton


def initialize_fp8_gemm_config(server_args: ServerArgs) -> None:  # 初始化FP8 GEMM配置
    """Initialize FP8 GEMM configuration."""  # 初始化FP8 GEMM配置
    global FP8_GEMM_RUNNER_BACKEND  # 声明全局变量

    backend = server_args.fp8_gemm_runner_backend  # 获取后端参数
    if backend == "auto" and is_sm120_supported():  # SM120自动模式
        # TODO(brayden): Verify if CUTLASS can be set by default once SwapAB is supported  # 确认CUTLASS是否可设为默认
        backend = "triton"  # SM120默认使用Triton

    FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend(backend)  # 设置全局后端


def get_fp8_gemm_runner_backend() -> Fp8GemmRunnerBackend:  # 获取当前FP8 GEMM后端
    """Get the current FP8 GEMM runner backend."""  # 获取当前FP8 GEMM运行器后端
    global FP8_GEMM_RUNNER_BACKEND  # 声明全局变量
    if FP8_GEMM_RUNNER_BACKEND is None:  # 如果未初始化
        FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend.AUTO  # 设为自动
    return FP8_GEMM_RUNNER_BACKEND  # 返回当前后端


def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(  # FlashInfer W8A8分块FP8线性层（带回退）
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    block_size: List[int],  # 块大小
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
) -> torch.Tensor:  # 返回线性层输出
    """FlashInfer W8A8分块FP8线性层实现，不支持时回退到Triton。"""  # 中文函数说明
    assert input_scale is None  # 断言无预计算输入缩放

    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维
    backend = _get_flashinfer_groupwise_backend()  # 获取后端
    # TRTLLM backend requires K dimension >= 256.  # TRTLLM后端要求K>=256
    if backend == "trtllm" and input_2d.shape[1] < 256:  # K太小回退
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    output_shape = [*input.shape[:-1], weight.shape[0]]  # 计算输出形状

    # TRTLLM uses the existing SGLang column-major scale layout.
    # CUTLASS with scale_major_mode="MN" expects (k//block_k, m), so we normalize below.  # TRTLLM使用列主序，CUTLASS使用(k//block_k, m)布局
    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=(backend == "trtllm")  # TRTLLM时使用列主序
    )
    if backend == "cutlass":  # CUTLASS后端需要特定缩放布局
        block_n, block_k = block_size  # 解包块大小
        m, k = input_2d.shape  # 获取输入形状
        n = weight.shape[0]  # 获取输出维度
        expected_x_scale_shape = (k // block_k, m)  # 期望的输入缩放形状
        expected_weight_scale_shape = (k // block_k, n // block_n)  # 期望的权重缩放形状
        if x_scale.shape == (m, k // block_k):  # 需要转置
            x_scale = x_scale.transpose(-1, -2).contiguous()  # 转置输入缩放
        if weight_scale.shape == (n // block_n, k // block_k):  # 需要转置
            weight_scale = weight_scale.transpose(-1, -2).contiguous()  # 转置权重缩放
        assert x_scale.shape == expected_x_scale_shape, (  # 断言输入缩放形状
            "FlashInfer CUTLASS groupwise FP8 expects A scale layout "
            f"(k//block_k, m) for scale_major_mode='MN', got {tuple(x_scale.shape)}; "
            f"expected {expected_x_scale_shape}. "
            f"strides={x_scale.stride()} is_contiguous={x_scale.is_contiguous()} "
            f"m={m} n={n} k={k} block_size={block_size}"
        )
        assert weight_scale.shape == expected_weight_scale_shape, (  # 断言权重缩放形状
            "FlashInfer CUTLASS groupwise FP8 expects B scale layout "
            f"(k//block_k, n//block_n) for scale_major_mode='MN', got {tuple(weight_scale.shape)}; "
            f"expected {expected_weight_scale_shape}. "
            f"strides={weight_scale.stride()} is_contiguous={weight_scale.is_contiguous()} "
            f"m={m} n={n} k={k} block_size={block_size}"
        )
        assert x_scale.dtype == torch.float32, (  # 断言输入缩放类型
            "FlashInfer CUTLASS groupwise FP8 expects x_scale dtype float32, "
            f"got {x_scale.dtype}."
        )
        assert weight_scale.dtype == torch.float32, (  # 断言权重缩放类型
            "FlashInfer CUTLASS groupwise FP8 expects weight_scale dtype float32, "
            f"got {weight_scale.dtype}."
        )
    # TRTLLM path continues using the original quantized scale layout.  # TRTLLM路径使用原始量化缩放布局
    output = gemm_fp8_nt_groupwise(  # 执行分组FP8 GEMM
        q_input,
        weight,
        x_scale,
        weight_scale,
        out_dtype=input_2d.dtype,
    )

    if bias is not None:  # 如果有偏置
        output += bias  # 添加偏置

    return output.to(dtype=input_2d.dtype).view(*output_shape)  # 返回结果


def flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback(  # FlashInfer DeepGEMM W8A8分块FP8线性层
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    block_size: List[int],  # 块大小
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
) -> torch.Tensor:  # 返回线性层输出
    """
    FlashInfer DeepGEMM backend for SM90 (Hopper) with swapAB optimization.

    Uses flashinfer.gemm.fp8_blockscale_gemm_sm90 which automatically selects
    the swapAB kernel for small M dimensions (M < 32) for better performance
    during decoding/low batch size scenarios.

    For SM90 (Hopper), this uses the DeepGEMM JIT with automatic swapAB selection.
    """  # SM90上的FlashInfer DeepGEMM后端，小M时自动使用swapAB内核
    assert input_scale is None  # 断言无预计算输入缩放

    output_dtype = input.dtype  # 获取输出类型
    dtype_supported = output_dtype == torch.bfloat16  # DeepGEMM仅支持bfloat16

    # fp8_blockscale_gemm_sm90 requires: N % 64 == 0, K % 128 == 0  # 形状要求
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0  # 检查形状

    if not (shape_supported and dtype_supported):  # 不支持时回退
        if weight_scale.dtype == torch.int32:  # UE8M0格式需要解包
            weight_scale = _unpack_ue8m0_scale_for_triton(
                weight_scale, weight.shape, block_size
            )
        return triton_w8a8_block_fp8_linear(  # 回退到Triton
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 输出形状

    # - input: (M, K) BF16 or FP8
    # - weight: (N, K) FP8 with weight_scale
    # - weight_scale: (N, K//128) for per-token or (N//128, K//128) for per-block  # 输入输出形状说明

    output = fp8_blockscale_gemm_sm90(  # 调用SM90分块缩放GEMM
        input_2d,
        weight,
        input_scale=None,  # BF16 input, internal quantization  # BF16输入，内部量化
        weight_scale=weight_scale,
        out_dtype=output_dtype,
    )

    if bias is not None:  # 添加偏置
        output += bias
    return output.view(*output_shape)  # 返回结果


def cutlass_w8a8_block_fp8_linear_with_fallback(  # CUTLASS W8A8分块FP8线性层（带回退）
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    block_size: List[int],  # 块大小
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
) -> torch.Tensor:  # 返回线性层输出
    """CUTLASS W8A8分块FP8线性层，形状不支持时回退到Triton。"""  # 中文函数说明
    assert input_scale is None  # 断言无预计算输入缩放

    # TODO: add more robust shape check here  # 需要更完善的形状检查
    shape_supported = weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0  # 形状检查

    if not shape_supported:  # 不支持时回退
        # fallback to triton  # 回退到Triton
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 输出形状

    q_input, x_scale = per_token_group_quant_fp8(  # 逐token组FP8量化
        input_2d, block_size[1], column_major_scales=True  # 使用列主序缩放
    )
    output = fp8_blockwise_scaled_mm(  # CUTLASS分块缩放矩阵乘法
        q_input, weight.T, x_scale, weight_scale.T, out_dtype=input_2d.dtype
    )
    if bias is not None:  # 添加偏置
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)  # 返回结果


def deepgemm_w8a8_block_fp8_linear_with_fallback(  # DeepGEMM W8A8分块FP8线性层（带回退）
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    block_size: List[int],  # 块大小
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
) -> torch.Tensor:  # 返回线性层输出
    """DeepGEMM W8A8分块FP8线性层，不支持时回退到Triton。"""  # 中文函数说明
    assert input_scale is None  # 断言无预计算输入缩放

    output_dtype = input.dtype  # 获取输出类型
    dtype_supported = output_dtype == torch.bfloat16  # DeepGEMM仅支持bfloat16

    # TODO: https://github.com/sgl-project/sglang/pull/6890#issuecomment-2943395737  # 待修复
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0  # 形状检查

    if not (shape_supported and dtype_supported):  # 不支持时回退
        # fall back to triton  # 回退到Triton
        # If weight_scale is in UE8M0 packed format (int32), convert back to float32
        # UE8M0 format has shape (N, K//block_k//4) with dtype int32
        # Triton expects shape (N//block_n, K//block_k) with dtype float32  # UE8M0打包格式需要解包为float32
        if weight_scale.dtype == torch.int32:  # UE8M0格式
            weight_scale = _unpack_ue8m0_scale_for_triton(
                weight_scale, weight.shape, block_size
            )
        return triton_w8a8_block_fp8_linear(  # 回退到Triton
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 输出形状

    if not _is_musa:  # 非MUSA平台
        q_input, x_scale = sglang_per_token_group_quant_fp8(  # 逐token组FP8量化
            input_2d,
            block_size[1],
            column_major_scales=True,  # 列主序缩放
            scale_tma_aligned=True,  # TMA对齐
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # UE8M0缩放
        )
    else:  # MUSA平台
        q_input, x_scale = sglang_per_token_group_quant_fp8(  # MUSA简化版本
            input_2d,
            block_size[1],
        )

    output = w8a8_block_fp8_matmul_deepgemm(  # DeepGEMM矩阵乘法
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=output_dtype
    )
    if bias is not None:  # 添加偏置
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)  # 返回结果


def _unpack_ue8m0_scale_for_triton(  # 将UE8M0打包缩放因子解包为Triton所需的float32格式
    sf_packed: torch.Tensor,  # 打包的缩放因子
    weight_shape: Tuple[int, int],  # 权重形状
    block_size: List[int],  # 块大小
) -> torch.Tensor:  # 返回解包后的缩放因子
    """
    Unpack UE8M0 packed scale tensor back to float32 format for triton kernel.

    The UE8M0 format packs scales as:
    - Shape: (N, K//block_k//4) with dtype int32
    - Each int32 contains 4 uint8 scale values

    Triton expects:
    - Shape: (N//block_n, K//block_k) with dtype float32

    Args:
        sf_packed: Packed scale tensor with shape (N, packed_k_groups) and dtype int32
        weight_shape: (N, K) shape of the weight tensor
        block_size: [block_n, block_k] quantization block size

    Returns:
        Unpacked scale tensor with shape (n_groups, k_groups) and dtype float32
    """  # 将UE8M0打包的缩放因子解包为Triton内核所需的float32格式
    assert sf_packed.dtype == torch.int32  # 断言int32类型
    assert len(sf_packed.shape) == 2  # 断言2维

    N, K = weight_shape  # 解包权重形状
    block_n, block_k = block_size  # 解包块大小
    n_groups = ceil_div(N, block_n)  # N方向组数
    k_groups = ceil_div(K, block_k)  # K方向组数

    mn_repeat, k_div_4 = sf_packed.shape  # 解包缩放形状
    k_packed = k_div_4 * 4  # K方向打包后的元素数

    # Unpack int32 -> 4x uint8 -> float32
    # Each uint8 represents an exponent in UE8M0 format  # 每个uint8表示UE8M0格式的指数
    sf_u8 = sf_packed.contiguous().view(torch.uint8).view(mn_repeat, k_packed)  # 视图转换
    sf_fp32 = (sf_u8.to(torch.int32) << 23).view(torch.float32)  # 指数转换为float32

    # Handle row dimension - may have 128x replication or direct mapping  # 处理行维度 - 可能有128倍复制或直接映射
    if mn_repeat == N:  # 行维度有128倍复制
        # Rows are replicated 128 times, take every 128th row
        # sf_fp32 shape: (N, k_packed) -> (n_groups, k_packed)
        # Select representative rows at indices 0, 128, 256, ...  # 每128行取一行代表
        indices = torch.arange(0, N, block_n, device=sf_packed.device)  # 采样索引
        sf_fp32 = sf_fp32.index_select(0, indices)  # 选择代表行
    elif mn_repeat == n_groups:  # 已经是正确的n_groups格式
        # Already in the correct n_groups format  # 格式正确
        pass
    else:  # 格式不匹配
        raise ValueError(
            f"Unexpected scale shape: sf_packed.shape={sf_packed.shape}, "
            f"weight_shape={weight_shape}, block_size={block_size}"
        )

    # Crop k dimension to expected size (remove padding if any)  # 裁剪K维度到期望大小
    sf_fp32 = sf_fp32[:, :k_groups].contiguous()  # 裁剪并确保连续

    return sf_fp32  # 返回解包结果


def aiter_w8a8_block_fp8_linear(  # AITER W8A8分块FP8线性层
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    block_size: List[int],  # 块大小
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
) -> torch.Tensor:  # 返回线性层输出
    """AITER W8A8分块FP8线性层实现，支持AITER和Triton混合后端。"""  # 中文函数说明
    # assert input_scale is None  # 断言（已注释）
    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 输出形状

    n, k = weight.shape  # 获取权重形状

    if _use_aiter_bpreshuffle_gfx95:  # gfx95上使用AITER
        use_triton = use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k)  # 检查是否使用Triton
    else:  # 其他平台
        use_triton = True  # 使用Triton

    # if input_scale not None, input is quanted  # 如果input_scale不为None，输入已量化
    if input_scale is not None:  # 输入已量化
        q_input = input_2d  # 直接使用
        x_scale = input_scale  # 直接使用缩放因子
        if not use_triton:  # 不使用Triton时需要转置缩放因子
            x_scale = x_scale.transpose(-1, -2).contiguous().view(*x_scale.shape)
    else:  # 输入未量化
        q_input, x_scale = aiter_per1x128_quant(  # AITER 1x128量化
            input_2d,
            quant_dtype=aiter.dtypes.fp8,  # FP8量化
            transpose_scale=not use_triton,  # 根据后端决定是否转置
        )

    if use_triton:  # 使用Triton后端
        gemm_a8w8_blockscale_op = triton_gemm_a8w8_blockscale  # Triton分块缩放GEMM
    else:  # 使用AITER bpreshuffle
        # TODO(1am9trash), to deal with chance of this branch changes  # 处理此分支可能的变更
        gemm_a8w8_blockscale_op = gemm_a8w8_blockscale_bpreshuffle  # AITER bpreshuffle GEMM

    output = gemm_a8w8_blockscale_op(  # 执行GEMM
        q_input,
        weight,
        x_scale,
        weight_scale,
        dtype=torch.bfloat16 if input_scale is not None else input.dtype,  # 输出类型
    )

    if bias is not None:  # 添加偏置
        output += bias

    return output.to(  # 转换类型并重塑形状
        dtype=torch.bfloat16 if input_scale is not None else input_2d.dtype
    ).view(*output_shape)


def triton_w8a8_block_fp8_linear(  # Triton W8A8分块FP8线性层
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    block_size: List[int],  # 块大小
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
) -> torch.Tensor:  # 返回线性层输出
    """Triton W8A8分块FP8线性层实现。"""  # 中文函数说明
    assert input_scale is None  # 断言无预计算输入缩放
    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 输出形状

    q_input, x_scale = per_token_group_quant_fp8(  # 逐token组FP8量化
        input_2d, block_size[1], column_major_scales=False  # 行主序缩放
    )
    output = w8a8_block_fp8_matmul_triton(  # Triton分块FP8矩阵乘法
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input_2d.dtype
    )
    if bias is not None:  # 添加偏置
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)  # 返回结果


@lru_cache(maxsize=1)  # LRU缓存
def _get_triton_mxfp8_downcast():  # 获取Triton MXFP8降精度函数
    """获取Triton Kernels的MXFP8降精度转换函数。"""  # 中文函数说明
    try:  # 尝试导入
        from triton_kernels.numerics_details.mxfp import downcast_to_mxfp  # 导入MXFP8降精度
    except Exception as err:  # 导入失败
        raise RuntimeError(
            "MXFP8 quantization requires triton_kernels with MXFP8 support."
        ) from err  # 抛出错误
    return downcast_to_mxfp  # 返回函数


def mxfp8_group_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # MXFP8组量化
    """Quantize a 2D contiguous tensor to MXFP8 with UE8M0 scales per group (32)."""  # 将2维连续张量量化为MXFP8，每组(32)一个UE8M0缩放因子
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"  # 断言2维
    assert x.is_contiguous(), "MXFP8 quantization requires a contiguous 2D tensor."  # 断言连续
    _, k = x.shape  # 获取K维度
    assert k % 32 == 0, f"{k=} must be divisible by 32"  # K必须能被32整除
    downcast_to_mxfp = _get_triton_mxfp8_downcast()  # 获取降精度函数
    q_input, scale_u8 = downcast_to_mxfp(x, torch.float8_e4m3fn, axis=1)  # 执行MXFP8量化
    return q_input.contiguous(), scale_u8.contiguous()  # 返回量化结果和缩放因子


def _pack_mxfp8_scales(scale_u8: torch.Tensor) -> torch.Tensor:  # 打包MXFP8缩放因子为tl.dot_scaled所需的布局
    """将MXFP8的UE8M0缩放因子打包为tl.dot_scaled所需的布局。"""  # 中文函数说明
    # Pack (M, K//32) UE8M0 scales into the layout expected by tl.dot_scaled.  # 将(M, K//32)的UE8M0缩放因子打包
    assert scale_u8.dim() == 2, f"Expected 2D scale tensor, got {scale_u8.dim()}D"  # 断言2维
    scale_u8 = scale_u8.contiguous()  # 确保连续
    m, k_groups = scale_u8.shape  # 获取形状
    assert (
        k_groups % 4 == 0
    ), f"{k_groups=} must be divisible by 4 (K must be multiple of 128)"  # K必须是128的倍数

    scale_m = ceil_div(m, 128)  # M方向组数
    if m % 128 != 0:  # 需要填充
        pad_rows = scale_m * 128 - m  # 填充行数
        pad = torch.full(  # 创建填充张量
            (pad_rows, k_groups),
            127,  # 127表示UE8M0中的1.0
            dtype=scale_u8.dtype,
            device=scale_u8.device,
        )
        scale_u8 = torch.cat([scale_u8, pad], dim=0)  # 拼接填充

    scale_k = k_groups // 4  # K方向打包后的组数
    scale_u8 = scale_u8.view(scale_m, 128, scale_k, 4)  # 重塑形状
    scale_u8 = scale_u8.view(scale_m, 4, 32, scale_k, 4)  # 进一步重塑
    packed = scale_u8.permute(0, 3, 2, 1, 4).contiguous()  # 排列维度
    return packed.view(1, scale_m, scale_k, 2, 256)  # 返回打包结果


@register_custom_op(
    op_name="triton_mxfp8_block_scaled_matmul",
    mutates_args=[],
    fake_impl=lambda a, a_scale, b, b_scale, output_dtype, block_m=128, block_n=256, block_k=128, num_stages=None: (  # noqa: E501
        a.new_empty((a.shape[0], b.shape[0]), dtype=output_dtype)
    ),
)
def triton_mxfp8_block_scaled_matmul(  # Triton MXFP8分块缩放矩阵乘法
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
    """Opaque custom op wrapper to prevent Dynamo tracing Triton grid math."""  # 不透明自定义算子包装，防止Dynamo追踪Triton网格计算
    return mxfp8_block_scaled_matmul_triton(  # 调用Triton实现
        a,
        a_scale,
        b,
        b_scale,
        output_dtype=output_dtype,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
    )


def _raw_triton_mxfp8_blockscaled_linear(  # Triton MXFP8分块缩放线性层的原始实现
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
    output_dtype: Optional[torch.dtype] = None,  # 输出数据类型
) -> torch.Tensor:  # 返回线性层输出
    """Triton MXFP8分块缩放线性层的原始实现。"""  # 中文函数说明
    if not (_is_cuda and (_is_sm100_supported or _is_sm120_supported)):  # 检查硬件支持
        raise RuntimeError("MXFP8 dense linear requires Blackwell GPUs (SM100/SM120).")

    input_2d = input.view(-1, input.shape[-1]).contiguous()  # 重塑为2维并确保连续
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 输出形状

    block_m = 128  # M方向块大小
    block_n = 256 if weight.shape[0] % 256 == 0 else 128  # N方向块大小
    block_k = 128  # K方向块大小

    m, k = input_2d.shape  # 获取输入形状
    n, k_w = weight.shape  # 获取权重形状
    assert k == k_w, f"{k=} does not match {k_w=}"  # 断言K一致
    assert k % 128 == 0, f"{k=} must be divisible by 128 for MXFP8"  # K必须128整除
    assert n % block_n == 0, f"{n=} must be divisible by {block_n}"  # N必须整除
    assert weight.dtype == torch.float8_e4m3fn, "MXFP8 weight must be FP8 E4M3."  # 断言权重类型
    assert weight_scale.dtype == torch.uint8, "MXFP8 weight_scale must be UE8M0 uint8."  # 断言缩放类型

    if input_scale is None:  # 需要量化输入
        q_input, x_scale_u8 = mxfp8_group_quantize(input_2d)  # MXFP8组量化
    else:  # 输入已量化
        q_input = input_2d  # 直接使用
        x_scale_u8 = input_scale  # 直接使用缩放因子
        assert x_scale_u8.dtype == torch.uint8, "MXFP8 input_scale must be UE8M0 uint8."  # 断言类型
        assert x_scale_u8.shape == (m, k // 32)  # 断言形状

    if output_dtype is None:  # 未指定输出类型
        if input_2d.dtype in (torch.float16, torch.bfloat16, torch.float32):  # 浮点输入
            output_dtype = input_2d.dtype  # 使用输入类型
        else:  # 非浮点
            output_dtype = torch.bfloat16  # 默认bfloat16

    if m % block_m != 0:  # M需要填充
        pad_rows = ceil_div(m, block_m) * block_m - m  # 填充行数
        q_input = torch.cat(  # 拼接零填充
            [
                q_input,
                torch.zeros((pad_rows, k), device=q_input.device, dtype=q_input.dtype),
            ],
            dim=0,
        )
        pad_scale = torch.full(  # 缩放因子填充
            (pad_rows, k // 32),
            127,  # 127表示UE8M0中的1.0
            device=x_scale_u8.device,
            dtype=x_scale_u8.dtype,
        )
        x_scale_u8 = torch.cat([x_scale_u8, pad_scale], dim=0)  # 拼接填充

    a_scale_packed = _pack_mxfp8_scales(x_scale_u8)  # 打包输入缩放因子
    b_scale_packed = _pack_mxfp8_scales(weight_scale)  # 打包权重缩放因子

    num_stages = 1 if _is_sm120_supported else (4 if _is_sm100_supported else 1)  # 自动选择阶段数
    output = triton_mxfp8_block_scaled_matmul(  # 执行MXFP8矩阵乘法
        q_input,
        a_scale_packed,
        weight.contiguous(),
        b_scale_packed,
        output_dtype=output_dtype,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
    )
    output = output[:m, :]  # 裁剪填充
    if bias is not None:  # 添加偏置
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)  # 返回结果


@register_custom_op(
    op_name="triton_mxfp8_blockscaled_linear",
    mutates_args=[],
    fake_impl=lambda input, weight, weight_scale, input_scale=None, bias=None, output_dtype=None: (
        input.new_empty(
            (*input.shape[:-1], weight.shape[0]),
            dtype=(output_dtype if output_dtype is not None else input.dtype),
        )
    ),
)
def triton_mxfp8_blockscaled_linear(  # Triton MXFP8分块缩放线性层
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
    output_dtype: Optional[torch.dtype] = None,  # 输出数据类型
) -> torch.Tensor:  # 返回线性层输出
    """Opaque custom-op wrapper to prevent Dynamo guards on MXFP8 padding branches."""  # 不透明自定义算子包装，防止Dynamo在MXFP8填充分支上设置守卫
    return _raw_triton_mxfp8_blockscaled_linear(  # 调用原始实现
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        input_scale=input_scale,
        bias=bias,
        output_dtype=output_dtype,
    )


def flashinfer_mxfp8_blockscaled_linear(  # FlashInfer MXFP8分块缩放线性层
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    bias: Optional[torch.Tensor] = None,  # 偏置
    output_dtype: Optional[torch.dtype] = None,  # 输出数据类型
) -> torch.Tensor:  # 返回线性层输出
    """MXFP8 dense linear via FlashInfer mm_mxfp8."""  # 通过FlashInfer mm_mxfp8实现MXFP8线性层
    input_2d = input.view(-1, input.shape[-1]).contiguous()  # 重塑为2维
    output_shape = [*input.shape[:-1], weight.shape[0]]  # 输出形状

    m, k = input_2d.shape  # 获取输入形状
    n, k_w = weight.shape  # 获取权重形状
    if k != k_w:  # K不一致
        raise ValueError(f"Input K={k} does not match weight K={k_w}.")
    if k % 32 != 0:  # K必须32整除
        raise ValueError(f"K={k} must be divisible by 32 for MXFP8.")
    if weight.dtype != torch.float8_e4m3fn:  # 权重类型错误
        raise TypeError("MXFP8 weight must be FP8 E4M3.")

    if input_scale is None:  # 需要量化输入
        q_input, x_scale_u8 = flashinfer_mxfp8_quantize(  # FlashInfer MXFP8量化
            input_2d, is_sf_swizzled_layout=True, alignment=32
        )
    else:  # 输入已量化
        q_input = input_2d  # 直接使用
        x_scale_u8 = input_scale.contiguous()  # 确保连续

    if output_dtype is None:  # 未指定输出类型
        if input_2d.dtype in (torch.float16, torch.bfloat16, torch.float32):  # 浮点输入
            output_dtype = input_2d.dtype  # 使用输入类型
        else:  # 非浮点
            output_dtype = torch.bfloat16  # 默认bfloat16

    # Ensure transposed tensors are contiguous for FlashInfer's internal runner.  # 确保转置张量连续
    weight_t = weight.contiguous().t()  # 转置权重

    if get_fp8_gemm_runner_backend().is_flashinfer_trtllm():  # TRTLLM后端

        weight_scale_t = weight_scale.contiguous().view(-1)  # 展平缩放因子
        output = flashinfer_mm_mxfp8(  # FlashInfer MXFP8矩阵乘法
            q_input,
            weight_t,
            x_scale_u8,
            weight_scale_t,
            out_dtype=output_dtype,
            use_8x4_sf_layout=False,
            backend="trtllm",
        )
    elif get_fp8_gemm_runner_backend().is_flashinfer_cutlass():  # CUTLASS后端
        weight_scale_t = (  # 转置缩放因子
            weight_scale.contiguous().t()
            if weight_scale.ndim == 2
            else weight_scale.contiguous()
        )
        output = flashinfer_mm_mxfp8(  # FlashInfer MXFP8矩阵乘法
            q_input,
            weight_t,
            x_scale_u8,
            weight_scale_t,
            out_dtype=output_dtype,
            use_8x4_sf_layout=False,
            backend="cutlass",
        )

    if bias is not None:  # 添加偏置
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)  # 返回结果


def dequant_mxfp4(  # MXFP4反量化函数
    w_block: torch.Tensor,  # 量化权重块
    w_scale: torch.Tensor,  # 缩放因子
    out_dtype,  # 输出数据类型
) -> torch.Tensor:  # 返回反量化结果
    """
    :param w_block: (batch, n, k, 16), uint8, pack two mxfp4 into one byte
    :param w_scale: (batch, n, k), uint8
    :return: (batch, n, k * 32), float32
    """  # MXFP4反量化，将打包的4位权重解量化为float32

    assert w_block.dtype == torch.uint8  # 断言uint8类型
    assert w_scale.dtype == torch.uint8  # 断言uint8类型

    batch, n, k, pack_dim = w_block.shape  # 解包形状
    batch_, n_, k_ = w_scale.shape  # 解包缩放形状
    assert pack_dim == 16  # 断言打包维度为16
    assert batch == batch_  # 断言batch一致
    assert n == n_  # 断言n一致
    assert k == k_  # 断言k一致

    out_raw = MXFP4QuantizeUtil.dequantize(  # 执行MXFP4反量化
        quantized_data=w_block, scale=w_scale, dtype=out_dtype, block_sizes=[32]
    )
    return out_raw.reshape(batch, n, k * 32)  # 重塑并返回


def input_to_float8(  # 将输入张量转换为FP8（逐张量量化）
    x: torch.Tensor, dtype: torch.dtype = fp8_dtype  # 输入张量和目标类型
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化结果和缩放因子
    """This function quantizes input values to float8 values with tensor-wise quantization."""  # 将输入值量化为FP8，使用逐张量量化
    min_val, max_val = x.aminmax()  # 计算最小最大值
    amax = torch.maximum(min_val.abs(), max_val.abs()).float().clamp(min=1e-12)  # 计算绝对值最大值

    if _is_fp8_fnuz:  # FNUZ格式
        dtype = fp8_dtype  # 使用FNUZ类型
        fp_max = fp8_max  # 使用FNUZ最大值
    else:  # FN格式
        finfo = torch.finfo(dtype)  # 获取类型信息
        fp_max = finfo.max  # 获取最大值

    scale = fp_max / amax  # 计算缩放因子
    x_scl_sat = (x.float() * scale).clamp(min=-fp_max, max=fp_max)  # 缩放并钳制
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()  # 返回量化结果和缩放因子倒数


def block_quant_to_tensor_quant(  # 将分块量化转换为逐张量量化
    x_q_block: torch.Tensor,  # 分块量化张量
    x_s: torch.Tensor,  # 分块缩放因子
    block_size: List[int],  # 块大小
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回逐张量量化结果和缩放因子
    """This function converts block-wise quantization to tensor-wise quantization.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise quantization scale.
    Note only float8 is supported for now.
    """  # 将分块量化转换为逐张量量化，目前仅支持FP8
    block_n, block_k = block_size[0], block_size[1]  # 解包块大小
    n, k = x_q_block.shape  # 获取形状
    n_tiles = (n + block_n - 1) // block_n  # N方向块数
    k_tiles = (k + block_k - 1) // block_k  # K方向块数
    assert n_tiles == x_s.shape[0]  # 断言
    assert k_tiles == x_s.shape[1]  # 断言

    x_dq_block = x_q_block.to(torch.float32)  # 反量化为float32

    x_dq_block_tiles = [  # 创建分块列表
        [
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)  # K方向遍历
        ]
        for j in range(n_tiles)  # N方向遍历
    ]

    for i in range(k_tiles):  # 遍历K块
        for j in range(n_tiles):  # 遍历N块
            x_dq_block_tiles[j][i][:, :] = x_dq_block_tiles[j][i] * x_s[j][i]  # 乘以缩放因子

    x_q_tensor, scale = (  # 重新执行逐张量量化
        scaled_fp8_quant(x_dq_block)
        if _is_cuda
        else input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    )
    return x_q_tensor, scale  # 返回结果


def block_quant_dequant(  # 分块量化反量化函数
    x_q_block: torch.Tensor,  # 分块量化张量
    x_s: torch.Tensor,  # 分块缩放因子
    block_size: List[int],  # 块大小
    dtype: torch.dtype,  # 输出数据类型
) -> torch.Tensor:  # 返回反量化结果
    """This function converts block-wise quantization to unquantized.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The output is an unquantized tensor with dtype.
    """  # 将分块量化转换为未量化张量
    block_n, block_k = block_size[0], block_size[1]  # 解包块大小
    *_, n, k = x_q_block.shape  # 获取形状

    # ... n_scale k_scale -> ... (n_scale block_n) (k_scale block_k)  # 缩放因子扩展到与量化数据相同的形状
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]  # 裁剪到正确大小

    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)  # 反量化并转换类型


def requant_weight_ue8m0_inplace(weight, weight_scale_inv, weight_block_size):  # 原地重新量化UE8M0权重
    """原地重新量化权重为UE8M0格式。"""  # 中文函数说明
    assert isinstance(weight, torch.nn.Parameter)  # 断言为Parameter
    assert isinstance(weight_scale_inv, torch.nn.Parameter)  # 断言为Parameter

    new_weight, new_weight_scale_inv = requant_weight_ue8m0(  # 重新量化
        weight.to(weight_scale_inv.device), weight_scale_inv, weight_block_size
    )

    offloader.update_param(weight, new_weight)  # 更新权重参数
    weight_scale_inv.data = new_weight_scale_inv  # 更新缩放因子


def requant_weight_ue8m0(  # 重新量化UE8M0权重
    weight: torch.Tensor,  # 权重张量
    weight_scale_inv: torch.Tensor,  # 权重缩放因子倒数
    weight_block_size: List[int],  # 权重块大小
):
    """将权重重新量化为UE8M0缩放格式。"""  # 中文函数说明
    assert weight_block_size == [128, 128]  # 断言块大小为[128, 128]

    *_, n, k = weight.shape  # 获取形状

    weight_dequant = block_quant_dequant(  # 反量化权重
        weight,
        weight_scale_inv,
        weight_block_size,
        torch.bfloat16,
    )

    out_w, out_s = quant_weight_ue8m0(  # 重新量化为UE8M0
        weight_dequant=weight_dequant,
        weight_block_size=weight_block_size,
    )

    out_s = transform_scale_ue8m0(out_s, mn=out_w.shape[-2])  # 转换缩放布局

    return out_w, out_s  # 返回结果


def quant_weight_ue8m0(  # UE8M0权重量化
    weight_dequant: torch.Tensor,  # 反量化权重
    weight_block_size: List[int],  # 块大小
):
    """将反量化权重量化为UE8M0缩放格式的FP8权重。"""  # 中文函数说明
    assert weight_block_size == [128, 128]  # 断言块大小
    assert (
        weight_dequant.dtype == torch.bfloat16
    ), f"{weight_dequant.dtype=} {weight_dequant.shape=}"  # 断言bfloat16

    *batch_dims, n, k = weight_dequant.shape  # 解包形状

    weight_dequant_flat = weight_dequant.view((-1, k))  # 展平batch维度
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat)  # 逐块FP8量化

    out_w = out_w_flat.view((*batch_dims, n, k))  # 重塑权重形状
    out_s = out_s_flat.view(  # 重塑缩放因子形状
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )

    return out_w, out_s  # 返回结果


def transform_scale_ue8m0_inplace(param, mn):  # 原地转换UE8M0缩放布局
    """原地转换UE8M0缩放因子的布局。"""  # 中文函数说明
    param.data = transform_scale_ue8m0(param.data, mn=mn)  # 转换并更新


# NOTE copy and modified from DeepGEMM  # 参考自DeepGEMM
def transform_scale_ue8m0(sf, mn, use_torch_impl: bool = False):  # 转换UE8M0缩放布局
    """将UE8M0缩放因子转换为TMA对齐的MN主序打包布局。"""  # 中文函数说明
    import deep_gemm.utils.layout  # 导入DeepGEMM布局工具

    get_mn_major_tma_aligned_packed_ue8m0_tensor = (  # 选择实现
        _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
        if use_torch_impl
        else deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor
    )

    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)  # 选择每128行的代表行
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)  # 转换为TMA对齐打包布局

    # In sgl-deep-gemm, the C++ deepgemm path returns through DLPack which collapses the stride
    # of size-1 trailing dims to 1 (happens when packed_sf_k == 1, i.e.
    # K <= block_k * 4). Restore the TMA-aligned stride so the deepgemm
    # assertion sf.stride(-1) == get_tma_aligned_size(mn, element_size) holds.  # 修复DLPack导致的步长问题
    if not use_torch_impl and sf.shape[-1] == 1:  # 需要修复步长
        from deep_gemm.utils import get_tma_aligned_size  # 导入TMA对齐大小函数

        aligned_mn = get_tma_aligned_size(sf.shape[-2], sf.element_size())  # 计算对齐大小
        if sf.stride(-1) != aligned_mn:  # 步长不对
            new_stride = list(sf.stride())  # 获取当前步长
            new_stride[-1] = aligned_mn  # 修复步长
            sf = sf.as_strided(sf.shape, tuple(new_stride))  # 应用新步长
    return sf  # 返回转换结果


# Copied from DeepGEMM tests  # 参考自DeepGEMM测试
def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(  # PyTorch实现的UE8M0缩放布局转换
    x: torch.Tensor,  # 输入缩放因子
) -> torch.Tensor:  # 返回转换后的缩放因子
    """PyTorch实现的MN主序TMA对齐UE8M0打包布局转换。"""  # 中文函数说明
    from deep_gemm.utils import align, get_tma_aligned_size  # 导入对齐工具

    assert x.dtype == torch.float and x.dim() in (2, 3)  # 断言float且2/3维

    # First, convert into UE8M0 `uint8_t`  # 转换为UE8M0 uint8
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)  # 提取指数部分

    # Second, make padded packed tensors  # 创建填充打包张量
    mn, k = x.shape[-2], x.shape[-1]  # 获取形状
    remove_dim = False  # 是否需要移除维度
    if x.dim() == 2:  # 2维需要增加batch维度
        x, remove_dim = x.unsqueeze(0), True  # 增加维度
    b = x.shape[0]  # 获取batch大小
    aligned_mn = get_tma_aligned_size(mn, 4)  # 计算MN对齐大小
    aligned_k = align(k, 4)  # 计算K对齐大小
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)  # 填充张量
    padded[:, :mn, :k] = ue8m0_tensor  # 复制数据
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)  # 打包为int32

    # Finally, transpose  # 最后转置
    transposed = torch.zeros(
        (b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int
    ).mT  # 创建转置张量
    transposed[:, :, :] = padded  # 复制数据
    aligned_x = transposed[:, :mn, :]  # 裁剪
    return aligned_x.squeeze(0) if remove_dim else aligned_x  # 返回结果


def inverse_transform_scale_ue8m0(sf_packed, mn):  # UE8M0缩放因子的逆转换
    """将打包的UE8M0缩放因子逆向转换为float32格式，并验证一致性。"""  # 中文函数说明
    sf_fp32 = _inverse_transform_scale_ue8m0_impl(sf_packed)  # 执行逆转换
    # Can call consistency check every time since this is only called on startup  # 启动时可以每次验证一致性
    sf_packed_recreated = transform_scale_ue8m0(sf_fp32, mn=mn, use_torch_impl=True)  # 重新创建
    assert torch.all(
        sf_packed == sf_packed_recreated
    ), f"{sf_packed=} {sf_packed_recreated=} {sf_fp32=}"  # 断言一致性
    return sf_fp32  # 返回float32缩放因子


# Inverse impl can refer to DeepGEMM's torch impl in get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl  # 逆转换可参考DeepGEMM的torch实现
def _inverse_transform_scale_ue8m0_impl(sf_packed):  # UE8M0缩放因子逆转换的实现
    """
    NOTE: We assume k is aligned
    :param sf_packed: (scale_mn, scale_k/4) int32
    :return: (scale_mn, scale_k), float32
    """  # 假设K已对齐，将int32打包格式转换为float32
    if len(sf_packed.shape) == 3:  # 3维batch
        return torch.stack(
            [_inverse_transform_scale_ue8m0_impl(x) for x in sf_packed], dim=0
        )

    block_size = 128  # 块大小
    assert len(sf_packed.shape) == 2, f"{sf_packed.shape=}"  # 断言2维
    assert sf_packed.dtype == torch.int32  # 断言int32

    mn_repeat_128, k_div_4 = sf_packed.shape  # 解包形状
    mn = mn_repeat_128 // block_size  # 计算实际MN
    k = k_div_4 * 4  # 计算实际K

    # packed u8 -> fp32  # 从打包uint8转换到float32
    sf_u8 = sf_packed.contiguous().flatten().view(torch.uint8).view(mn_repeat_128, k)  # 解包
    sf_fp32 = (sf_u8.to(torch.int32) << 23).view(torch.float32)  # 指数转float32

    # remove repeat  # 移除128倍重复
    sf_reshaped = sf_fp32.view(mn, block_size, k)  # 重塑
    sf_unrepeated = sf_reshaped[:, 0:1, :]  # 取每128行的第一行
    if not torch.all(sf_unrepeated == sf_reshaped):  # 验证重复行一致
        from sglang.srt.debug_utils.dumper import get_tensor_info

        raise AssertionError(
            f"sf_unrepeated != sf_reshaped ({get_tensor_info(sf_unrepeated)=} {get_tensor_info(sf_reshaped)=})"
        )
    sf_unrepeated = sf_unrepeated.squeeze(1).contiguous()  # 移除重复维度

    assert sf_unrepeated.shape == (mn, k)  # 断言形状
    return sf_unrepeated  # 返回结果


# COPIED FROM DeepGEMM  # 参考自DeepGEMM
def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 逐块转换为FP8
    """将张量按128x128的块进行FP8量化，返回量化结果和UE8M0缩放因子。"""  # 中文函数说明
    assert x.dim() == 2  # 断言2维
    m, n = x.shape  # 获取形状
    x_padded = torch.zeros(
        (ceil_align(m, 128), ceil_align(n, 128)), dtype=x.dtype, device=x.device
    )  # 填充到128的倍数
    x_padded[:m, :n] = x  # 复制数据
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)  # 重塑为块视图
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)  # 计算每块最大值
    sf = ceil_to_ue8m0(x_amax / 448.0)  # 计算UE8M0缩放因子
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)  # 量化为FP8
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(  # 返回结果
        x_view.size(0), x_view.size(2)
    )


# COPIED FROM DeepGEMM  # 参考自DeepGEMM
def ceil_to_ue8m0(x: torch.Tensor):  # 向上取整为UE8M0格式（2的幂次）
    """将缩放因子向上取整为2的幂次（UE8M0格式要求）。"""  # 中文函数说明
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))  # 2^ceil(log2(|x|))


def channel_quant_to_tensor_quant(  # 将通道量化转换为逐张量量化
    x_q_channel: torch.Tensor,  # 通道量化张量
    x_s: torch.Tensor,  # 通道缩放因子
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回逐张量量化结果和缩放因子
    """将逐通道量化转换为逐张量量化。"""  # 中文函数说明
    x_dq_channel = x_q_channel.to(torch.float32) * x_s  # 反量化
    x_q_tensor, scale = (  # 重新逐张量量化
        scaled_fp8_quant(x_dq_channel)
        if _is_cuda
        else input_to_float8(x_dq_channel, dtype=x_q_channel.dtype)
    )
    return x_q_tensor, scale  # 返回结果


def _process_scaled_mm_output(output, input_2d_shape, output_shape):  # 处理scaled_mm输出
    """处理torch._scaled_mm的输出，裁剪填充并重塑形状。"""  # 中文函数说明
    if type(output) is tuple and len(output) == 2:  # 输出为元组
        output = output[0]  # 取第一个元素
    return torch.narrow(output, 0, 0, input_2d_shape[0]).view(*output_shape)  # 裁剪并重塑


def _apply_fallback_scaled_mm(  # 应用回退的缩放矩阵乘法
    qinput,  # 量化输入
    weight,  # 权重
    x_scale,  # 输入缩放因子
    weight_scale,  # 权重缩放因子
    input_2d_shape,  # 输入2维形状
    output_shape,  # 输出形状
    bias,  # 偏置
    input_dtype,  # 输入数据类型
):
    """使用未融合的反量化方式执行缩放矩阵乘法的回退实现。"""  # 中文函数说明
    global TORCH_DEVICE_IDENTITY  # 声明全局变量
    if TORCH_DEVICE_IDENTITY is None:  # 如果未初始化
        TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32, device=weight.device)  # 创建虚拟缩放

    output = torch._scaled_mm(  # 调用torch缩放矩阵乘法
        qinput,
        weight,
        scale_a=TORCH_DEVICE_IDENTITY,  # 使用虚拟缩放
        scale_b=TORCH_DEVICE_IDENTITY,  # 使用虚拟缩放
        out_dtype=torch.float32,  # float32输出
    )

    output = _process_scaled_mm_output(output, input_2d_shape, output_shape)  # 处理输出
    x_scale = torch.narrow(x_scale, 0, 0, input_2d_shape[0])  # 裁剪缩放因子

    output = output * x_scale * weight_scale.t()  # 应用缩放因子
    if bias is not None:  # 添加偏置
        output = output + bias
    return output.to(dtype=input_dtype)  # 转换类型并返回


def apply_fp8_linear(  # 应用FP8线性层
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    input_scale_ub: Optional[torch.Tensor] = None,  # 输入缩放上界
    bias: Optional[torch.Tensor] = None,  # 偏置
    cutlass_fp8_supported: bool = cutlass_fp8_supported(),  # CUTLASS FP8是否支持
    use_per_token_if_dynamic: bool = False,  # 动态量化时是否逐token
    pad_output: Optional[bool] = None,  # 是否填充输出
    compressed_tensor_quant: bool = False,  # 是否使用压缩张量量化
) -> torch.Tensor:  # 返回线性层输出
    """应用FP8线性层，支持多种量化模式和后端。"""  # 中文函数说明
    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.  # 填充输入因为_scaled_mm在batch>16时性能更好
    if pad_output is None:  # 未指定填充
        pad_output = not cutlass_fp8_supported and not get_bool_env_var(
            "SGLANG_ENABLE_TORCH_COMPILE"
        )  # 自动决定
    output_padding = 17 if pad_output else None  # 填充大小

    # View input as 2D matrix for fp8 methods  # 将输入视为2维矩阵
    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维
    output_shape = [*input.shape[:-1], weight.shape[1]]  # 输出形状

    if compressed_tensor_quant:  # 压缩张量量化模式
        # Maybe apply padding to output, see comment in __init__  # 可能填充输出
        num_token_padding = output_padding  # token填充数
        if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:  # 逐通道缩放
            num_token_padding = None  # 不需要填充
        # For static per-tensor activation scales when using inductor compiler,
        # use pure PyTorch ops instead of the opaque sgl_kernel quant kernel.
        # Inductor fuses these with surrounding ops (RMSNorm, residual add),
        # eliminating a separate kernel launch per linear layer.
        # weight_scale shape does not matter here -- it is only used in the
        # GEMM epilogue, not in the activation quant fusion. Only activates when
        # piecewise_cuda_graph_compiler=inductor; eager PCG and decode both
        # use the faster custom kernel.  # 使用inductor时用纯PyTorch操作代替自定义量化内核
        if (
            input_scale is not None
            and input_scale.numel() == 1
            and get_global_server_args().piecewise_cuda_graph_compiler == "inductor"
        ):
            qinput = (  # 纯PyTorch量化
                (input_2d * input_scale.reciprocal())
                .clamp(min=fp8_min, max=fp8_max)
                .to(fp8_dtype)
            )
            x_scale = input_scale  # 使用输入缩放
        else:  # 使用自定义量化内核
            qinput, x_scale = scaled_fp8_quant(
                input_2d,
                input_scale,
                num_token_padding=num_token_padding,
                use_per_token_if_dynamic=use_per_token_if_dynamic,
            )
    else:  # 非压缩张量量化
        # cutlass w8a8 fp8 sgl-kernel only supports per-token scale  # CUTLASS W8A8 FP8仅支持逐token缩放
        if input_scale is not None:  # 有预计算缩放因子
            assert input_scale.numel() == 1  # 断言标量
            # broadcast per-tensor scale to per-token scale when supporting cutlass  # 广播逐张量缩放为逐token缩放
            qinput, x_scale = static_quant_fp8(
                input_2d, input_scale, repeat_scale=cutlass_fp8_supported
            )
        else:  # 无预计算缩放因子
            # default use per-token quantization if dynamic  # 动态量化默认逐token
            if _is_cuda:  # CUDA平台
                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)  # SGLang逐token量化
            else:  # 非CUDA平台
                # TODO(kkhuang): temporarily enforce per-tensor activation scaling if weight is per-tensor scaling
                # final solution should be: 1. add support to per-tensor activation scaling.
                # 2. solve the torch.compile error from weight_scale.numel() == 1 and x_scale.numel() > 1 (below line#308)  # 临时方案
                if _is_hip and weight_scale.numel() == 1:  # HIP且权重逐张量缩放
                    qinput, x_scale = scaled_fp8_quant(  # 使用逐张量量化
                        input_2d,
                        input_scale,
                        use_per_token_if_dynamic=use_per_token_if_dynamic,
                    )
                else:  # 其他情况
                    qinput, x_scale = per_token_group_quant_fp8(  # 逐token组量化
                        input_2d, group_size=input_2d.shape[1]
                    )

    if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:  # CUTLASS且逐通道缩放
        cutlass_compatible_b = weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0  # 检查兼容性
        if not cutlass_compatible_b or use_triton_w8a8_fp8_kernel:  # 不兼容或使用Triton
            # Massage the input to be 2D  # 重塑为2维
            qinput = qinput.view(-1, qinput.shape[-1])  # 重塑
            output = triton_scaled_mm(  # Triton缩放矩阵乘法
                qinput, weight, x_scale, weight_scale, input.dtype, bias
            )
        else:  # CUTLASS兼容
            output = fp8_scaled_mm(  # CUTLASS缩放矩阵乘法
                qinput,
                weight,
                x_scale,
                weight_scale,
                out_dtype=input.dtype,
                bias=bias,
            )
        return output.view(*output_shape)  # 返回结果

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token  # torch._scaled_mm仅支持逐张量，逐通道或逐token需要回退
    per_tensor_weights = weight_scale.numel() == 1  # 逐张量权重缩放
    # When the number of token is 1,
    # per-token scale has shape (1, 1), per-tensor scale has shape (1) or ().  # token数为1时区分逐token和逐张量缩放
    per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2  # 逐张量激活缩放

    if (
        use_per_token_if_dynamic
        and not per_tensor_weights
        and not per_tensor_activations
        and (USE_ROWWISE_TORCH_SCALED_MM or _use_aiter)
    ):  # 逐token逐通道缩放
        # into this sector means use dynamic per-token-per-channel quant
        # per-token scale quant for input matrix, every row(one token) have one scale factor
        # per-channel scale quant for weight matrix, every col(one channel) have one scale factor  # 逐token逐通道量化
        if _use_aiter:  # AITER后端
            # gemm_a8w8_bpreshuffle(XQ, WQ, x_scale, w_scale, dtype)
            # XQ -> input tensor, shape = (m, k)
            # WQ -> weight tensor, shape = (n, k), with preshuffe get better perf
            # x_scale -> input scale tensor, shape = (m, 1)
            # w_scale -> weight scale tensor, shape = (n ,1)
            # dtype -> output dtype  # AITER GEMM参数说明
            output = gemm_a8w8_bpreshuffle(
                XQ=qinput,
                WQ=weight.T,
                x_scale=x_scale,
                w_scale=weight_scale,
                dtype=input.dtype,
            )
            if bias is not None:  # 添加偏置
                output += bias
            return _process_scaled_mm_output(output, input_2d.shape, output_shape)  # 处理并返回
        else:  # torch._scaled_mm行方向缩放
            # For now validated on ROCm platform
            # fp8 rowwise scaling in torch._scaled_mm is introduced in
            # https://github.com/pytorch/pytorch/pull/144432 using hipBLASLt
            # and ROCm 6.3, which only exists in torch 2.7 and above.
            # For CUDA platform please validate if the
            # torch._scaled_mm support rowwise scaled GEMM
            # Fused GEMM_DQ Rowwise GEMM  # ROCm平台已验证，CUDA待验证
            output = torch._scaled_mm(  # 行方向缩放矩阵乘法
                qinput,
                weight,
                out_dtype=input.dtype,
                scale_a=x_scale,
                scale_b=weight_scale.t(),
                bias=bias,
            )
            return _process_scaled_mm_output(output, input_2d.shape, output_shape)  # 处理并返回

    if per_tensor_weights and per_tensor_activations:  # 逐张量权重和激活缩放
        # Fused GEMM_DQ; _scaled_mm with torch.compile requires len(weight_scale.shape) == len(x_scale.shape)  # 融合GEMM_DQ
        if weight_scale.ndim == 0 and x_scale.ndim == 1:  # 维度不匹配
            weight_scale = weight_scale.unsqueeze(0)  # 增加维度
        output = torch._scaled_mm(  # 逐张量缩放矩阵乘法
            qinput,
            weight,
            out_dtype=input.dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias,
        )
        return _process_scaled_mm_output(output, input_2d.shape, output_shape)  # 处理并返回

    # Fallback for channelwise case, where we use unfused DQ
    # due to limitations with scaled_mm  # 逐通道情况回退到未融合反量化

    # Symmetric quantized GEMM by definition computes the following:
    #   C = (s_x * X) (s_w * W) + bias
    # This is equivalent to dequantizing the weights and activations
    # before applying a GEMM.
    #
    # In order to compute quantized operands, a quantized kernel
    # will rewrite the above like so:
    #   C = s_w * s_x * (X * W) + bias
    #
    # For the scaled_mm fallback case, we break this down, since it
    # does not support s_w being a vector.  # 对称量化GEMM的数学推导和回退实现
    return _apply_fallback_scaled_mm(  # 回退实现
        qinput,
        weight,
        x_scale,
        weight_scale,
        input_2d.shape,
        output_shape,
        bias,
        input.dtype,
    )


def can_auto_enable_marlin_fp8() -> bool:  # 检查是否可以自动启用Marlin FP8
    """检测当前GPU是否可以自动启用Marlin FP8（SM80-SM88）。"""  # 中文函数说明
    try:  # 尝试获取设备信息
        major, minor = get_device_capability()  # 获取计算能力
        sm = major * 10 + minor  # 计算SM版本
        return 80 <= sm < 89  # SM80-SM88可以自动启用
    except Exception:  # 获取失败
        return False  # 返回False


def apply_fp8_ptpc_linear(  # 应用FP8逐token逐通道线性层
    input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # 输入或已量化输入元组
    weight: torch.Tensor,  # 权重张量
    weight_scale: torch.Tensor,  # 权重缩放因子
    input_scale: Optional[torch.Tensor] = None,  # 输入缩放因子
    input_scale_ub: Optional[torch.Tensor] = None,  # 输入缩放上界
    bias: Optional[torch.Tensor] = None,  # 偏置
    cutlass_fp8_supported: bool = cutlass_fp8_supported(),  # CUTLASS FP8支持
    use_per_token_if_dynamic: bool = False,  # 动态量化时是否逐token
    pad_output: Optional[bool] = None,  # 是否填充输出
    compressed_tensor_quant: bool = False,  # 是否压缩张量量化
) -> torch.Tensor:  # 返回线性层输出
    """FP8 per-token per-channel linear. Only used with the aiter (ROCm) backend."""  # FP8逐token逐通道线性层，仅AITER(ROCm)后端使用
    # Handle pre-quantized (fp8_tensor, scale) tuple from fused RMSNorm+Quant  # 处理融合RMSNorm+Quant的预量化输入
    if isinstance(input, tuple):  # 输入为元组
        q_input, x_scale = input  # 解包
        q_input = q_input.view(-1, q_input.shape[-1])  # 重塑为2维
        output_shape = [*q_input.shape[:-1], weight.shape[0]]  # 输出形状
        output = aiter.gemm_a8w8_bpreshuffle(  # AITER GEMM
            q_input, weight, x_scale, weight_scale, None, torch.bfloat16
        )
        if bias is not None:  # 添加偏置
            output = output + bias
        return output.view(*output_shape)  # 返回结果

    # View input as 2D matrix for fp8 methods  # 将输入视为2维矩阵
    input_2d = input.view(-1, input.shape[-1])  # 重塑为2维

    # weight is transposed (K, N)  # 权重已转置(K, N)
    output_shape = [*input.shape[:-1], weight.shape[1]]  # 输出形状

    q_input, x_scale = aiter.per_token_quant_hip(input_2d, quant_dtype=aiter.dtypes.fp8)  # AITER逐token量化

    per_tensor_weights = (weight_scale.numel() == 1) and weight_scale.dim() < 2  # 逐张量权重缩放
    per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2  # 逐张量激活缩放

    if not (per_tensor_weights and per_tensor_activations):  # 非逐张量
        # weight is in (N, K)  # 权重形状为(N, K)
        output_shape = [*input.shape[:-1], weight.shape[0]]  # 更新输出形状

    output = aiter.gemm_a8w8_bpreshuffle(  # AITER GEMM
        q_input, weight, x_scale, weight_scale, None, input.dtype
    )
    if bias is not None:  # 添加偏置
        output = output + bias
    return output.view(*output_shape)  # 返回结果


def validate_fp8_block_shape(  # 验证FP8分块形状
    layer: torch.nn.Module,  # 层对象
    input_size: int,  # 输入大小
    output_size: int,  # 输出大小
    input_size_per_partition: int,  # 每个分区的输入大小
    output_partition_sizes: list[int],  # 输出分区大小列表
    block_size: list[int],  # 块大小
) -> None:  # 无返回值
    """Validate block quantization shapes for tensor parallelism."""  # 验证张量并行的分块量化形状
    from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入TP世界大小

    tp_size = getattr(layer, "tp_size", get_tensor_model_parallel_world_size())  # 获取TP大小
    block_n, block_k = block_size[0], block_size[1]  # 解包块大小

    # Required by row parallel  # 行并行要求
    if (
        tp_size > 1
        and input_size // input_size_per_partition == tp_size
        and input_size_per_partition % block_k != 0
    ):  # 检查输入分区是否可被block_k整除
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} "
            f"is not divisible by weight quantization block_k = {block_k}."
        )

    # Required by column parallel or enabling merged weights  # 列并行或合并权重要求
    is_tp_split = tp_size > 1 and output_size // sum(output_partition_sizes) == tp_size  # 是否TP分割
    is_merged_gemm = len(output_partition_sizes) > 1  # 是否合并GEMM
    if is_tp_split or is_merged_gemm:  # 需要检查
        sizes_to_check = output_partition_sizes  # 检查的分区大小
        if not is_tp_split and is_merged_gemm:  # 仅合并GEMM
            # In case of merged matrices, we allow the last
            # matrix to not be a multiple of block size  # 合并矩阵允许最后一个不整除block_size
            sizes_to_check = output_partition_sizes[:-1]  # 检查除最后一个外的所有分区
        for output_partition_size in sizes_to_check:  # 遍历检查
            if output_partition_size % block_n != 0:  # 不可整除
                raise ValueError(
                    f"Weight output_partition_size = "
                    f"{output_partition_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
