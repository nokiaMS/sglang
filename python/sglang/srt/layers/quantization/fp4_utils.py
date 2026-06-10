# FP4量化工具模块：提供FP4量化函数注册、FP4 GEMM运行后端枚举及配置初始化功能
from __future__ import annotations  # 启用延迟注解评估，支持类型提示中的前向引用

import logging  # 导入日志模块
from enum import Enum  # 导入枚举基类，用于定义后端类型枚举
from typing import TYPE_CHECKING, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch张量库

from sglang.srt.utils.common import is_sm100_supported, is_sm120_supported  # 导入SM架构支持检测函数
from sglang.srt.utils.custom_op import register_custom_op_from_extern  # 导入自定义算子注册函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免运行时循环依赖
    from sglang.srt.server_args import ServerArgs  # 导入服务器参数类型

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


fp4_quantize = None  # 初始化FP4量化函数为None，后续尝试从flashinfer加载
try:  # 尝试导入flashinfer的FP4量化实现
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize  # 从flashinfer导入FP4量化函数

    _flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"  # 根据GPU架构选择后端：SM100用cute-dsl，否则用cuda

    def _round_up(x: int, y: int) -> int:  # 将x向上取整到y的最近倍数
        return ((x + y - 1) // y) * y  # 计算向上取整结果

    def _flashinfer_fp4_quantize_impl(  # flashinfer FP4量化的实际实现函数
        input: torch.Tensor,  # 输入张量
        global_scale: Optional[torch.Tensor] = None,  # 全局缩放因子，可选
        sf_vec_size: int = 16,  # 缩放因子向量大小，默认16
        sf_use_ue8m0: bool = False,  # 是否使用UE8M0格式缩放因子，默认否
        is_sf_swizzled_layout: bool = True,  # 是否使用交错布局存储缩放因子，默认是
        is_sf_8x4_layout: bool = False,  # 是否使用8x4布局存储缩放因子，默认否
        enable_pdl: Optional[bool] = None,  # 是否启用PDL（可编程数据流），可选
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回量化张量和缩放因子张量的元组
        return _flashinfer_fp4_quantize(  # 调用flashinfer的FP4量化函数
            input=input,  # 传入输入张量
            global_scale=global_scale,  # 传入全局缩放因子
            sf_vec_size=sf_vec_size,  # 传入缩放因子向量大小
            sf_use_ue8m0=sf_use_ue8m0,  # 传入是否使用UE8M0格式
            is_sf_swizzled_layout=is_sf_swizzled_layout,  # 传入是否使用交错布局
            is_sf_8x4_layout=is_sf_8x4_layout,  # 传入是否使用8x4布局
            enable_pdl=enable_pdl,  # 传入是否启用PDL
            backend=_flashinfer_fp4_quantize_backend,  # 传入后端类型
        )

    def _flashinfer_fp4_quantize_fake(  # flashinfer FP4量化的伪实现，用于torch.compile的fake tensor模式
        input: torch.Tensor,  # 输入张量
        global_scale: Optional[torch.Tensor] = None,  # 全局缩放因子，可选
        sf_vec_size: int = 16,  # 缩放因子向量大小，默认16
        sf_use_ue8m0: bool = False,  # 是否使用UE8M0格式缩放因子，默认否
        is_sf_swizzled_layout: bool = True,  # 是否使用交错布局存储缩放因子，默认是
        is_sf_8x4_layout: bool = False,  # 是否使用8x4布局存储缩放因子，默认否
        enable_pdl: Optional[bool] = None,  # 是否启用PDL，可选
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回伪量化张量和伪缩放因子张量的元组
        is_column_major = input.stride(-2) == 1  # 判断输入是否为列主序（stride倒数第二维为1）
        if is_column_major:  # 如果是列主序
            m = input.shape[-1]  # 列主序时m为最后一维大小
            K = input.shape[-2]  # 列主序时K为倒数第二维大小
        else:  # 如果是行主序
            m = input.numel() // input.shape[-1]  # 行主序时m为总元素数除以最后一维
            K = input.shape[-1]  # 行主序时K为最后一维大小
        if is_column_major:  # 列主序时创建量化张量
            x_q = input.new_empty((*input.shape[:-2], K // 2, m), dtype=torch.uint8)  # 列主序量化结果，每个FP4占4bit，所以K//2
        else:  # 行主序时创建量化张量
            x_q = input.new_empty((*input.shape[:-1], K // 2), dtype=torch.uint8)  # 行主序量化结果
        if is_sf_swizzled_layout:  # 如果使用交错布局
            row_size = 8 if is_sf_8x4_layout else 128  # 8x4布局时行大小为8，否则为128
            sf_rows = _round_up(m, row_size)  # 向上取整行数到row_size的倍数
            sf_cols = _round_up(K // sf_vec_size, 4)  # 向上取整列数到4的倍数
        else:  # 非交错布局
            sf_rows = m  # 行数等于m
            sf_cols = K // sf_vec_size  # 列数等于K除以缩放因子向量大小
        if is_column_major:  # 列主序时创建缩放因子张量
            sf = input.new_empty((sf_cols, sf_rows), dtype=torch.uint8)  # 列主序缩放因子
        else:  # 行主序时创建缩放因子张量
            sf = input.new_empty((sf_rows, sf_cols), dtype=torch.uint8)  # 行主序缩放因子
        return x_q, sf  # 返回伪量化张量和伪缩放因子张量

    fp4_quantize = register_custom_op_from_extern(  # 将flashinfer的FP4量化注册为自定义算子
        _flashinfer_fp4_quantize_impl,  # 实际实现函数
        op_name="flashinfer_fp4_quantize",  # 自定义算子名称
        fake_impl=_flashinfer_fp4_quantize_fake,  # 伪实现函数，用于torch.compile
    )
except ImportError:  # 如果flashinfer未安装，则保持fp4_quantize为None
    fp4_quantize = None  # flashinfer不可用时设为None


class Fp4GemmRunnerBackend(Enum):  # FP4 GEMM运行后端枚举类
    """Enum for FP4 GEMM runner backend selection."""  # FP4 GEMM运行后端选择枚举

    AUTO = "auto"  # 自动选择后端
    CUTLASS = "cutlass"  # 使用CUTLASS后端
    FLASHINFER_CUDNN = "flashinfer_cudnn"  # 使用FlashInfer cuDNN后端
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"  # 使用FlashInfer CuTe-dsl后端
    FLASHINFER_CUTLASS = "flashinfer_cutlass"  # 使用FlashInfer CUTLASS后端
    FLASHINFER_TRTLLM = "flashinfer_trtllm"  # 使用FlashInfer TensorRT-LLM后端

    def is_auto(self) -> bool:  # 判断是否为自动选择模式
        return self == Fp4GemmRunnerBackend.AUTO  # 比较当前值是否为AUTO

    def is_cutlass(self) -> bool:  # 判断是否为CUTLASS后端
        return self == Fp4GemmRunnerBackend.CUTLASS  # 比较当前值是否为CUTLASS

    def is_flashinfer_cudnn(self) -> bool:  # 判断是否为FlashInfer cuDNN后端
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN  # 比较当前值是否为FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:  # 判断是否为FlashInfer CUTLASS后端
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS  # 比较当前值是否为FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:  # 判断是否为FlashInfer TensorRT-LLM后端
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM  # 比较当前值是否为FLASHINFER_TRTLLM

    def is_flashinfer_cutedsl(self) -> bool:  # 判断是否为FlashInfer CuTe-dsl后端
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL  # 比较当前值是否为FLASHINFER_CUTEDSL

    def is_flashinfer(self) -> bool:  # 判断是否为任意FlashInfer后端
        return self.value.startswith("flashinfer_")  # 检查值是否以"flashinfer_"开头

    def get_flashinfer_backend(self) -> str:  # 获取传递给FlashInfer mm_fp4 API的后端字符串
        """Get the backend string to pass to FlashInfer's mm_fp4 API.
        # 获取传递给FlashInfer mm_fp4 API的后端字符串

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        将SGLang用户面向的后端名称映射为FlashInfer的API名称
        Examples:  # 示例
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
            'flashinfer_cutedsl' -> 'cute-dsl'
        """
        if self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL:  # CuTe-dsl后端需要特殊处理
            return "cute-dsl"  # 返回cute-dsl字符串
        if self.value.startswith("flashinfer_"):  # 其他flashinfer后端
            return self.value.removeprefix("flashinfer_")  # 去掉"flashinfer_"前缀后返回
        else:  # 非flashinfer后端
            return self.value  # 直接返回原始值


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None  # 全局FP4 GEMM运行后端配置，初始为None


def initialize_fp4_gemm_config(server_args: ServerArgs) -> None:  # 根据服务器参数初始化FP4 GEMM配置
    """Initialize FP4 GEMM configuration from server args."""  # 从服务器参数初始化FP4 GEMM配置
    global FP4_GEMM_RUNNER_BACKEND  # 声明使用全局变量

    backend = server_args.fp4_gemm_runner_backend  # 从服务器参数获取后端配置
    if backend == "auto":  # 如果是自动选择模式
        if is_sm120_supported():  # 如果支持SM120（Blackwell架构）
            # flashinfer_cutlass produces NaN in dense MLP layers with
            # heterogeneous batches on SM120 (Blackwell).  cudnn is stable.
            # flashinfer_cutlass在SM120(Blackwell)上的异构批次密集MLP层中会产生NaN。cudnn是稳定的。
            # See: https://github.com/sgl-project/sglang/issues/20043
            backend = "flashinfer_cudnn"  # SM120上使用cudnn后端
        elif is_sm100_supported():  # 如果支持SM100架构
            backend = "flashinfer_cutedsl"  # SM100上使用cute-dsl后端
        else:  # 其他架构
            backend = "flashinfer_cutlass"  # 默认使用flashinfer_cutlass后端

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)  # 将后端字符串转换为枚举并赋值给全局变量


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:  # 获取当前FP4 GEMM运行后端配置
    """Get the current FP4 GEMM runner backend."""  # 获取当前FP4 GEMM运行后端
    global FP4_GEMM_RUNNER_BACKEND  # 声明使用全局变量
    if FP4_GEMM_RUNNER_BACKEND is None:  # 如果尚未初始化
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO  # 默认设为AUTO
    return FP4_GEMM_RUNNER_BACKEND  # 返回当前后端配置
