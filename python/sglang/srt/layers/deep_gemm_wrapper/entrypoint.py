# DeepGEMM入口模块 - 提供各种GEMM操作的统一接口，封装DeepGEMM库的调用
import logging  # 导入日志模块
from contextlib import contextmanager  # 导入上下文管理器装饰器
from typing import Any, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.deep_gemm_wrapper import compile_utils  # 导入编译工具模块
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401  # 导入配置常量
    DEEPGEMM_BLACKWELL,  # Blackwell架构标志
    DEEPGEMM_NEED_TMA_ALIGNED_SCALES,  # 是否需要TMA对齐缩放
    DEEPGEMM_SCALE_UE8M0,  # UE8M0缩放格式标志
    ENABLE_JIT_DEEPGEMM,  # JIT DeepGEMM启用标志
)
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类

logger = logging.getLogger(__name__)  # 创建日志记录器

if ENABLE_JIT_DEEPGEMM:  # 如果启用了JIT DeepGEMM
    import deep_gemm  # 导入deep_gemm库
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401  # 导入TMA对齐张量工具

_SANITY_CHECK = envs.SGLANG_DEEPGEMM_SANITY_CHECK.get()  # 是否启用健全性检查


# TODO maybe rename these functions
# TODO 可能重命名这些函数
def grouped_gemm_nt_f8f8bf16_masked(  # 分组FP8矩阵乘法（掩码模式）
    lhs: Tuple[torch.Tensor, torch.Tensor],  # 左侧FP8数据及缩放因子
    rhs: Tuple[torch.Tensor, torch.Tensor],  # 右侧FP8数据及缩放因子
    out: torch.Tensor,  # 输出张量
    masked_m: torch.Tensor,  # 掩码M张量
    expected_m: int,  # 期望的M值
    overlap_args: Optional[Any] = None,  # 重叠参数（用于2批次重叠）
    max_block_n: int = 256,  # N方向最大块大小
    recipe_a: Optional[Tuple[int, int]] = None,  # FP4量化配方A
    recipe_b: Optional[Tuple[int, int]] = None,  # FP4量化配方B
):
    num_groups, _, k = lhs[0].shape  # 获取分组数和K维度
    _, n, _ = rhs[0].shape  # 获取N维度
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED  # 设置内核类型

    _sanity_check_input(lhs)  # 对左侧输入进行健全性检查
    _sanity_check_input(rhs)  # 对右侧输入进行健全性检查

    lhs = _ensure_cuda(lhs)  # 确保左侧数据在CUDA上
    rhs = _ensure_cuda(rhs)  # 确保右侧数据在CUDA上

    with compile_utils.deep_gemm_execution_hook(  # 执行钩子，可能触发JIT编译
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(  # 配置SM数量
            overlap_args.num_sms if overlap_args is not None else None  # 使用重叠参数中的SM数或None
        ):

            fp4_kwargs = {}  # FP4参数字典
            if recipe_a is not None:  # 如果有FP4配方A
                fp4_kwargs["recipe_a"] = recipe_a  # 添加到参数字典
            if recipe_b is not None:  # 如果有FP4配方B
                fp4_kwargs["recipe_b"] = recipe_b  # 添加到参数字典

            return deep_gemm.fp8_m_grouped_gemm_nt_masked(  # 调用FP8掩码分组GEMM
                lhs,  # 左侧数据
                rhs,  # 右侧数据
                out,  # 输出
                masked_m,  # 掩码M
                expected_m,  # 期望M值
                **fp4_kwargs,  # FP4参数
                **(  # 重叠参数（如果存在）
                    dict(
                        enable_overlap=True,  # 启用重叠
                        max_block_n=max_block_n,  # 最大N块大小
                        signal=overlap_args.signal,  # 信号量
                    )
                    if overlap_args is not None  # 如果有重叠参数
                    else {}  # 否则为空
                ),
            )


def _ensure_cuda(  # 确保张量对在CUDA设备上
    pair: Tuple[torch.Tensor, torch.Tensor],  # 张量对（数据和缩放因子）
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        pair[0].cuda() if not pair[0].is_cuda else pair[0],  # 第一个张量：不在CUDA则移到CUDA
        pair[1].cuda() if not pair[1].is_cuda else pair[1],  # 第二个张量：不在CUDA则移到CUDA
    )


def grouped_gemm_nt_bf16_masked(  # 分组BF16矩阵乘法（掩码模式）
    a: torch.Tensor,  # 左侧BF16张量
    b: torch.Tensor,  # 右侧BF16张量
    d: torch.Tensor,  # 输出张量
    masked_m: torch.Tensor,  # 掩码M张量
    expected_m: int,  # 期望的M值
):
    num_groups, _, k = a.shape  # 获取分组数和K维度
    _, n, _ = b.shape  # 获取N维度
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED  # 设置内核类型

    with compile_utils.deep_gemm_execution_hook(  # 执行钩子，可能触发JIT编译
        expected_m, n, k, num_groups, kernel_type
    ):
        return deep_gemm.m_grouped_bf16_gemm_nt_masked(  # 调用BF16掩码分组GEMM
            a,  # 左侧数据
            b,  # 右侧数据
            d,  # 输出
            masked_m,  # 掩码M
            expected_m,  # 期望M值
        )


def grouped_gemm_nt_f8f8bf16_contig(  # 分组FP8矩阵乘法（连续模式）
    lhs: Tuple[torch.Tensor, torch.Tensor],  # 左侧FP8数据及缩放因子
    rhs: Tuple[torch.Tensor, torch.Tensor],  # 右侧FP8数据及缩放因子
    out: torch.Tensor,  # 输出张量
    m_indices: torch.Tensor,  # M索引张量
    recipe_a: Optional[Tuple[int, int]] = None,  # FP4量化配方A
    recipe_b: Optional[Tuple[int, int]] = None,  # FP4量化配方B
):
    m, k = lhs[0].shape  # 获取M和K维度
    num_groups, n, _ = rhs[0].shape  # 获取分组数和N维度
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG  # 设置内核类型

    if m == 0:  # 如果M为0，无需计算
        return  # 直接返回

    _sanity_check_input(lhs)  # 对左侧输入进行健全性检查
    _sanity_check_input(rhs)  # 对右侧输入进行健全性检查

    fp4_kwargs = {}  # FP4参数字典
    if recipe_a is not None:  # 如果有FP4配方A
        fp4_kwargs["recipe_a"] = recipe_a  # 添加到参数字典
    if recipe_b is not None:  # 如果有FP4配方B
        fp4_kwargs["recipe_b"] = recipe_b  # 添加到参数字典

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):  # 执行钩子
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(  # 调用FP8连续分组GEMM
            lhs, rhs, out, m_indices, **fp4_kwargs
        )


def grouped_gemm_nt_bf16_contig(  # 分组BF16矩阵乘法（连续模式）
    a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, m_indices: torch.Tensor
):
    m, k = a.shape  # 获取M和K维度
    num_groups, n, _ = b.shape  # 获取分组数和N维度
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG  # 设置内核类型

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):  # 执行钩子
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)  # 调用BF16连续分组GEMM


def gemm_nt_f8f8bf16(  # 普通FP8矩阵乘法
    lhs: Tuple[torch.Tensor, torch.Tensor],  # 左侧FP8数据及缩放因子
    rhs: Tuple[torch.Tensor, torch.Tensor],  # 右侧FP8数据及缩放因子
    out: torch.Tensor,  # 输出张量
):
    m, k = lhs[0].shape  # 获取M和K维度
    n, _ = rhs[0].shape  # 获取N维度
    num_groups = 1  # 非分组GEMM，分组数为1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16  # 设置内核类型

    _sanity_check_input(lhs)  # 对左侧输入进行健全性检查
    _sanity_check_input(rhs)  # 对右侧输入进行健全性检查

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):  # 执行钩子
        deep_gemm.fp8_gemm_nt(  # 调用FP8 GEMM
            lhs,  # 左侧数据
            rhs,  # 右侧数据
            out,  # 输出
        )


def gemm_nt_bf16bf16f32(  # 普通BF16矩阵乘法（FP32输出）
    lhs: torch.Tensor,  # 左侧BF16张量
    rhs: torch.Tensor,  # 右侧BF16张量
    out: torch.Tensor,  # 输出FP32张量
):
    m, k = lhs.shape  # 获取M和K维度
    n, _ = rhs.shape  # 获取N维度
    num_groups = 1  # 非分组GEMM，分组数为1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16F32  # 设置内核类型

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):  # 执行钩子
        deep_gemm.bf16_gemm_nt(lhs, rhs, out)  # 调用BF16 GEMM


def tf32_hc_prenorm_gemm(  # TF32 Householder预归一化矩阵乘法
    x: torch.Tensor,  # 输入BF16张量
    fn: torch.Tensor,  # 权重FP32张量
    out: torch.Tensor,  # 输出张量
    sqrsum: torch.Tensor,  # 平方和输出张量
    num_splits: Optional[int],  # 分割数（None表示不分割）
):
    m, k = x.shape  # 获取M和K维度
    n, _ = fn.shape  # 获取N维度
    num_splits_key = num_splits if num_splits is not None else 0  # 将None转换为0作为键
    kernel_type = compile_utils.DeepGemmKernelType.TF32_HC_PRENORM_GEMM  # 设置内核类型

    if m == 0:  # 如果M为0，无需计算
        return  # 直接返回

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_splits_key, kernel_type):  # 执行钩子
        deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=num_splits)  # 调用TF32 HC预归一化GEMM


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):  # 更新DeepGEMM配置
    compile_utils.update_deep_gemm_config(gpu_id, server_args)  # 委托给compile_utils模块


@contextmanager
def configure_deep_gemm_num_sms(num_sms):  # 配置DeepGEMM使用的SM数量上下文管理器
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:  # 如果不需要配置或DeepGEMM未启用
        yield  # 不做任何配置
    else:
        original_num_sms = deep_gemm.get_num_sms()  # 保存原始SM数量
        deep_gemm.set_num_sms(num_sms)  # 设置新的SM数量
        try:
            yield  # 执行实际计算
        finally:
            deep_gemm.set_num_sms(original_num_sms)  # 恢复原始SM数量


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):  # FP8输入健全性检查
    if not _SANITY_CHECK:  # 如果未启用健全性检查
        return  # 直接返回

    x, x_scale = x_fp8  # 解包数据和缩放因子

    if x_scale.dtype == torch.int:  # 如果缩放因子是整数类型（UE8M0格式）
        return  # 跳过检查

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0  # 导入UE8M0向上取整工具

    x_scale_ceil = ceil_to_ue8m0(x_scale)  # 将缩放因子向上取整到UE8M0
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"  # 断言缩放因子已正确对齐
