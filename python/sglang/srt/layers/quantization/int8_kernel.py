# INT8量化内核模块 - 提供INT8量化和矩阵乘法的Triton加速实现
# 包含逐token量化、逐token组量化以及分块INT8矩阵乘法等功能

import functools  # 导入functools模块，用于缓存装饰器
import json  # 导入json模块，用于读取配置文件
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于文件路径操作
from typing import Any, Dict, List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.utils import get_device_name, is_cuda  # 从sglang工具模块导入设备名称获取和CUDA检测函数

_is_cuda = is_cuda()  # 检测当前是否为CUDA环境
if _is_cuda:  # 如果是CUDA环境
    # Temporary  # 临时处理
    try:  # 尝试导入v2版本的量化内核
        from sgl_kernel import sgl_per_token_group_quant_8bit  # 从sgl_kernel导入8bit组量化函数

        enable_sgl_per_token_group_quant_8bit = True  # 启用v2版本的8bit组量化
    except ImportError:  # 如果导入失败
        from sgl_kernel import sgl_per_token_group_quant_int8  # 从sgl_kernel导入INT8组量化函数（旧版本）

        enable_sgl_per_token_group_quant_8bit = False  # 禁用v2版本的8bit组量化

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


@triton.jit  # Triton JIT编译装饰器，将函数编译为GPU内核
def _per_token_quant_int8(  # 逐token INT8量化的Triton内核函数
    x_ptr,  # 输入张量指针
    xq_ptr,  # 量化后输出张量指针
    scale_ptr,  # 缩放因子输出指针
    x_sum_ptr,  # 输入行求和结果指针
    stride_x,  # 输入张量的行步长
    stride_xq,  # 量化输出张量的行步长
    N,  # 输入张量的列数
    CAL_SUM: tl.constexpr,  # 是否计算行求和的常量标志
    BLOCK: tl.constexpr,  # 块大小的常量
):
    # Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282  # 参考自lmdeploy项目
    row_id = tl.program_id(0)  # 获取当前程序的行ID

    cols = tl.arange(0, BLOCK)  # 生成列偏移量序列
    mask = cols < N  # 创建列掩码，防止越界访问

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)  # 加载输入数据并转换为float32
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)  # 计算当前行绝对值最大值，避免除零
    scale_x = absmax / 127  # 计算量化缩放因子（INT8最大值为127）
    x_q = x * (127 / absmax)  # 将输入值缩放到INT8范围
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)  # 四舍五入并转换为int8类型
    if CAL_SUM:  # 如果需要计算行求和
        x_sum = tl.sum(x, axis=0)  # 计算当前行所有元素的和
        tl.store(x_sum_ptr + row_id, x_sum.to(x_sum_ptr.dtype.element_ty))  # 存储行求和结果

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)  # 存储量化结果
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))  # 存储缩放因子


def per_token_quant_int8(x, scale_dtype=torch.float32, cal_sum=False):  # 逐token INT8量化函数
    """对输入张量执行逐token的INT8量化，返回量化结果和缩放因子。"""  # 中文函数说明
    M = x.numel() // x.shape[-1]  # 计算行数（token数）
    N = x.shape[-1]  # 获取列数（特征维度）
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)  # 创建INT8量化的输出张量
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype)  # 创建缩放因子输出张量
    if cal_sum:  # 如果需要计算行求和
        x_sum = torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype)  # 创建行求和输出张量
    else:  # 否则
        x_sum = None  # 行求和输出设为None
    BLOCK = triton.next_power_of_2(N)  # 计算大于等于N的最小2的幂次作为块大小
    # heuristics for number of warps  # 启发式计算warp数量
    num_warps = min(max(BLOCK // 256, 1), 8)  # 根据块大小确定warp数量，范围为1-8

    assert x.is_contiguous()  # 确保输入张量是连续的
    _per_token_quant_int8[(M,)](  # 调用Triton内核进行量化
        x,  # 输入张量
        x_q,  # 量化输出张量
        scales,  # 缩放因子输出
        x_sum,  # 行求和输出
        stride_x=x.stride(-2),  # 输入行步长
        stride_xq=x_q.stride(-2),  # 量化输出行步长
        N=N,  # 列数
        CAL_SUM=cal_sum,  # 是否计算行求和
        BLOCK=BLOCK,  # 块大小
        num_warps=num_warps,  # warp数量
        num_stages=1,  # 流水线阶段数
    )
    if cal_sum:  # 如果需要计算行求和
        return x_q, scales, x_sum  # 返回量化结果、缩放因子和行求和
    else:  # 否则
        return x_q, scales  # 仅返回量化结果和缩放因子


@triton.jit  # Triton JIT编译装饰器
def _per_token_group_quant_int8(  # 逐token组INT8量化的Triton内核函数
    # Pointers to inputs and output  # 输入和输出的指针
    y_ptr,  # 输入张量指针
    y_q_ptr,  # 量化输出指针
    y_s_ptr,  # 缩放因子输出指针
    # Stride of input  # 输入张量的步长
    y_stride,  # 输入行步长
    # Columns of input  # 输入的列数
    N,  # 每组的元素数量
    # Avoid to divide zero  # 避免除零
    eps,  # 最小值阈值
    # Information for int8  # INT8相关信息
    int8_min,  # INT8最小值
    int8_max,  # INT8最大值
    # Meta-parameters  # 元参数
    BLOCK: tl.constexpr,  # 块大小的常量
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into int8 values.
    """  # Triton加速的逐token组量化函数，将张量值转换为INT8值
    # Map the program id to the row of X and Y it should compute.  # 将程序ID映射到要计算的行
    g_id = tl.program_id(0)  # 获取当前程序的组ID
    y_ptr += g_id * y_stride  # 移动输入指针到当前组位置
    y_q_ptr += g_id * y_stride  # 移动量化输出指针到当前组位置
    y_s_ptr += g_id  # 移动缩放因子指针到当前组位置

    cols = tl.arange(0, BLOCK)  # N <= BLOCK  # 生成列偏移序列
    mask = cols < N  # 创建列掩码

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载输入数据并转换为float32
    # Quant  # 量化操作
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)  # 计算当前组绝对值最大值，确保不小于eps
    y_s = _absmax / int8_max  # 计算缩放因子
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)  # 量化并裁剪到INT8范围

    tl.store(y_q_ptr + cols, y_q, mask=mask)  # 存储量化结果
    tl.store(y_s_ptr, y_s)  # 存储缩放因子


def per_token_group_quant_int8(  # 逐token组INT8量化函数
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 量化组大小
    eps: float = 1e-10,  # 避免除零的最小值，默认1e-10
    dtype: torch.dtype = torch.int8,  # 输出数据类型，默认int8
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回量化张量和缩放因子的元组
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8` is supported for now.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """  # 对输入张量执行逐token组量化，将值转换为有符号INT8并返回量化张量和缩放因子
    assert (  # 断言
        x.shape[-1] % group_size == 0  # 最后一维必须能被group_size整除
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"  # 断言输入张量必须是连续的

    iinfo = torch.iinfo(dtype)  # 获取INT8数据类型的信息
    int8_max = iinfo.max  # INT8最大值（127）
    int8_min = iinfo.min  # INT8最小值（-128）

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)  # 创建量化输出张量
    M = x.numel() // group_size  # 计算总组数
    N = group_size  # 每组元素数
    x_s = torch.empty(  # 创建缩放因子输出张量
        x.shape[:-1] + (x.shape[-1] // group_size,),  # 形状为输入形状去掉最后一维加上组数
        device=x.device,  # 设备
        dtype=torch.float32,  # float32类型
    )

    BLOCK = triton.next_power_of_2(N)  # 计算大于等于N的最小2的幂次
    # heuristics for number of warps  # 启发式计算warp数量
    num_warps = min(max(BLOCK // 256, 1), 8)  # 根据块大小确定warp数量
    num_stages = 1  # 流水线阶段数设为1
    _per_token_group_quant_int8[(M,)](  # 调用Triton内核执行量化
        x,  # 输入张量
        x_q,  # 量化输出
        x_s,  # 缩放因子输出
        group_size,  # 组大小
        N,  # 每组元素数
        eps,  # 最小值阈值
        int8_min=int8_min,  # INT8最小值
        int8_max=int8_max,  # INT8最大值
        BLOCK=BLOCK,  # 块大小
        num_warps=num_warps,  # warp数量
        num_stages=num_stages,  # 流水线阶段数
    )

    return x_q, x_s  # 返回量化结果和缩放因子


def sglang_per_token_group_quant_int8(  # SGLang优化的逐token组INT8量化函数
    x: torch.Tensor,  # 输入张量
    group_size: int,  # 量化组大小
    eps: float = 1e-10,  # 避免除零的最小值
    dtype: torch.dtype = torch.int8,  # 输出数据类型
    enable_v2: Optional[bool] = None,  # 是否启用v2内核，默认自动选择
):
    """SGLang优化的逐token组INT8量化，自动选择最优内核实现。"""  # 中文函数说明
    assert (  # 断言
        x.shape[-1] % group_size == 0  # 最后一维必须能被group_size整除
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"  # 断言输入张量必须是连续的

    iinfo = torch.iinfo(dtype)  # 获取INT8数据类型信息
    int8_max = iinfo.max  # INT8最大值
    int8_min = iinfo.min  # INT8最小值

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)  # 创建量化输出张量
    x_s = torch.empty(  # 创建缩放因子输出张量
        x.shape[:-1] + (x.shape[-1] // group_size,),  # 缩放因子形状
        device=x.device,  # 设备
        dtype=torch.float32,  # float32类型
    )

    # Temporary  # 临时处理
    if enable_sgl_per_token_group_quant_8bit:  # 如果启用了v2版本的8bit组量化
        sgl_per_token_group_quant_8bit(  # 调用v2版本内核
            x, x_q, x_s, group_size, eps, int8_min, int8_max, enable_v2=enable_v2
        )
    else:  # 否则使用旧版本
        assert not enable_v2  # 旧版本不支持v2
        sgl_per_token_group_quant_int8(x, x_q, x_s, group_size, eps, int8_min, int8_max)  # 调用旧版本INT8组量化内核

    return x_q, x_s  # 返回量化结果和缩放因子


@triton.jit  # Triton JIT编译装饰器
def _w8a8_block_int8_matmul(  # W8A8分块INT8矩阵乘法的Triton内核
    # Pointers to inputs and output  # 输入和输出的指针
    A,  # 激活张量A的指针
    B,  # 权重张量B的指针
    C,  # 输出张量C的指针
    As,  # A的缩放因子指针
    Bs,  # B的缩放因子指针
    # Shape for matmul  # 矩阵乘法的形状参数
    M,  # M维度（行数）
    N,  # N维度（列数）
    K,  # K维度（内积维度）
    # Block size for block-wise quantization  # 分块量化的块大小
    group_n,  # N方向的块大小
    group_k,  # K方向的块大小
    # Stride for inputs and output  # 输入和输出的步长
    stride_am,  # A的M方向步长
    stride_ak,  # A的K方向步长
    stride_bk,  # B的K方向步长
    stride_bn,  # B的N方向步长
    stride_cm,  # C的M方向步长
    stride_cn,  # C的N方向步长
    stride_As_m,  # As的M方向步长
    stride_As_k,  # As的K方向步长
    stride_Bs_k,  # Bs的K方向步长
    stride_Bs_n,  # Bs的N方向步长
    # Meta-parameters  # 元参数
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小常量
    BLOCK_SIZE_N: tl.constexpr,  # N方向块大小常量
    BLOCK_SIZE_K: tl.constexpr,  # K方向块大小常量
    GROUP_SIZE_M: tl.constexpr,  # M方向组大小常量（用于超级行分组）
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """  # Triton加速的分块量化矩阵乘法内核，对A和B执行点积并将结果存入C

    pid = tl.program_id(axis=0)  # 获取当前程序ID
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M方向的块数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N方向的块数
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每个组内的程序数
    group_id = pid // num_pid_in_group  # 当前程序所属的组ID
    first_pid_m = group_id * GROUP_SIZE_M  # 当前组的第一个M方向块ID
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 当前组的M方向实际大小
    pid_m = first_pid_m + (pid % group_size_m)  # 当前程序的M方向块ID
    pid_n = (pid % num_pid_in_group) // group_size_m  # 当前程序的N方向块ID

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  # A的M方向偏移量
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # B的N方向偏移量
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # K方向偏移量序列
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # A的数据指针
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # B的数据指针

    As_ptrs = As + offs_am * stride_As_m  # A缩放因子的指针
    offs_bsn = offs_bn // group_n  # B缩放因子的N方向偏移
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n  # B缩放因子的指针

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器为零
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  # 遍历K方向的块
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)  # 加载A的当前块
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)  # 加载B的当前块

        k_start = k * BLOCK_SIZE_K  # 当前K块的起始位置
        offs_ks = k_start // group_k  # 缩放因子的K方向偏移
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)  # 加载A的缩放因子
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)  # 加载B的缩放因子

        accumulator += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]  # 执行点积并乘以缩放因子
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 移动A指针到下一个K块
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 移动B指针到下一个K块

    if C.dtype.element_ty == tl.bfloat16:  # 如果输出类型为bfloat16
        c = accumulator.to(tl.bfloat16)  # 转换为bfloat16
    elif C.dtype.element_ty == tl.float16:  # 如果输出类型为float16
        c = accumulator.to(tl.float16)  # 转换为float16
    else:  # 否则
        c = accumulator.to(tl.float32)  # 转换为float32

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # C的M方向偏移
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # C的N方向偏移
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]  # C的输出指针
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  # C的输出掩码
    tl.store(c_ptrs, c, mask=c_mask)  # 存储输出结果


@functools.lru_cache  # LRU缓存装饰器，避免重复计算
def get_w8a8_block_int8_configs(  # 获取W8A8分块INT8内核的优化配置
    N: int, K: int, block_n: int, block_k: int  # 矩阵维度和块大小参数
) -> Optional[Dict[int, Any]]:  # 返回配置字典或None
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """  # 返回W8A8分块INT8内核的优化配置，将批量大小映射到对应的内核配置

    # First look up if an optimized configuration is available in the configs
    # directory  # 首先查找配置目录中是否有优化配置
    device_name = get_device_name().replace(" ", "_")  # 获取设备名称并替换空格
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=int8_w8a8,block_shape=[{block_n}, {block_k}].json"  # 构建配置文件名

    config_file_path = os.path.join(  # 构建配置文件的完整路径
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name  # 当前文件目录下的configs子目录
    )
    if os.path.exists(config_file_path):  # 如果配置文件存在
        with open(config_file_path) as f:  # 打开配置文件
            logger.info(  # 记录信息日志
                "Using configuration from %s for W8A8 Block INT8 kernel.",
                config_file_path,  # 配置文件路径
            )
            # If a configuration has been found, return it  # 如果找到配置则返回
            return {int(key): val for key, val in json.load(f).items()}  # 将JSON配置的键转换为整数

    # If no optimized configuration is available, we will use the default
    # configuration  # 如果没有优化配置，则使用默认配置
    logger.warning(  # 记录警告日志
        (
            "Using default W8A8 Block INT8 kernel config. Performance might be sub-optimal! "
            "Config file not found at %s"
        ),  # 使用默认W8A8分块INT8内核配置，性能可能不是最优的
        config_file_path,  # 未找到的配置文件路径
    )
    return None  # 返回None表示使用默认配置


def w8a8_block_int8_matmul(  # W8A8分块INT8矩阵乘法函数
    A: torch.Tensor,  # 激活张量A
    B: torch.Tensor,  # 权重张量B
    As: torch.Tensor,  # A的缩放因子
    Bs: torch.Tensor,  # B的缩放因子
    block_size: List[int],  # 分块大小列表[block_n, block_k]
    output_dtype: torch.dtype = torch.float16,  # 输出数据类型，默认float16
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
    """  # 执行分块量化的矩阵乘法，接收两个输入张量及其缩放因子，返回指定类型的输出
    assert len(block_size) == 2  # 断言块大小必须为2维
    block_n, block_k = block_size[0], block_size[1]  # 解包块大小

    assert A.shape[-1] == B.shape[-1]  # 断言A和B的内积维度相同
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()  # 断言A和As形状匹配且A连续
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]  # 断言As的列数与分块数匹配
    M = A.numel() // A.shape[-1]  # 计算A的行数

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2  # 断言B是2维且连续，Bs也是2维
    N, K = B.shape  # 获取B的形状
    assert triton.cdiv(N, block_n) == Bs.shape[0]  # 断言Bs的行数与N方向分块数匹配
    assert triton.cdiv(K, block_k) == Bs.shape[1]  # 断言Bs的列数与K方向分块数匹配

    C_shape = A.shape[:-1] + (N,)  # 计算输出形状
    C = A.new_empty(C_shape, dtype=output_dtype)  # 创建输出张量

    configs = get_w8a8_block_int8_configs(N, K, block_size[0], block_size[1])  # 获取优化配置
    if configs:  # 如果有优化配置
        # If an optimal configuration map has been found, look up the
        # optimal config  # 找到最优配置后查找最接近M的配置
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]  # 选择最接近M的配置
    else:  # 否则使用默认配置
        # Default config  # 默认配置
        # Block-wise quant: BLOCK_SIZE_K must be divisible by block_size[1]  # 分块量化要求BLOCK_SIZE_K能被block_k整除
        config = {  # 默认内核配置
            "BLOCK_SIZE_M": 64,  # M方向块大小为64
            "BLOCK_SIZE_N": block_size[0],  # N方向块大小等于block_n
            "BLOCK_SIZE_K": block_size[1],  # K方向块大小等于block_k
            "GROUP_SIZE_M": 32,  # M方向组大小为32
            "num_warps": 4,  # warp数量为4
            "num_stages": 3,  # 流水线阶段数为3
        }

    def grid(META):  # 定义内核启动的网格大小函数
        return (  # 返回网格大小元组
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),  # 总块数
        )

    _w8a8_block_int8_matmul[grid](  # 调用Triton内核执行矩阵乘法
        A,  # 激活张量A
        B,  # 权重张量B
        C,  # 输出张量C
        As,  # A的缩放因子
        Bs,  # B的缩放因子
        M,  # M维度
        N,  # N维度
        K,  # K维度
        block_n,  # N方向块大小
        block_k,  # K方向块大小
        A.stride(-2),  # A的M方向步长
        A.stride(-1),  # A的K方向步长
        B.stride(1),  # B的K方向步长（转置后）
        B.stride(0),  # B的N方向步长（转置后）
        C.stride(-2),  # C的M方向步长
        C.stride(-1),  # C的N方向步长
        As.stride(-2),  # As的M方向步长
        As.stride(-1),  # As的K方向步长
        Bs.stride(1),  # Bs的K方向步长（转置后）
        Bs.stride(0),  # Bs的N方向步长（转置后）
        **config,  # 其他配置参数
    )

    return C  # 返回矩阵乘法结果
