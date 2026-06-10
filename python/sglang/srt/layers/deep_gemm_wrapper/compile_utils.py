# DeepGEMM编译工具模块 - 提供DeepGEMM JIT内核的预编译、预热执行和内存管理功能
import logging  # 导入日志模块
import os  # 导入操作系统模块
import time  # 导入时间模块
from contextlib import contextmanager, nullcontext  # 导入上下文管理器工具
from enum import IntEnum, auto  # 导入枚举类型
from typing import Dict, List, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from tqdm import tqdm  # 导入进度条库

from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入对称内存上下文管理
    disable_symmetric_memory_context,  # 禁用对称内存上下文
    restore_symmetric_memory_context,  # 恢复对称内存上下文
)
from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM  # 导入JIT DeepGEMM启用标志
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # 导入前向传播模式枚举
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类
from sglang.srt.utils import ceil_align, ceil_div, get_available_gpu_memory, is_musa  # 导入工具函数

logger = logging.getLogger(__name__)  # 创建日志记录器

_is_musa = is_musa()  # 判断当前是否为MUSA平台

if ENABLE_JIT_DEEPGEMM:  # 如果启用了JIT DeepGEMM
    import deep_gemm  # 导入deep_gemm库


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))  # 内置M值列表，范围1到16384
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()  # 是否启用JIT预编译
_DO_COMPILE_ALL = True  # 是否编译所有M值
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()  # 是否是节点上第一个rank
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()  # 是否处于预编译阶段
_FAST_WARMUP = envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()  # 是否启用快速预热

# Force redirect deep_gemm cache_dir
# 强制重定向deep_gemm缓存目录
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)  # 设置DeepGEMM JIT缓存目录，默认为~/.cache/deep_gemm

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
# 参考上述提交：NVRTC在某些情况下可能有性能损失，NVCC JIT速度也比参考提交快9倍
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")  # 设置是否使用NVRTC，默认不使用


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):  # 更新DeepGEMM配置
    global _BUILTIN_M_LIST  # 声明全局M值列表
    global _DO_COMPILE_ALL  # 声明全局编译所有标志
    global _IS_FIRST_RANK_ON_NODE  # 声明全局首个rank标志

    _BUILTIN_M_LIST = []  # 重置M值列表

    if _FAST_WARMUP:  # 如果启用快速预热
        # In fast warmup mode, only compile a small set of typical Ms
        # 在快速预热模式下，只编译一小部分典型M值

        # First cover all the small bs to ensure decode performance
        # 首先覆盖所有小批量大小以确保解码性能
        _BUILTIN_M_LIST += list(range(1, 1025))  # 添加1到1024的M值

        # Then cover larger batch sizes with gradually increasing steps
        # 然后用逐步增大的步长覆盖更大的批量大小
        # For example, when chunekd prefill size is 16384
        # 例如，当分块预填充大小为16384时
        # The sampled Ms would be:
        # 采样得到的M值为：
        #   1024, 1026, ... 2046 (step 2)
        #   1024, 1026, ... 2046 (步长 2)
        #   2048, 2052, ... 4092 (step 4)
        #   2048, 2052, ... 4092 (步长 4)
        #   4096, 5004, ... 8184 (step 8)
        #   4096, 5004, ... 8184 (步长 8)
        #   8192, 9008, ... 16384 (step 16)
        #   8192, 9008, ... 16384 (步长 16)
        # Totally 1024 + 1024 / 2 + 2048 / 4 + 4096 / 8 + 8192 / 16 = 3072 kernels
        # 总共 1024 + 1024 / 2 + 2048 / 4 + 4096 / 8 + 8192 / 16 = 3072个内核
        next_m, sample_step = 1024, 2  # 初始M值和采样步长
        max_prefill_bs = (  # 计算最大预填充批量大小
            min(server_args.chunked_prefill_size, 32 * 1024)  # 取分块预填充大小和32768的较小值
            if server_args.chunked_prefill_size >= 1  # 如果分块预填充大小大于等于1
            else 16 * 1024  # 否则使用16384
        )
        while next_m < max_prefill_bs:  # 循环直到next_m超过最大预填充批量大小
            _BUILTIN_M_LIST += list(range(next_m, 2 * next_m, sample_step))  # 按步长采样M值
            next_m = next_m * 2  # M值翻倍
            sample_step = sample_step * 2  # 步长翻倍
        _BUILTIN_M_LIST.append(max_prefill_bs)  # 添加最大预填充批量大小
        _BUILTIN_M_LIST = sorted(list(set(_BUILTIN_M_LIST)))  # 去重并排序
    else:
        # When fast warmup isn't enabled, generate m_max and compile all the covered Ms.
        # 当快速预热未启用时，生成m_max并编译所有覆盖的M值
        m_max = 1024 * 16  # 默认最大M值为16384
        if server_args.chunked_prefill_size < 1:  # 如果分块预填充大小小于1
            m_max = 1024 * 64  # 使用65536
        elif server_args.chunked_prefill_size > 8192:  # 如果分块预填充大小大于8192
            m_max = server_args.chunked_prefill_size * 2  # 使用分块预填充大小的两倍
        m_max = min(1024 * 128, m_max)  # 限制最大不超过131072
        _BUILTIN_M_LIST += list(range(1, m_max + 1))  # 添加1到m_max的所有M值

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id  # 判断是否是节点上第一个rank

    # Check if is the first rank on node.
    # 检查是否是节点上的第一个rank。
    # Default each rank will try compile all Ms to
    # 默认每个rank会尝试编译所有M值
    # load all symbols at the launch stages.
    # 在启动阶段加载所有符号。
    # Avoid loading symbols at the serving stages.
    # 避免在服务阶段加载符号。
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE  # 只有第一个rank编译所有内核


class DeepGemmKernelType(IntEnum):  # DeepGEMM内核类型枚举
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()  # 分组FP8矩阵乘法（掩码模式）
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()  # 分组FP8矩阵乘法（连续模式）
    GROUPED_GEMM_NT_BF16_MASKED = auto()  # 分组BF16矩阵乘法（掩码模式）
    GROUPED_GEMM_NT_BF16_CONTIG = auto()  # 分组BF16矩阵乘法（连续模式）
    GEMM_NT_F8F8BF16 = auto()  # 普通FP8矩阵乘法
    GEMM_NT_BF16BF16F32 = auto()  # 普通BF16矩阵乘法（FP32输出）
    TF32_HC_PRENORM_GEMM = auto()  # TF32 Householder预归一化矩阵乘法


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()  # 初始化状态字典，记录已编译的内核组合


# TODO improve code
# TODO 改进代码
def _maybe_compile_deep_gemm_one_type_all(  # 可能编译某一种类型的所有DeepGEMM内核
    kernel_type: DeepGemmKernelType,  # 内核类型
    n: int,  # N维度大小
    k: int,  # K维度大小
    num_groups: int,  # 分组数量
) -> None:
    global _INITIALIZATION_DICT  # 声明全局初始化字典
    global _BUILTIN_M_LIST  # 声明全局M值列表

    query_key = (kernel_type, n, k, num_groups)  # 构造查询键
    if (  # 如果满足预编译条件
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE  # 启用了JIT预编译
        and _DO_COMPILE_ALL  # 需要编译所有
        and _INITIALIZATION_DICT.get(query_key) is None  # 尚未初始化该组合
    ):
        _INITIALIZATION_DICT[query_key] = True  # 标记为已初始化

        # TODO maybe improve logs
        # TODO 可能改进日志
        if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:  # 如果不在预编译阶段且是第一个rank
            logger.warning(  # 输出警告日志
                "Entering DeepGEMM JIT Pre-Compile session. "  # 正在进入DeepGEMM JIT预编译会话
                "It may take a long time (typically 10-20 mins) "  # 可能需要很长时间（通常10-20分钟）
                "if you have not run `sglang.compile_deep_gemm`. "  # 如果你还没有运行过预编译命令
                "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"  # 建议使用与启动服务器相同的参数运行预编译
                " for pre-compilation to reduce the overhead if you have not run it before. "  # 进行预编译以减少开销
                "For example: "  # 例如：
                "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
            )

        logger.info(  # 输出信息日志
            f"Try DeepGEMM JIT Compiling for "  # 尝试为以下配置进行DeepGEMM JIT编译
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."  # 内核类型、N、K、分组数及所有M值
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"  # 如果已预编译过则只需很短时间
        )

        _compile_deep_gemm_one_type_all(  # 执行编译
            kernel_type=kernel_type,  # 内核类型
            n=n,  # N维度
            k=k,  # K维度
            num_groups=num_groups,  # 分组数量
            m_list=_BUILTIN_M_LIST,  # M值列表
        )


# NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
# 注意(alcanderian): 当引入2批次重叠时，get_num_sms应该被修改
def _compile_deep_gemm_one_type_all(  # 编译某一种类型的所有DeepGEMM内核
    kernel_type: DeepGemmKernelType,  # 内核类型
    n: int,  # N维度大小
    k: int,  # K维度大小
    num_groups: int,  # 分组数量
    m_list: List[int],  # M值列表
) -> None:
    # Symmetric memory allocation performs a collective operation across all the GPUs.
    # 对称内存分配在所有GPU之间执行集合操作。
    # Temporary disable symmetric memory during compilation since it only runs on the first rank.
    # 编译期间临时禁用对称内存，因为编译只在第一个rank上运行。
    saved_context = disable_symmetric_memory_context()  # 保存并禁用对称内存上下文
    try:
        if kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:  # 如果是FP8连续分组GEMM
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()  # 获取连续布局的M对齐要求
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))  # 过滤掉不满足对齐要求的M值
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:  # 如果是BF16连续分组GEMM
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()  # 获取连续布局的M对齐要求
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))  # 过滤掉不满足对齐要求的M值

        # Here the precompilation is only run on the first rank, so gpu_id should be 0
        # 这里预编译只在第一个rank上运行，所以gpu_id应为0
        memory_budget = get_available_gpu_memory(device="cuda", gpu_id=0)  # 获取可用GPU显存

        # If the memory budget is less memory requirement, we need to reduce max_m to avoid out of memory, which might further cause hanging during warmup
        # 如果可用显存小于内存需求，需要减小max_m以避免内存不足，否则可能导致预热期间挂起
        max_m = max(m_list)  # 获取M值列表中的最大值
        required_memory = _BaseWarmupExecutor.get_memory_requirement(  # 计算所需内存
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        logger.info(  # 输出日志信息
            f"Required memory for warmup: {required_memory}GB, Available memory: {memory_budget}GB"  # 预热所需内存和可用内存
        )
        if memory_budget < required_memory:  # 如果可用内存不足
            # TODO: Maybe compute the max_m based on the memory budget
            # TODO: 也许可以根据内存预算计算max_m
            while (  # 循环减小max_m直到内存足够
                _BaseWarmupExecutor.get_memory_requirement(  # 计算当前max_m所需内存
                    kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                )
                > memory_budget  # 大于可用内存
                and max_m > 4096  # max_m仍大于4096
            ):
                max_m = max_m // 2  # 将max_m减半
            logger.warning(  # 输出警告日志
                f"Available memory {memory_budget}GB is less than required memory {required_memory}GB for warmup, reducing max_m to {max_m} to avoid out of memory"  # 可用内存不足，减小max_m以避免OOM
            )
            m_list = [m for m in m_list if m <= max_m]  # 过滤掉超过max_m的M值

        # Need some methods to estimate needed memory for warmup
        # 需要一些方法来估计预热所需的内存
        executor = _BaseWarmupExecutor.create(  # 创建预热执行器
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )

        has_compile_mode_api = hasattr(deep_gemm, "get_compile_mode") and hasattr(  # 检查deep_gemm是否支持编译模式API
            deep_gemm, "set_compile_mode"
        )
        if has_compile_mode_api:  # 如果支持编译模式API
            old_compile_mode = deep_gemm.get_compile_mode()  # 保存旧编译模式
            deep_gemm.set_compile_mode(1)  # 设置为编译模式

        # TODO can use multi thread
        # TODO 可以使用多线程
        for m in tqdm(m_list, desc="DeepGEMM warmup"):  # 遍历所有M值进行预热
            executor.execute(m=m)  # 执行预热
        if has_compile_mode_api:  # 如果支持编译模式API
            deep_gemm.set_compile_mode(old_compile_mode)  # 恢复旧编译模式

        # clean up input buffers
        # 清理输入缓冲区
        torch.cuda.current_stream().synchronize()  # 同步CUDA流
        del executor  # 删除执行器
        torch.cuda.empty_cache()  # 清空CUDA缓存
    finally:
        # Restore symmetric memory context
        # 恢复对称内存上下文
        restore_symmetric_memory_context(saved_context)  # 恢复之前保存的对称内存上下文


class _BaseWarmupExecutor:  # 预热执行器基类
    @staticmethod
    def create(kernel_type: DeepGemmKernelType, **kwargs):  # 根据内核类型创建对应的执行器
        return {
            DeepGemmKernelType.GEMM_NT_F8F8BF16: _NormalWarmupExecutor,  # FP8普通GEMM执行器
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: _GroupedContWarmupExecutor,  # FP8连续分组GEMM执行器
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: _GroupedMaskedWarmupExecutor,  # FP8掩码分组GEMM执行器
            DeepGemmKernelType.GEMM_NT_BF16BF16F32: _BF16F32WarmupExecutor,  # BF16普通GEMM执行器
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG: _BF16GroupedContWarmupExecutor,  # BF16连续分组GEMM执行器
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED: _BF16GroupedMaskedWarmupExecutor,  # BF16掩码分组GEMM执行器
            DeepGemmKernelType.TF32_HC_PRENORM_GEMM: _TF32HcPrenormWarmupExecutor,  # TF32 Householder预归一化GEMM执行器
        }[kernel_type](**kwargs)  # 根据内核类型实例化对应的执行器

    @staticmethod
    def get_memory_requirement(  # 获取预热所需的内存（GB）
        kernel_type: DeepGemmKernelType, max_m: int, n: int, k: int, num_groups: int
    ) -> int:
        # Return the required memory space in GB for warmup executor
        # 返回预热执行器所需的内存空间（GB）
        _GB = 1 << 30  # 1GB的字节数
        if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:  # FP8普通GEMM
            return (max_m * k + n * k + max_m * n * 2) / _GB  # lhs + rhs + out
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:  # FP8连续分组GEMM
            return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB  # lhs + rhs + m_indices + out
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:  # BF16连续分组GEMM
            return (
                max_m * k * 2 + num_groups * n * k * 2 + max_m * 4 + max_m * n * 2
            ) / _GB  # lhs_bf16 + rhs_bf16 + m_indices + out
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:  # FP8掩码分组GEMM
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB  # lhs + rhs + masked_m + out
        elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16F32:  # BF16普通GEMM（FP32输出）
            # bf16 lhs + bf16 rhs + fp32 out
            # bf16左侧 + bf16右侧 + fp32输出
            return (max_m * k * 2 + n * k * 2 + max_m * n * 4) / _GB  # lhs_bf16 + rhs_bf16 + out_fp32
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED:  # BF16掩码分组GEMM
            return (
                num_groups * max_m * k * 2
                + num_groups * n * k * 2
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB  # lhs + rhs + masked_m + out
        elif kernel_type == DeepGemmKernelType.TF32_HC_PRENORM_GEMM:  # TF32 Householder预归一化GEMM
            # The generic hook's fourth dimension is num_splits for MHC.
            # 通用钩子的第四维度是MHC的num_splits。
            # A value of 0 represents DeepGEMM's unsplit num_splits=None path.
            # 值为0代表DeepGEMM的不分割num_splits=None路径。
            num_splits = num_groups if num_groups > 0 else 1  # 将0转换为1用于内存计算
            return (max_m * k * 2 + n * k * 4 + num_splits * max_m * (n + 1) * 4) / _GB  # x_bf16 + fn_fp32 + out+sqrsum_fp32
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")  # 无效的内核类型

    def execute(self, m):  # 执行预热（需子类实现）
        raise NotImplementedError  # 未实现错误


def _empty_token_fp8(size):  # 创建FP8空token张量及其缩放因子
    *dims, k = size  # 解包尺寸，最后一维为K
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),  # FP8数据张量
        torch.empty(  # FP8缩放因子张量
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32  # 缩放因子尺寸为K/BLOCK_SIZE向上取整
        ),
    )


def _empty_block_fp8(size):  # 创建FP8空块张量及其缩放因子
    *dims, n, k = size  # 解包尺寸，最后两维为N和K
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),  # FP8数据张量
        torch.empty(  # FP8缩放因子张量
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),  # 缩放因子尺寸为N和K分别除以BLOCK_SIZE向上取整
            device="cuda",
            dtype=torch.float32,
        ),
    )


_BLOCK_SIZE = 128  # FP8缩放因子的块大小


class _NormalWarmupExecutor(_BaseWarmupExecutor):  # 普通FP8 GEMM预热执行器
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):  # 初始化
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))  # 左侧FP8数据及缩放因子
        self.rhs_q, self.rhs_s = _empty_block_fp8((n, k))  # 右侧FP8数据及缩放因子
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)  # 输出BF16张量

    def execute(self, m):  # 执行指定M值的预热
        deep_gemm.fp8_gemm_nt(  # 调用FP8 GEMM内核
            (self.lhs_q[:m], self.lhs_s[:m]),  # 左侧切片
            (self.rhs_q, self.rhs_s),  # 右侧完整
            self.out[:m],  # 输出切片
        )


class _GroupedContWarmupExecutor(_BaseWarmupExecutor):  # FP8连续分组GEMM预热执行器
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):  # 初始化
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))  # 左侧FP8数据及缩放因子
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))  # 右侧FP8数据及缩放因子（分组）
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)  # M索引张量
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)  # 输出BF16张量

    def execute(self, m):  # 执行指定M值的预热
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(  # 调用FP8连续分组GEMM内核
            (self.lhs_q[:m], self.lhs_s[:m]),  # 左侧切片
            (self.rhs_q, self.rhs_s),  # 右侧完整
            self.out[:m],  # 输出切片
            self.m_indices[:m],  # M索引切片
        )


class _BF16GroupedContWarmupExecutor(_BaseWarmupExecutor):  # BF16连续分组GEMM预热执行器
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):  # 初始化
        self.a = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)  # 左侧BF16张量
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)  # 右侧BF16张量（分组）
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)  # M索引张量
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)  # 输出BF16张量

    def execute(self, m):  # 执行指定M值的预热
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(  # 调用BF16连续分组GEMM内核
            self.a[:m],  # 左侧切片
            self.b,  # 右侧完整
            self.out[:m],  # 输出切片
            self.m_indices[:m],  # M索引切片
        )


class _GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):  # FP8掩码分组GEMM预热执行器
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):  # 初始化
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))  # 左侧FP8数据及缩放因子（分组）
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))  # 右侧FP8数据及缩放因子（分组）
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)  # 掩码M张量
        self.out = torch.empty(  # 输出BF16张量（分组）
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):  # 执行指定M值的预热
        deep_gemm.fp8_m_grouped_gemm_nt_masked(  # 调用FP8掩码分组GEMM内核
            (self.lhs_q, self.lhs_s),  # 左侧完整
            (self.rhs_q, self.rhs_s),  # 右侧完整
            self.out,  # 输出完整
            masked_m=self.masked_m,  # 掩码M
            # DeepGEMM uses `expect_m` instead of input shape for `get_best_config`
            # DeepGEMM使用`expected_m`而不是输入形状来获取最佳配置
            expected_m=m,  # 期望的M值
        )


class _BF16F32WarmupExecutor(_BaseWarmupExecutor):  # BF16 GEMM（FP32输出）预热执行器
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):  # 初始化
        self.lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)  # 左侧BF16张量
        self.rhs = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)  # 右侧BF16张量
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.float32)  # 输出FP32张量

    def execute(self, m):  # 执行指定M值的预热
        deep_gemm.bf16_gemm_nt(self.lhs[:m], self.rhs, self.out[:m])  # 调用BF16 GEMM内核


class _BF16GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):  # BF16掩码分组GEMM预热执行器
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):  # 初始化
        self.a = torch.empty(  # 左侧BF16张量（分组）
            (num_groups, max_m, k), device="cuda", dtype=torch.bfloat16
        )
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)  # 右侧BF16张量（分组）
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)  # 掩码M张量
        self.out = torch.empty(  # 输出BF16张量（分组）
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):  # 执行指定M值的预热
        deep_gemm.m_grouped_bf16_gemm_nt_masked(  # 调用BF16掩码分组GEMM内核
            self.a,  # 左侧完整
            self.b,  # 右侧完整
            self.out,  # 输出完整
            masked_m=self.masked_m,  # 掩码M
            # DeepGEMM uses `expect_m` instead of input shape for `get_best_config`
            # DeepGEMM使用`expected_m`而不是输入形状来获取最佳配置
            expected_m=m,  # 期望的M值
        )


class _TF32HcPrenormWarmupExecutor(_BaseWarmupExecutor):  # TF32 Householder预归一化GEMM预热执行器
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):  # 初始化
        self.x = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)  # 输入BF16张量
        self.fn = torch.empty((n, k), device="cuda", dtype=torch.float32)  # 权重FP32张量
        self.n = n  # N维度大小
        # The generic warmup executor's num_groups argument is num_splits here.
        # 通用预热执行器的num_groups参数在此处为num_splits。
        # A value of 0 represents DeepGEMM's unsplit num_splits=None path.
        # 值为0代表DeepGEMM的不分割num_splits=None路径。
        self.num_splits = num_groups if num_groups > 0 else None  # 将0转换为None

    def execute(self, m):  # 执行指定M值的预热
        if self.num_splits is None:  # 如果不分割
            out = torch.empty((m, self.n), device="cuda", dtype=torch.float32)  # 输出FP32张量
            sqrsum = torch.empty((m,), device="cuda", dtype=torch.float32)  # 平方和FP32张量
        else:  # 如果分割
            # Slicing the middle dimension of a preallocated
            # 对预分配张量的中间维度进行切片
            # (num_splits, max_m, n) output would create a strided view.
            # (num_splits, max_m, n)输出会创建一个跨步视图。
            out = torch.empty(  # 输出FP32张量（分割）
                (self.num_splits, m, self.n), device="cuda", dtype=torch.float32
            )
            sqrsum = torch.empty(  # 平方和FP32张量（分割）
                (self.num_splits, m), device="cuda", dtype=torch.float32
            )
        deep_gemm.tf32_hc_prenorm_gemm(  # 调用TF32 Householder预归一化GEMM内核
            self.x[:m],  # 输入切片
            self.fn,  # 权重完整
            out,  # 输出
            sqrsum,  # 平方和
            num_splits=self.num_splits,  # 分割数
        )


def deep_gemm_execution_hook(  # DeepGEMM执行钩子，用于JIT编译触发
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if _is_musa:  # 如果是MUSA平台
        return nullcontext()  # 返回空上下文

    return _deep_gemm_execution_hook(m, n, k, num_groups, kernel_type)  # 返回实际的执行钩子


@contextmanager
def _deep_gemm_execution_hook(  # DeepGEMM执行钩子上下文管理器
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if m > 0:  # 如果M大于0
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)  # 触发可能的JIT编译
    yield  # 执行实际计算


def pp_parallel_deep_gemm_warmup(model_runner) -> None:  # 流水线并行DeepGEMM预热
    """Run per-PP-rank dummy DECODE+EXTEND forwards so each rank's
    DeepGEMM JIT compiles in parallel instead of serially via the warmup
    /generate flowing through the pipeline. Opt-in via
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.
    """
    # 运行每个PP rank的虚拟DECODE+EXTEND前向传播，使每个rank的DeepGEMM JIT编译
    # 并行进行，而不是通过预热/生成流经流水线串行编译。
    # 通过SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP选择启用。
    # n_splits ~= n_sms / ceil(bs/block_m) with block_m=64; sweep 5 bs to
    # cover the brackets real /generate hits (smallest decode shape,
    # mid-low, two mid, and n_splits=1 for ~5K+ token prefill). Ceil-align
    # to attn_cp_size for DSA prefill CP's seq_len % cp_size == 0 assert.
    # n_splits ~= n_sms / ceil(bs/block_m)，block_m=64；扫描5个批量大小以
    # 覆盖实际/生成命中的范围（最小解码形状、中低、两个中间值，
    # 以及~5K+ token预填充的n_splits=1）。向上对齐到attn_cp_size，
    # 以满足DSA预填充CP的seq_len % cp_size == 0断言。
    n_sms = torch.cuda.get_device_properties(model_runner.device).multi_processor_count  # 获取GPU流多处理器数量
    block_m = 64  # M方向的块大小
    cp = max(model_runner.attn_cp_size, 1)  # 注意力上下文并行大小
    batch_sizes = sorted(  # 排序后的批量大小集合
        {
            ceil_align(bs, cp)  # 将批量大小对齐到cp
            for bs in (
                1,  # 最小解码批量
                2 * block_m,  # 2个块
                max(n_sms // 8, 2) * block_m,  # 中低批量
                max(n_sms // 4, 4) * block_m,  # 中间批量
                n_sms * block_m,  # 最大批量
            )
        }
    )

    # In PD, prefill-only nodes never decode (indexer would OOM at large
    # bs) and decode-only nodes never extend.
    # 在PD模式下，仅预填充节点不会解码（索引器在大批量时会OOM），
    # 仅解码节点不会扩展。
    disagg_mode = model_runner.server_args.disaggregation_mode  # 获取分离模式
    run_decode = model_runner.is_generation and disagg_mode != "prefill"  # 是否运行解码
    run_extend = disagg_mode != "decode"  # 是否运行扩展

    logger.info(  # 输出日志信息
        "PP-parallel DeepGEMM warmup start "  # PP并行DeepGEMM预热开始
        "(pp_rank=%d, tp_rank=%d, batch_sizes=%s, disagg=%s).",  # PP rank、TP rank、批量大小、分离模式
        model_runner.pp_rank,
        model_runner.tp_rank,
        batch_sizes,
        disagg_mode,
    )

    t0 = time.perf_counter()  # 记录开始时间
    with torch.inference_mode():  # 推理模式
        for bs in batch_sizes:  # 遍历所有批量大小
            if run_decode:  # 如果运行解码
                model_runner._dummy_run(  # 运行虚拟解码
                    batch_size=bs, forward_mode_override=ForwardMode.DECODE
                )
            if run_extend:  # 如果运行扩展
                model_runner._dummy_run(  # 运行虚拟扩展
                    batch_size=bs, forward_mode_override=ForwardMode.EXTEND
                )

    logger.info(  # 输出日志信息
        "PP-parallel DeepGEMM warmup done in %.2fs (pp_rank=%d).",  # PP并行DeepGEMM预热完成，耗时和PP rank
        time.perf_counter() - t0,  # 计算耗时
        model_runner.pp_rank,
    )
