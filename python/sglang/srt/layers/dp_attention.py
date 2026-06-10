# 数据并行注意力机制（DP Attention）实现，包含 DP 填充模式、缓冲区管理、
# 令牌收集/分散、AllReduce/AllGather 融合通信操作。

from __future__ import annotations  # 启用延迟类型注解求值

import functools  # 导入函数工具
import logging  # 导入日志模块
from contextlib import contextmanager  # 导入上下文管理器
from enum import IntEnum, auto  # 导入枚举类型
from typing import TYPE_CHECKING, List, Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch
import triton  # 导入 Triton
import triton.language as tl  # 导入 Triton 语言

from sglang.srt.distributed import (  # 导入分布式通信相关函数
    GroupCoordinator,
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
    get_attn_cp_group,
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
)
from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group  # 导入 MoE DP 组获取函数
from sglang.srt.distributed import (  # 导入张量并行相关函数
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入对称内存工具
    use_symmetric_memory,
)
from sglang.srt.utils import get_bool_env_var, is_hip  # 导入环境变量和平台检测工具

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.configs.model_config import ModelConfig  # 导入模型配置
    from sglang.srt.server_args import ServerArgs  # 导入服务器参数

logger = logging.getLogger(__name__)  # 获取日志记录器

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息

_ATTN_DP_RANK: Optional[int] = None  # 注意力 DP 全局排名
_ATTN_DP_SIZE: Optional[int] = None  # 注意力 DP 大小
_LOCAL_ATTN_DP_SIZE: Optional[int] = None  # 本地注意力 DP 大小
_LOCAL_ATTN_DP_RANK: Optional[int] = None  # 本地注意力 DP 排名
_ENABLE_DP_ATTENTION_FLAG: bool = False  # 是否启用 DP 注意力标志

_is_hip = is_hip()  # 检测是否为 AMD ROCm 平台
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")  # 是否使用 ROCm 7.0.0 Alpha 临时解决方案


class DpPaddingMode(IntEnum):  # DP 填充模式枚举

    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    # 将令牌填充到最大长度，然后使用 all_gather_into_tensor 收集令牌
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    # 将令牌填充到总长度，然后使用 all_reduce 收集令牌
    SUM_LEN = auto()

    def is_max_len(self):  # 判断是否为最大长度模式
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):  # 判断是否为总长度模式
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(  # 根据 extend 模式和全局令牌数选择 DP 填充模式
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        dp_size = get_attention_dp_size()  # 获取注意力 DP 大小

        # When is_extend_in_batch and dp_size > 1, use SUM_LEN to avoid padding
        # overhead from uneven token distribution.
        # For dp_size=1, max_len equals sum_len, so prefer MAX_LEN mode
        # to enable symmetric memory optimization (needed for DSA CP, etc.).
        # 当 is_extend_in_batch 且 dp_size > 1 时，使用 SUM_LEN 避免不均匀令牌分布的填充开销。
        # 对于 dp_size=1，max_len 等于 sum_len，因此优先使用 MAX_LEN 模式以启用对称内存优化（DSA CP 等需要）。
        if is_extend_in_batch and dp_size > 1:  # extend 模式且 DP 大于 1
            return DpPaddingMode.SUM_LEN  # 返回 SUM_LEN 模式

        # we choose the mode that minimizes the communication cost
        # prefer MAX_LEN when communication cost is equal to enable symmetric memory
        # 选择通信成本最小的模式；通信成本相等时优先 MAX_LEN 以启用对称内存
        max_len = max(global_num_tokens)  # 计算最大令牌数
        sum_len = sum(global_num_tokens)  # 计算总令牌数
        if sum_len * 2 >= max_len * dp_size:  # 如果 all_reduce 通信量不小于 all_gather
            return cls.MAX_LEN  # 返回 MAX_LEN 模式
        else:
            return cls.SUM_LEN  # 返回 SUM_LEN 模式

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:  # 获取 CUDA Graph 中的默认填充模式
        # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
        # it can be safely removed later, once RCCL fixed
        # TODO(kkhuang-amd): noqa，ROCm 7.0.0 Alpha 的临时解决方案，RCCL 修复后可安全移除
        if _USE_ROCM700A_WA:  # 如果使用 ROCm 7.0.0 Alpha 临时解决方案
            return cls.SUM_LEN  # 返回 SUM_LEN 模式
        else:
            return cls.MAX_LEN  # 返回 MAX_LEN 模式


class _DpGatheredBufferWrapper:  # DP 收集缓冲区包装器，管理全局和本地缓冲区

    _hidden_size: int  # 隐藏层大小
    _dtype: torch.dtype  # 数据类型
    _device: torch.device  # 设备
    _global_dp_buffer_len: int  # 全局 DP 缓冲区长度
    _local_dp_buffer_len: int  # 本地 DP 缓冲区长度
    _dp_max_padding: bool  # 是否使用最大长度填充
    _global_num_tokens: Optional[List[int]]  # 全局令牌数列表
    _is_extend_in_batch: bool  # 是否为批次内 extend 模式

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):  # 设置元数据
        cls._hidden_size = hidden_size  # 保存隐藏层大小
        cls._dtype = dtype  # 保存数据类型
        cls._device = device  # 保存设备

    @classmethod
    def set_dp_buffer_len(  # 设置 DP 缓冲区长度
        cls,
        global_dp_buffer_len: int,  # 全局 DP 缓冲区长度
        local_dp_buffer_len: int,  # 本地 DP 缓冲区长度
        dp_max_padding: bool,  # 是否使用最大长度填充
        global_num_tokens: Optional[List[int]] = None,  # 全局令牌数列表
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len  # 保存全局缓冲区长度
        cls._local_dp_buffer_len = local_dp_buffer_len  # 保存本地缓冲区长度
        cls._dp_max_padding = dp_max_padding  # 保存填充模式标志
        cls._global_num_tokens = global_num_tokens  # 保存全局令牌数

    @classmethod
    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:  # 获取全局 DP 缓冲区
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):  # 条件性使用对称内存
            buffer = torch.empty(  # 分配空缓冲区
                (cls._global_dp_buffer_len, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer  # 返回全局缓冲区

    @classmethod
    def get_local_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:  # 获取本地 DP 缓冲区
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):  # 条件性使用对称内存
            buffer = torch.empty(  # 分配空缓冲区
                (cls._local_dp_buffer_len, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer  # 返回本地缓冲区

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:  # 获取全局 DP 缓冲区长度
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:  # 获取本地 DP 缓冲区长度
        return cls._local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:  # 获取全局令牌数列表
        return cls._global_num_tokens

    @classmethod
    def get_dp_hidden_size(cls) -> int:  # 获取隐藏层大小
        return cls._hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:  # 获取数据类型
        return cls._dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:  # 获取设备
        return cls._device

    @classmethod
    def set_is_extend_in_batch(cls, is_extend_in_batch: bool):  # 设置是否为批次内 extend 模式
        cls._is_extend_in_batch = is_extend_in_batch

    @classmethod
    def get_is_extend_in_batch(cls) -> bool:  # 获取是否为批次内 extend 模式
        return cls._is_extend_in_batch

    @classmethod
    def is_dp_max_padding(cls) -> bool:  # 判断是否使用最大长度填充
        return cls._dp_max_padding


def set_dp_buffer_len(  # 设置 DP 缓冲区长度（模块级接口）
    global_dp_buffer_len: int,  # 全局 DP 缓冲区长度
    local_dp_buffer_len: int,  # 本地 DP 缓冲区长度
    dp_max_padding: bool,  # 是否使用最大长度填充
    global_num_tokens: Optional[List[int]] = None,  # 全局令牌数列表
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len, local_dp_buffer_len, dp_max_padding, global_num_tokens
    )


def get_global_dp_buffer(group: GroupCoordinator) -> torch.Tensor:  # 获取全局 DP 缓冲区
    return _DpGatheredBufferWrapper.get_global_dp_buffer(group=group)


def get_local_dp_buffer(group: GroupCoordinator) -> torch.Tensor:  # 获取本地 DP 缓冲区
    return _DpGatheredBufferWrapper.get_local_dp_buffer(group=group)


def get_global_dp_buffer_len() -> int:  # 获取全局 DP 缓冲区长度
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:  # 获取本地 DP 缓冲区长度
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def get_dp_global_num_tokens() -> List[int]:  # 获取全局令牌数列表
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:  # 获取隐藏层大小
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:  # 获取数据类型
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:  # 获取设备
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):  # 设置是否为批次内 extend 模式
    _DpGatheredBufferWrapper.set_is_extend_in_batch(is_extend_in_batch)


def get_is_extend_in_batch() -> bool:  # 获取是否为批次内 extend 模式
    return _DpGatheredBufferWrapper.get_is_extend_in_batch()


def is_dp_max_padding() -> bool:  # 判断是否使用最大长度填充
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(  # 计算 DP 注意力的全局信息
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    attn_dp_size = dp_size if enable_dp_attention else 1  # DP 大小：启用时为 dp_size，否则为 1
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size  # 注意力 TP 大小
    attn_tp_rank = tp_rank % attn_tp_size  # 注意力 TP 排名

    if not enable_dp_attention:  # 如果未启用 DP 注意力
        attn_dp_rank = 0  # DP 排名设为 0
    else:
        # Rank layout is (dp, cp, tp) where tp is the fastest-changing dim:
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        # 排名布局为 (dp, cp, tp)，其中 tp 是最快变化维度
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)  # 计算注意力 DP 排名

    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size  # 返回注意力 TP 排名、TP 大小、DP 排名、DP 大小


def compute_dp_attention_local_info(  # 计算 DP 注意力的本地信息
    enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
):
    if not enable_dp_attention:  # 如果未启用 DP 注意力
        return tp_rank, tp_size, 0  # 直接返回 TP 信息

    local_tp_size = moe_dense_tp_size if moe_dense_tp_size else tp_size  # 本地 TP 大小
    local_tp_rank = tp_rank % local_tp_size  # 本地 TP 排名
    local_dp_size = max(1, dp_size // (tp_size // local_tp_size))  # 本地 DP 大小

    local_attn_tp_size = local_tp_size // local_dp_size  # 本地注意力 TP 大小
    local_attn_dp_rank = local_tp_rank // local_attn_tp_size  # 本地注意力 DP 排名
    local_attn_tp_rank = local_tp_rank % local_attn_tp_size  # 本地注意力 TP 排名

    return local_attn_tp_rank, local_attn_tp_size, local_attn_dp_rank  # 返回本地注意力 TP 排名、TP 大小、DP 排名


def initialize_dp_attention(  # 初始化 DP 注意力
    server_args: ServerArgs,  # 服务器参数
    model_config: ModelConfig,  # 模型配置
):
    global _ATTN_DP_RANK, _ATTN_DP_SIZE  # 声明全局变量
    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG
    enable_dp_attention = server_args.enable_dp_attention  # 获取是否启用 DP 注意力
    dp_size = server_args.dp_size  # 获取 DP 大小
    moe_dense_tp_size = server_args.moe_dense_tp_size  # 获取 MoE dense TP 大小
    attn_cp_size = server_args.attn_cp_size  # 获取注意力 CP 大小

    _ENABLE_DP_ATTENTION_FLAG = enable_dp_attention  # 设置全局标志

    tp_rank = get_tensor_model_parallel_rank()  # 获取 TP 排名
    tp_size = get_tensor_model_parallel_world_size()  # 获取 TP 大小

    _, _, _ATTN_DP_RANK, _ = compute_dp_attention_world_info(  # 计算全局 DP 注意力信息
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    _, _, _LOCAL_ATTN_DP_RANK = compute_dp_attention_local_info(  # 计算本地 DP 注意力信息
        enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
    )

    if enable_dp_attention:  # 如果启用 DP 注意力
        _ATTN_DP_SIZE = dp_size  # 设置全局 DP 大小
        if moe_dense_tp_size is None:  # 如果没有 MoE dense TP 大小
            _LOCAL_ATTN_DP_SIZE = _ATTN_DP_SIZE  # 本地等于全局
        else:
            _LOCAL_ATTN_DP_SIZE = max(1, dp_size // (tp_size // moe_dense_tp_size))  # 计算本地 DP 大小
    else:
        _ATTN_DP_SIZE = 1  # 全局 DP 大小设为 1
        _LOCAL_ATTN_DP_SIZE = 1  # 本地 DP 大小设为 1

    _DpGatheredBufferWrapper.set_metadata(  # 设置缓冲区元数据
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )


def is_dp_attention_enabled() -> bool:  # 检查 DP 注意力是否启用
    return _ENABLE_DP_ATTENTION_FLAG


def is_allocation_symmetric() -> bool:  # 检查分配是否对称
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_tp_group() -> GroupCoordinator:  # 获取注意力 TP 组
    return get_attn_tp_group()


def get_attention_tp_rank() -> int:  # 获取注意力 TP 排名
    return get_attn_tensor_model_parallel_rank()


def get_attention_tp_size() -> int:  # 获取注意力 TP 大小
    return get_attn_tensor_model_parallel_world_size()


def get_attention_cp_group() -> GroupCoordinator:  # 获取注意力 CP 组
    return get_attn_cp_group()


def get_attention_cp_rank() -> int:  # 获取注意力 CP 排名
    return get_attn_context_model_parallel_rank()


def get_attention_cp_size() -> int:  # 获取注意力 CP 大小
    return get_attn_context_model_parallel_world_size()


def get_attention_dp_rank() -> int:  # 获取注意力 DP 排名
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"  # 断言已初始化
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:  # 获取注意力 DP 大小
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"  # 断言已初始化
    return _ATTN_DP_SIZE


def get_local_attention_dp_rank() -> int:  # 获取本地注意力 DP 排名
    assert _LOCAL_ATTN_DP_RANK is not None, "dp attention not initialized!"  # 断言已初始化
    return _LOCAL_ATTN_DP_RANK


def get_local_attention_dp_size() -> int:  # 获取本地注意力 DP 大小
    assert _LOCAL_ATTN_DP_SIZE is not None, "dp attention not initialized!"  # 断言已初始化
    return _LOCAL_ATTN_DP_SIZE


@contextmanager
def disable_dp_size():  # 临时禁用 DP 大小的上下文管理器
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """  # 临时修补 TP 组直到函数结束。此方法用于推测解码的 draft worker，以不同于目标模型 worker 的 TP 度运行 draft 模型。参数：tp_group（GroupCoordinator）：TP 组协调器。
    global _ATTN_DP_SIZE  # 声明全局变量
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"  # 断言已初始化

    old_dp_size = _ATTN_DP_SIZE  # 保存旧的 DP 大小
    _ATTN_DP_SIZE = 1  # 临时设为 1
    try:
        yield  # 执行上下文中的代码
    finally:
        _ATTN_DP_SIZE = old_dp_size  # 恢复旧的 DP 大小


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:  # 获取 DP 本地信息（起始位置和令牌数）
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.
    # get_dp_local_info 只在全局 DP 收集和分散中调用。这里使用全局 DP 排名。
    dp_rank = get_attention_dp_rank()  # 获取全局 DP 排名

    if forward_batch.dp_local_start_pos is None:  # 如果尚未缓存
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)  # 计算累积令牌数
        if dp_rank == 0:  # 如果是第一个 DP 排名
            local_start_pos = torch.zeros_like(cumtokens[0])  # 起始位置为 0
        else:
            local_start_pos = cumtokens[dp_rank - 1]  # 起始位置为前一个排名的累积令牌数
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]  # 本地令牌数

        forward_batch.dp_local_start_pos = local_start_pos  # 缓存起始位置
        forward_batch.dp_local_num_tokens = local_num_tokens  # 缓存令牌数

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens  # 返回缓存值


def get_dp_local_slice_cpu(  # 获取 DP 本地切片的 CPU 信息
    forward_batch: ForwardBatch,
    can_run_graph: bool,  # 是否可运行 CUDA Graph
    cuda_graph_batch: Optional[int],  # CUDA Graph 批次大小
) -> Tuple[int, int]:  # 返回 (起始位置, 长度)
    # CPU (start, length) slice for DP-local data in a rank-padded buffer.
    # Returns Python ints (no D2H sync) and handles the cuda-graph-padded layout.
    # DP 本地数据在排名填充缓冲区中的 CPU（起始，长度）切片。返回 Python 整数（无 D2H 同步），处理 CUDA Graph 填充布局。
    global_num_tokens = forward_batch.global_num_tokens_cpu  # 获取 CPU 端全局令牌数
    dp_rank = get_attention_dp_rank()  # 获取 DP 排名
    local_num_tokens = global_num_tokens[dp_rank]  # 获取本地令牌数
    if can_run_graph:  # 如果可运行 CUDA Graph
        local_start_pos = dp_rank * cuda_graph_batch  # 起始位置为排名乘以批次大小
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])  # 起始位置为前排名令牌数之和
    return local_start_pos, local_num_tokens  # 返回起始位置和令牌数


@triton.jit  # Triton JIT 编译的内存拷贝内核
def memcpy_triton_kernel(  # 内存拷贝 Triton 内核
    dst_ptr,  # 目标指针
    src_ptr,  # 源指针
    offset_ptr,  # 偏移量指针
    sz_ptr,  # 大小指针
    offset_src: tl.constexpr,  # 是否对源应用偏移
    chunk_size,  # multiplied for offset and sz # 用于偏移和大小的乘数
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(axis=0).to(tl.int64)  # 获取程序 ID
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size  # 计算偏移量
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size  # 计算大小

    start_index = pid * BLOCK_SIZE  # 计算起始索引
    offs = tl.arange(0, BLOCK_SIZE)  # 生成偏移
    mask = start_index + offs < sz  # 生成掩码

    if offset_src:  # 如果对源应用偏移（即从偏移位置读取）
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)  # 从偏移位置加载
        tl.store(dst_ptr + start_index + offs, data, mask=mask)  # 存储到目标起始位置
    else:  # 如果对目标应用偏移（即写入到偏移位置）
        data = tl.load(src_ptr + start_index + offs, mask=mask)  # 从源起始位置加载
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)  # 存储到偏移位置


def prod(x):  # 计算序列的乘积
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):  # 使用 Triton 进行内存拷贝
    max_size = min(src.numel(), dst.numel())  # 计算最大拷贝大小
    assert dim == 0, "dim != 0 unsupported"  # 仅支持 dim=0
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"  # 断言形状一致
    chunk_size = prod(src.shape[1:])  # 计算每个块的元素数
    BLOCK_SIZE = 8192  # 块大小
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)  # 设置网格大小

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)  # 启动内核


def _dp_gather_via_all_reduce(  # 通过 AllReduce 方式进行 DP 收集
    global_tokens: torch.Tensor,  # 全局令牌张量
    local_tokens: torch.Tensor,  # 本地令牌张量
    forward_batch: ForwardBatch,  # 前向批次
    is_partial: bool,  # 是否为部分结果
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)  # 获取本地信息

    global_tokens.fill_(0)  # 全局缓冲区清零
    assert local_tokens.is_contiguous()  # 断言本地令牌连续
    assert global_tokens.is_contiguous()  # 断言全局令牌连续

    if local_tokens.shape[0] > 0 and (is_partial or get_attention_tp_rank() == 0):  # 如果有令牌且是部分结果或 TP 排名为 0
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"  # 断言无别名

        memcpy_triton(  # 将本地令牌拷贝到全局缓冲区的对应位置
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    # 输入 ID 是 int32。对于本地情况应使用 inplace_all_reduce，因为有自定义 all reduce。
    NUM_GPUS_PER_NODE = 8  # 每节点 GPU 数
    if (
        not local_tokens.dtype.is_floating_point  # 如果是非浮点类型
        and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE  # 且 TP 大小不超过每节点 GPU 数
    ):
        from sglang.srt.distributed.parallel_state import inplace_all_reduce  # 导入原地 AllReduce

        inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)  # 执行原地 AllReduce

    else:
        global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)  # 执行常规 AllReduce


def _dp_gather_via_all_gather(  # 通过 AllGather 方式进行 DP 收集
    global_tokens: torch.Tensor,  # 全局令牌张量
    local_tokens: torch.Tensor,  # 本地令牌张量
    forward_batch: ForwardBatch,  # 前向批次
    is_partial: bool,  # 是否为部分结果
):
    if get_attention_tp_size() == 1:  # 如果注意力 TP 大小为 1
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)  # 直接 AllGather
        return

    if not is_partial:  # 如果不是部分结果（即完整结果需要复制）
        if get_attention_tp_rank() != 0:  # 非 TP 排名 0 的结果清零
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(get_attention_tp_size())[  # 按注意力 TP 大小切分
        get_attention_tp_rank()
    ]
    get_attention_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)  # ReduceScatter
    get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)  # AllGather


def _dp_gather(  # DP 收集统一入口
    global_tokens: torch.Tensor,  # 全局令牌张量
    local_tokens: torch.Tensor,  # 本地令牌张量
    forward_batch: ForwardBatch,  # 前向批次
    is_partial: bool,  # 是否为部分结果
):
    if forward_batch.dp_padding_mode.is_max_len():  # 如果使用最大长度填充模式
        _dp_gather_via_all_gather(  # 使用 AllGather 方式收集
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:  # 使用总长度填充模式
        _dp_gather_via_all_reduce(  # 使用 AllReduce 方式收集
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(  # DP 收集部分结果
    global_tokens: torch.Tensor,  # 全局令牌张量
    local_tokens: torch.Tensor,  # 本地令牌张量
    forward_batch: ForwardBatch,  # 前向批次
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(  # DP 收集复制结果
    global_tokens: torch.Tensor,  # 全局令牌张量
    local_tokens: torch.Tensor,  # 本地令牌张量
    forward_batch: ForwardBatch,  # 前向批次
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_scatter(  # DP 分散操作
    local_tokens: torch.Tensor,  # output # 输出本地令牌张量
    global_tokens: torch.Tensor,  # input # 输入全局令牌张量
    forward_batch: ForwardBatch,  # 前向批次
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    # local_num_tokens 不一定等于 local_tokens.shape[0]，因为 local_tokens 可能因 CUDA Graph 而被填充
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)  # 获取本地信息

    local_tokens.fill_(0)  # 本地缓冲区清零
    assert local_tokens.is_contiguous()  # 断言本地令牌连续
    assert global_tokens.is_contiguous()  # 断言全局令牌连续
    if local_tokens.shape[0] > 0:  # 如果有令牌
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between local_tokens and global_tokens not allowed"  # 断言无别名

        memcpy_triton(  # 从全局缓冲区拷贝到本地
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):  # DP ReduceScatter 张量
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():  # 如果 TP 大小等于 DP 大小
        get_tp_group().reduce_scatter_tensor(output, input)  # 直接 ReduceScatter
    else:  # 否则需要先 ReduceScatter 再 AllGather
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]  # 按排名切分
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)  # ReduceScatter
        get_attention_tp_group().all_gather_into_tensor(output, scattered_local_tokens)  # AllGather


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):  # 注意力 TP ReduceScatter
    return get_attention_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):  # 注意力 CP ReduceScatter
    return get_attention_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):  # 注意力 TP AllReduce
    return get_attention_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):  # 注意力 TP AllGather
    return get_attention_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):  # 注意力 CP AllGather
    return get_attention_cp_group().all_gather_into_tensor(output, input)


def get_moe_cp_group() -> GroupCoordinator:  # 获取 MoE CP/DP 组
    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""  # 返回 MOE_DP 组，当 attn_cp_size > moe_dp_size 时包含 CP 伙伴。
    return _get_moe_dp_group()


def get_moe_cp_rank() -> int:  # 获取 MoE CP 排名
    return _get_moe_dp_group().rank_in_group


def get_moe_cp_size() -> int:  # 获取 MoE CP 大小
    return _get_moe_dp_group().world_size


def is_enable_moe_cp_allgather() -> bool:  # 检查是否启用 MoE CP AllGather
    """True when moe_dp_size < attn_cp_size, requiring allgather across CP ranks before MoE."""  # 当 moe_dp_size < attn_cp_size 时为 True，需要在 MoE 前跨 CP 排名进行 AllGather。
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    return sa.attn_cp_size > sa.moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):  # MoE CP AllGather
    return _get_moe_dp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):  # 注意力 TP AllGather（列表版本）
    return get_attention_tp_group().all_gather(input, output_tensor_list=output_list)
