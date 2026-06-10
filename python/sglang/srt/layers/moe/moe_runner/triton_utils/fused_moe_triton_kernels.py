# 融合MoE（混合专家）Triton内核实现文件
# 本文件包含用于MoE推理的Triton GPU内核，包括：
# - 融合MoE矩阵乘法内核（支持FP8/INT8/INT4量化）
# - 激活函数与乘法融合内核
# - MoE求和归约内核
# - 共享专家追加内核
# - TMA描述符缓存管理

from __future__ import annotations  # 启用延迟类型注解评估

import functools  # 导入functools模块，用于lru_cache装饰器
from collections import OrderedDict  # 导入有序字典，用于LRU缓存
from typing import Any, Dict, List, Optional  # 导入类型提示

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled  # 导入批量不变模式检测函数
from sglang.srt.layers.moe.utils import get_moe_padding_size  # 导入MoE填充大小获取函数
from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8量化相关内核
    per_token_group_quant_fp8,  # 按组分每token FP8量化
    scaled_fp8_quant,  # 缩放FP8量化
    sglang_per_token_group_quant_fp8,  # SGLang专用按组分每token FP8量化
)
from sglang.srt.layers.quantization.int8_kernel import (  # 导入INT8量化相关内核
    per_token_group_quant_int8,  # 按组分每token INT8量化
    per_token_quant_int8,  # 每token INT8量化
    sglang_per_token_group_quant_int8,  # SGLang专用按组分每token INT8量化
)
from sglang.srt.utils import (  # 导入工具函数
    cpu_has_amx_support,  # 检测CPU是否支持AMX指令集
    get_bool_env_var,  # 获取布尔型环境变量
    is_cpu,  # 检测是否为CPU平台
    is_cuda,  # 检测是否为CUDA平台
    is_hip,  # 检测是否为HIP(AMD ROCm)平台
    is_sm90_supported,  # 检测是否支持SM90(Hopper)架构
)

try:  # 尝试导入Triton张量描述符
    from triton.tools.tensor_descriptor import TensorDescriptor  # 导入张量描述符类

    _support_tensor_descriptor = True  # 标记支持张量描述符
except:  # 导入失败时
    _support_tensor_descriptor = False  # 标记不支持张量描述符

_is_hip = is_hip()  # 检测当前是否为HIP平台
_is_cuda = is_cuda()  # 检测当前是否为CUDA平台
_is_cpu_amx_available = cpu_has_amx_support()  # 检测CPU AMX是否可用
_is_cpu = is_cpu()  # 检测当前是否为CPU平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER（仅HIP平台）

if _is_cuda:  # 如果是CUDA平台
    pass  # 无额外操作
elif _is_cpu and _is_cpu_amx_available:  # 如果是CPU平台且支持AMX
    pass  # 无额外操作
elif _is_hip:  # 如果是HIP平台
    pass  # 无额外操作

padding_size = get_moe_padding_size(_use_aiter)  # 获取MoE填充大小，取决于是否使用AITER


def support_tensor_descriptor():  # 检查是否支持张量描述符
    """检查当前环境是否支持Triton张量描述符"""
    return _support_tensor_descriptor  # 返回张量描述符支持状态


# swap_ab benefits SM90 GPUs (H20, H100, H200, etc.) for certain block shapes.
# swap_ab在SM90 GPU（H20、H100、H200等）上对特定块形状有益。
@functools.lru_cache(maxsize=8)  # 使用LRU缓存，最多缓存8个结果
def should_enable_swap_ab(  # 判断是否应启用swap_ab优化
    BLOCK_SIZE_M: int,  # M维度块大小
    BLOCK_SIZE_N: int,  # N维度块大小
) -> bool:  # 返回是否启用swap_ab
    """判断是否应在给定块大小下启用AB矩阵交换优化，仅SM90 CUDA平台且BLOCK_SIZE_M<64且BLOCK_SIZE_N>=64时启用"""
    if not _is_cuda or is_batch_invariant_mode_enabled():  # 非CUDA平台或启用批量不变模式
        return False  # 不启用swap_ab

    return is_sm90_supported() and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64  # SM90平台且块大小满足条件时启用


@triton.jit  # Triton JIT编译装饰器
def write_zeros_to_output(  # 将零值写入输出张量的内核函数
    c_ptr,  # 输出张量指针
    stride_cm,  # 输出张量M维度步长
    stride_cn,  # 输出张量N维度步长
    pid_n,  # N维度程序ID
    N,  # N维度大小
    offs_token,  # token偏移量
    token_mask,  # token掩码
    BLOCK_SIZE_M,  # M维度块大小（编译时常量）
    BLOCK_SIZE_N,  # N维度块大小（编译时常量）
    compute_type,  # 计算数据类型（编译时常量）
):  # 当专家不在当前EP秩时，向输出写入零值
    """向输出张量的指定块写入零值，用于过滤掉的专家"""
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)  # 创建零值累加器
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算N维度列偏移
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  # 计算输出指针
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  # 计算存储掩码
    tl.store(c_ptrs, accumulator, mask=c_mask)  # 将零值写入输出


@triton.jit  # Triton JIT编译装饰器
def fused_moe_kernel_gptq_awq(  # GPTQ/AWQ量化融合MoE内核
    # Pointers to matrices
    # 矩阵指针
    a_ptr,  # 输入矩阵A指针
    b_ptr,  # 权重矩阵B指针
    c_ptr,  # 输出矩阵C指针
    b_scale_ptr,  # 权重缩放因子指针
    b_zp_ptr,  # 权重零点指针
    topk_weights_ptr,  # TopK路由权重指针
    sorted_token_ids_ptr,  # 排序后的token ID指针
    expert_ids_ptr,  # 专家ID指针
    num_tokens_post_padded_ptr,  # 填充后token数量指针
    # Matrix dimensions
    # 矩阵维度
    N: tl.constexpr,  # 输出特征维度（编译时常量）
    K: tl.constexpr,  # 输入特征维度（编译时常量）
    EM,  # 专家数*token块数
    num_valid_tokens,  # 有效token数量
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    # 步长变量表示在特定维度上移动1个元素时指针需要增加的量。例如`stride_am`是
    # 向下移动一行时`a_ptr`需要增加的量（A有M行）。
    stride_am,  # A矩阵M维度步长
    stride_ak,  # A矩阵K维度步长
    stride_be,  # B矩阵专家维度步长
    stride_bk,  # B矩阵K维度步长
    stride_bn,  # B矩阵N维度步长
    stride_cm,  # C矩阵M维度步长
    stride_cn,  # C矩阵N维度步长
    stride_bse,  # B缩放因子专家维度步长
    stride_bsk,  # B缩放因子K维度步长
    stride_bsn,  # B缩放因子N维度步长
    stride_bze,  # B零点专家维度步长
    stride_bzk,  # B零点K维度步长
    stride_bzn,  # B零点N维度步长
    group_size: tl.constexpr,  # 量化分组大小（编译时常量）
    # Meta-parameters
    # 元参数
    BLOCK_SIZE_M: tl.constexpr,  # M维度块大小（编译时常量）
    BLOCK_SIZE_N: tl.constexpr,  # N维度块大小（编译时常量）
    BLOCK_SIZE_K: tl.constexpr,  # K维度块大小（编译时常量）
    GROUP_SIZE_M: tl.constexpr,  # M维度分组大小（编译时常量）
    MUL_ROUTED_WEIGHT: tl.constexpr,  # 是否乘以路由权重（编译时常量）
    top_k: tl.constexpr,  # TopK值（编译时常量）
    compute_type: tl.constexpr,  # 计算数据类型（编译时常量）
    has_zp: tl.constexpr,  # 是否有零点（编译时常量）
    use_int4_w4a16: tl.constexpr,  # 是否使用INT4 W4A16量化（编译时常量）
    use_int8_w8a16: tl.constexpr,  # 是否使用INT8 W8A16量化（编译时常量）
    even_Ks: tl.constexpr,  # K维度是否整除BLOCK_SIZE_K（编译时常量）
    filter_expert: tl.constexpr,  # 是否过滤专家（编译时常量）
):  # GPTQ/AWQ量化融合MoE矩阵乘法内核
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.

    实现使用token和专家矩阵的混合专家(MOE)融合计算。
    关键参数：
    - A: 表示token的输入张量，形状为(*, K)，'*'可以是任意批次形状，K是每个token的特征维度。
    - B: 堆叠的MoE权重张量，形状为(E, N, K)，E是专家数量，K是输入特征维度，N是输出特征维度。
    - C: 输出缓存张量，形状为(M, topk, N)，M是填充后的token总数，topk是每个token重复次数，N是输出特征维度。
    - sorted_token_ids: 包含排序后token索引的张量，重复topk次并按分配的专家索引排列。
    - expert_ids: 包含每个块对应专家索引的张量，决定A中每个块使用B中哪个专家矩阵。
    此内核执行token与其对应专家矩阵（由expert_ids决定）的乘法。sorted_token_ids按专家索引
    排序并填充以确保能被BLOCK_SIZE_M整除，这对同一专家处理的不同块间块矩阵乘法的一致性是必要的。
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # -----------------------------------------------------------
    # 将程序ID `pid`映射到它应计算的C矩阵块。
    # 使用分组排序以促进L2数据复用。
    pid = tl.program_id(axis=0)  # 获取当前程序ID
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)  # 计算M维度总块数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # 计算N维度总块数
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 计算每组内的总块数
    group_id = pid // num_pid_in_group  # 计算当前组ID
    first_pid_m = group_id * GROUP_SIZE_M  # 计算当前组的第一个M维度块ID
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 计算当前组实际M维度块大小
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # 计算当前M维度块ID
    pid_n = (pid % num_pid_in_group) // group_size_m  # 计算当前N维度块ID

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # ----------------------------------------------------------
    # 为A和B的第一个块创建指针。
    # 随着K方向移动将推进此指针并累加
    # `a_ptrs`是[BLOCK_SIZE_M, BLOCK_SIZE_K]的指针块
    # `b_ptrs`是[BLOCK_SIZE_K, BLOCK_SIZE_N]的指针块
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)  # 加载填充后token总数
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:  # 如果当前块超出填充范围
        return  # 直接返回，不处理
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)  # 计算token ID偏移
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  # 加载排序后的token ID
    token_mask = offs_token < num_valid_tokens  # 生成有效token掩码

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)  # 加载当前块对应的专家ID
    if filter_expert and off_experts == -1:  # 如果过滤专家且专家ID为-1
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        # -----------------------------------------------------------
        # 当专家不在当前专家并行秩时，向输出写回零值。
        write_zeros_to_output(  # 调用零值写入内核
            c_ptr,  # 输出指针
            stride_cm,  # M维度步长
            stride_cn,  # N维度步长
            pid_n,  # N维度块ID
            N,  # N维度大小
            offs_token,  # token偏移
            token_mask,  # token掩码
            BLOCK_SIZE_M,  # M维度块大小
            BLOCK_SIZE_N,  # N维度块大小
            compute_type,  # 计算类型
        )
        return  # 直接返回

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N  # 计算B矩阵N维度偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # 计算K维度偏移
    a_ptrs = a_ptr + (  # 计算A矩阵指针
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak  # 根据token和K偏移计算
    )

    if use_int4_w4a16:  # 如果使用INT4 W4A16量化
        b_ptrs = (  # 计算B矩阵指针（INT4需要每2个元素合并）
            b_ptr  # 权重基址
            + off_experts * stride_be  # 专家偏移
            + (offs_k[:, None] // 2) * stride_bk  # K维度偏移（每2个K合并）
            + offs_bn[None, :] * stride_bn  # N维度偏移
        )
        b_shifter = (offs_k[:, None] % 2) * 4  # INT4位移量（0或4位）
    elif use_int8_w8a16:  # 如果使用INT8 W8A16量化
        b_ptrs = (  # 计算B矩阵指针
            b_ptr  # 权重基址
            + off_experts * stride_be  # 专家偏移
            + offs_k[:, None] * stride_bk  # K维度偏移
            + offs_bn[None, :] * stride_bn  # N维度偏移
        )

    if not has_zp and use_int4_w4a16:  # 无零点且使用INT4量化
        b_zp_num = 8  # INT4无零点时的默认偏移值为8
    if not has_zp and use_int8_w8a16:  # 无零点且使用INT8量化
        b_zp_num = 128  # INT8无零点时的默认偏移值为128
    elif has_zp and use_int4_w4a16:  # 有零点且使用INT4量化
        b_zp_shifter = (offs_bn[None, :] % 2) * 4  # 零点位移量

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # -----------------------------------------------------------
    # 迭代计算C矩阵的一个块。
    # 累加到[BLOCK_SIZE_M, BLOCK_SIZE_N]的fp32值块中以获得更高精度。
    # 循环结束后`accumulator`将被转换回fp16。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化fp32累加器
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  # 沿K维度迭代
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        # 加载A和B的下一个块，通过检查K维度生成掩码。

        if not even_Ks:  # K维度不整除BLOCK_SIZE_K时
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K  # 生成K维度掩码
            k_other = 0.0  # 越界填充值
        else:  # K维度整除BLOCK_SIZE_K时
            k_mask = None  # 不需要掩码
            k_other = None  # 不需要填充值

        a = tl.load(  # 加载A矩阵块
            a_ptrs,  # A矩阵指针
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),  # 加载掩码
            other=0.0,  # 越界填充值
        )
        b = tl.load(b_ptrs)  # 加载B矩阵块
        if use_int4_w4a16:  # 如果使用INT4量化
            b = (b >> b_shifter) & 0xF  # 右移并掩码提取4位INT4值

        b_scale_ptrs = (  # 计算权重缩放因子指针
            b_scale_ptr  # 缩放因子基址
            + off_experts * stride_bse  # 专家偏移
            + offs_bn[None, :] * stride_bsn  # N维度偏移
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk  # 量化分组偏移
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)  # 加载缩放因子
        b_scale = b_scale.to(tl.float32)  # 转换为fp32

        if has_zp and use_int4_w4a16:  # 有零点且INT4量化
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size  # 计算真实K偏移（量化分组）
            b_zp_ptrs = (  # 计算零点指针
                b_zp_ptr  # 零点基址
                + off_experts * stride_bze  # 专家偏移
                + (offs_bn[None, :] // 2) * stride_bzn  # N维度偏移（每2个合并）
                + offs_k_true * stride_bzk  # 量化分组偏移
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)  # 加载零点
            b_zp = (b_zp >> b_zp_shifter) & 0xF  # 右移并掩码提取4位零点值
            b_zp = b_zp.to(tl.float32)  # 转换为fp32
        elif has_zp and use_int8_w8a16:  # 有零点且INT8量化
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size  # 计算真实K偏移（量化分组）
            b_zp_ptrs = (  # 计算零点指针
                b_zp_ptr  # 零点基址
                + off_experts * stride_bze  # 专家偏移
                + offs_bn[None, :] * stride_bzn  # N维度偏移
                + offs_k_true * stride_bzk  # 量化分组偏移
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)  # 加载零点
            b_zp = b_zp.to(tl.float32)  # 转换为fp32

        # We accumulate along the K dimension.
        # 沿K维度累加。
        if has_zp:  # 有零点时
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)  # 反量化：减零点后乘缩放因子
        else:  # 无零点时
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)  # 反量化：减默认偏移后乘缩放因子
        accumulator = tl.dot(a, b, acc=accumulator)  # 执行矩阵乘法并累加

        # Advance the ptrs to the next K block.
        # 推进指针到下一个K块。
        a_ptrs += BLOCK_SIZE_K * stride_ak  # A矩阵指针推进
        if use_int4_w4a16:  # INT4量化时
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk  # B矩阵指针推进（INT4每2个K合并）
        else:  # 其他量化方式
            b_ptrs += BLOCK_SIZE_K * stride_bk  # B矩阵指针推进

    if MUL_ROUTED_WEIGHT:  # 如果需要乘以路由权重
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)  # 加载路由权重
        accumulator = accumulator * moe_weight[:, None]  # 累加结果乘以路由权重

    accumulator = accumulator.to(compute_type)  # 将累加器转换回计算类型
    # -----------------------------------------------------------
    # Write back the block of the output
    # 写回输出块
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算输出N维度偏移
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  # 计算输出指针
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  # 计算存储掩码
    tl.store(c_ptrs, accumulator, mask=c_mask)  # 存储计算结果


@triton.jit  # Triton JIT编译装饰器
def fused_moe_kernel(  # 融合MoE矩阵乘法内核（主内核）
    # Pointers to matrices
    # 矩阵指针
    a_ptr,  # 输入矩阵A指针
    a_desc,  # A矩阵TMA描述符
    b_ptr,  # 权重矩阵B指针
    b_desc,  # B矩阵TMA描述符
    bias_ptr,  # 偏置指针
    c_ptr,  # 输出矩阵C指针
    a_scale_ptr,  # A缩放因子指针
    b_scale_ptr,  # B缩放因子指针
    topk_weights_ptr,  # TopK路由权重指针
    sorted_token_ids_ptr,  # 排序后的token ID指针
    expert_ids_ptr,  # 专家ID指针
    num_tokens_post_padded_ptr,  # 填充后token数量指针
    add_mask_ptr,  # 加法掩码指针（用于融合加法到输出）
    # Matrix dimensions
    # 矩阵维度
    N,  # 输出特征维度
    K,  # 输入特征维度
    EM,  # 专家数*token块数
    num_valid_tokens,  # 有效token数量
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    # 步长变量表示在特定维度上移动1个元素时指针需要增加的量。例如`stride_am`是
    # 向下移动一行时`a_ptr`需要增加的量（A有M行）。
    stride_am,  # A矩阵M维度步长
    stride_ak,  # A矩阵K维度步长
    stride_be,  # B矩阵专家维度步长
    stride_bk,  # B矩阵K维度步长
    stride_bn,  # B矩阵N维度步长
    stride_bias_e,  # 偏置专家维度步长
    stride_bias_n,  # 偏置N维度步长
    stride_cm,  # C矩阵M维度步长
    stride_cn,  # C矩阵N维度步长
    stride_asm,  # A缩放因子M维度步长
    stride_ask,  # A缩放因子K维度步长
    stride_bse,  # B缩放因子专家维度步长
    stride_bsk,  # B缩放因子K维度步长
    stride_bsn,  # B缩放因子N维度步长
    # Block size for block-wise quantization
    # 块级量化的块大小
    group_n: tl.constexpr,  # N维度量化分组大小（编译时常量）
    group_k: tl.constexpr,  # K维度量化分组大小（编译时常量）
    # Meta-parameters
    # 元参数
    BLOCK_SIZE_M: tl.constexpr,  # M维度块大小（编译时常量）
    BLOCK_SIZE_N: tl.constexpr,  # N维度块大小（编译时常量）
    BLOCK_SIZE_K: tl.constexpr,  # K维度块大小（编译时常量）
    GROUP_SIZE_M: tl.constexpr,  # M维度分组大小（编译时常量）
    MUL_ROUTED_WEIGHT: tl.constexpr,  # 是否乘以路由权重（编译时常量）
    top_k: tl.constexpr,  # TopK值（编译时常量）
    compute_type: tl.constexpr,  # 计算数据类型（编译时常量）
    use_fp8_w8a8: tl.constexpr,  # 是否使用FP8 W8A8量化（编译时常量）
    use_int8_w8a8: tl.constexpr,  # 是否使用INT8 W8A8量化（编译时常量）
    use_int8_w8a16: tl.constexpr,  # 是否使用INT8 W8A16量化（编译时常量）
    per_channel_quant: tl.constexpr,  # 是否使用逐通道量化（编译时常量）
    even_Ks: tl.constexpr,  # K维度是否整除BLOCK_SIZE_K（编译时常量）
    c_sorted: tl.constexpr,  # C输出是否排序（编译时常量）
    filter_expert: tl.constexpr,  # 是否过滤专家（编译时常量）
    swap_ab: tl.constexpr,  # 是否交换A和B矩阵（编译时常量）
    FUSE_ADD_TO_OUTPUT: tl.constexpr,  # 是否融合加法到输出（编译时常量）
    FUSE_SUM_ALL_REDUCE: tl.constexpr,  # 是否融合全归约求和（编译时常量）
    ROUTER_TOPK: tl.constexpr,  # 路由器TopK值（编译时常量）
):  # 融合MoE矩阵乘法内核，支持FP8/INT8量化和多种融合操作
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.

    实现使用token和专家矩阵的混合专家(MOE)融合计算。

    关键参数：
    - A: 表示token的输入张量，形状为(*, K)，'*'可以是任意批次形状，K是每个token的特征维度。
    - B: 堆叠的MoE权重张量，形状为(E, N, K)，E是专家数量，K是输入特征维度，N是输出特征维度。
    - C: 输出缓存张量，形状为(M, topk, N)，M是填充后的token总数，topk是每个token重复次数，N是输出特征维度。
    - sorted_token_ids: 包含排序后token索引的张量，重复topk次并按分配的专家索引排列。
    - expert_ids: 包含每个块对应专家索引的张量，决定A中每个块使用B中哪个专家矩阵。

    此内核执行token与其对应专家矩阵（由expert_ids决定）的乘法。sorted_token_ids按专家索引
    排序并填充以确保能被BLOCK_SIZE_M整除，这对同一专家处理的不同块间块矩阵乘法的一致性是必要的。
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # -----------------------------------------------------------
    # 将程序ID `pid`映射到它应计算的C矩阵块。
    # 使用分组排序以促进L2数据复用。
    pid = tl.program_id(axis=0)  # 获取当前程序ID
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)  # 计算M维度总块数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # 计算N维度总块数
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 计算每组内的总块数
    group_id = pid // num_pid_in_group  # 计算当前组ID
    first_pid_m = group_id * GROUP_SIZE_M  # 计算当前组的第一个M维度块ID
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 计算当前组实际M维度块大小
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # 计算当前M维度块ID
    pid_n = (pid % num_pid_in_group) // group_size_m  # 计算当前N维度块ID

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # ----------------------------------------------------------
    # 为A和B的第一个块创建指针。
    # 随着K方向移动将推进此指针并累加
    # `a_ptrs`是[BLOCK_SIZE_M, BLOCK_SIZE_K]的指针块
    # `b_ptrs`是[BLOCK_SIZE_K, BLOCK_SIZE_N]的指针块
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)  # 加载填充后token总数
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:  # 如果当前块超出填充范围
        return  # 直接返回，不处理
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)  # 计算token ID偏移
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  # 加载排序后的token ID
    offs_token = offs_token.to(tl.int64)  # 转换为int64类型
    token_mask = offs_token < num_valid_tokens  # 生成有效token掩码

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)  # 以int32加载专家ID
    off_experts = off_experts_i32.to(tl.int64)  # 转换为int64类型

    if filter_expert and off_experts == -1:  # 如果过滤专家且专家ID为-1
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        # -----------------------------------------------------------
        # 当专家不在当前专家并行秩时，向输出写回零值。
        if not FUSE_ADD_TO_OUTPUT:  # 如果不融合加法到输出
            # skip the zero-write to preserve existing values.
            # 跳过零值写入以保留现有值。
            write_zeros_to_output(  # 调用零值写入内核
                c_ptr,  # 输出指针
                stride_cm,  # M维度步长
                stride_cn,  # N维度步长
                pid_n,  # N维度块ID
                N,  # N维度大小
                offs_token,  # token偏移
                token_mask,  # token掩码
                BLOCK_SIZE_M,  # M维度块大小
                BLOCK_SIZE_N,  # N维度块大小
                compute_type,  # 计算类型
            )
        return  # 直接返回

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N  # 计算B矩阵N维度偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # 计算K维度偏移
    if a_desc is not None:  # 如果使用A矩阵TMA描述符
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0  # 断言：TMA仅用于块级FP8量化
        start_offs_m = pid_m * BLOCK_SIZE_M  # 计算M维度起始偏移
    else:  # 不使用TMA描述符
        a_ptrs = a_ptr + (  # 计算A矩阵指针
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak  # 根据token和K偏移计算
        )

    if b_desc is not None:  # 如果使用B矩阵TMA描述符
        start_offs_n = pid_n * BLOCK_SIZE_N  # 计算N维度起始偏移
    else:  # 不使用TMA描述符
        b_ptrs = (  # 计算B矩阵指针
            b_ptr  # 权重基址
            + off_experts * stride_be  # 专家偏移
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # K和N维度偏移
        )

    if bias_ptr is not None:  # 如果有偏置
        bias = tl.load(  # 加载偏置
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n  # 计算偏置指针
        )
    if use_int8_w8a16:  # 如果使用INT8 W8A16量化
        b_scale_ptrs = (  # 计算权重缩放因子指针
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn  # 专家和N维度偏移
        )
        b_scale = tl.load(b_scale_ptrs)  # 加载缩放因子

    if use_fp8_w8a8 or use_int8_w8a8:  # 如果使用FP8或INT8 W8A8量化
        # block-wise
        # 块级量化
        if group_k > 0 and group_n > 0:  # 块级量化模式
            if a_desc is not None:  # 使用TMA时
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm  # A缩放因子指针
            else:  # 不使用TMA时
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm  # A缩放因子指针
            if BLOCK_SIZE_N > group_n:  # N维度块大于量化分组
                offs_bsn = offs_bn // group_n  # 计算B缩放因子N维度偏移
            else:  # N维度块不大于量化分组
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n  # 按块计算B缩放因子N维度偏移
            b_scale_ptrs = (  # 计算B缩放因子指针
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn  # 专家和N维度偏移
            )
        # channel-wise
        # 逐通道量化
        elif per_channel_quant:  # 逐通道量化模式
            b_scale_ptrs = (  # 计算B缩放因子指针
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn  # 专家和N维度偏移
            )
            b_scale = tl.load(b_scale_ptrs)  # 加载B缩放因子
            # Load per-token scale for activations
            # 加载激活值的逐token缩放因子
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm  # 计算A缩放因子指针
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]  # 加载A缩放因子
        # tensor-wise
        # 张量级量化
        else:  # 张量级量化模式
            a_scale = tl.load(a_scale_ptr)  # 加载A张量级缩放因子
            b_scale = tl.load(b_scale_ptr + off_experts)  # 加载B张量级缩放因子

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # -----------------------------------------------------------
    # 迭代计算C矩阵的一个块。
    # 累加到[BLOCK_SIZE_M, BLOCK_SIZE_N]的fp32值块中以获得更高精度。
    # 循环结束后`accumulator`将被转换回fp16。
    if swap_ab:  # 如果交换A和B矩阵
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)  # 创建转置形状的累加器
    else:  # 不交换
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 创建正常形状的累加器

    for k_start in range(0, K, BLOCK_SIZE_K):  # 沿K维度迭代
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        # 加载A和B的下一个块，通过检查K维度生成掩码。
        if a_desc is not None:  # 使用TMA描述符加载A
            a = a_desc.load([start_offs_m, k_start])  # 通过TMA加载A矩阵块
        elif even_Ks:  # K维度整除时
            a = tl.load(  # 加载A矩阵块
                a_ptrs,  # A矩阵指针
                mask=token_mask[:, None],  # token掩码
                other=0.0,  # 越界填充值
            )
        else:  # K维度不整除时
            a = tl.load(  # 加载A矩阵块
                a_ptrs,  # A矩阵指针
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),  # token和K维度掩码
                other=0.0,  # 越界填充值
            )

        if b_desc is not None:  # 使用TMA描述符加载B
            b = (  # 通过TMA加载B矩阵块
                b_desc.load([off_experts_i32, start_offs_n, k_start])  # 加载指定位置
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)  # 重塑形状
                .T  # 转置
            )
        elif even_Ks:  # K维度整除时
            b = tl.load(b_ptrs)  # 加载B矩阵块
        else:  # K维度不整除时
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)  # 带掩码加载B矩阵块

        # We accumulate along the K dimension.
        # 沿K维度累加。
        if use_int8_w8a16:  # INT8 W8A16量化
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)  # 矩阵乘法并累加
        elif use_fp8_w8a8 or use_int8_w8a8:  # FP8或INT8 W8A8量化
            if group_k > 0 and group_n > 0:  # 块级量化
                offs_ks = k_start // group_k  # 计算K维度量化分组偏移
                a_scale = tl.load(  # 加载A缩放因子
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0  # 带掩码加载
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)  # 加载B缩放因子
                if swap_ab:  # 如果交换AB
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))  # 转置A和B
                    a_scale, b_scale = b_scale, a_scale  # 交换缩放因子
                if BLOCK_SIZE_N > group_n:  # N维度块大于量化分组
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]  # 逐元素乘缩放因子
                else:  # N维度块不大于量化分组
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)  # 缩放因子广播相乘
            else:  # 非块级量化
                if use_fp8_w8a8:  # FP8 W8A8量化
                    if swap_ab:  # 如果交换AB
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))  # 转置A和B
                    accumulator = tl.dot(a, b, acc=accumulator)  # 矩阵乘法并累加
                else:  # INT8 W8A8量化
                    accumulator += tl.dot(a, b)  # 矩阵乘法并累加
        else:  # 无量化
            accumulator += tl.dot(a, b)  # 矩阵乘法并累加
        # Advance the ptrs to the next K block.
        # 推进指针到下一个K块。
        if a_desc is None:  # 不使用TMA时
            a_ptrs += BLOCK_SIZE_K * stride_ak  # A矩阵指针推进
        if b_desc is None:  # 不使用TMA时
            b_ptrs += BLOCK_SIZE_K * stride_bk  # B矩阵指针推进

    if swap_ab:  # 如果交换了AB
        accumulator = tl.trans(accumulator, (1, 0))  # 转置累加器回正常顺序

    if use_int8_w8a16:  # INT8 W8A16量化
        accumulator *= b_scale  # 乘以B缩放因子
    elif use_fp8_w8a8 or use_int8_w8a8:  # FP8或INT8 W8A8量化
        if group_k == 0 or group_n == 0:  # 非块级量化
            accumulator *= a_scale * b_scale  # 乘以A和B缩放因子

    if bias_ptr is not None:  # 如果有偏置
        accumulator += bias  # 加上偏置

    if MUL_ROUTED_WEIGHT:  # 如果需要乘以路由权重
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)  # 加载路由权重
        accumulator *= moe_weight[:, None]  # 累加结果乘以路由权重

    accumulator = accumulator.to(compute_type)  # 将累加器转换回计算类型
    # -----------------------------------------------------------
    # Write back the block of the output
    # 写回输出块
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算输出N维度偏移

    if FUSE_ADD_TO_OUTPUT:  # 融合加法到输出模式
        # Accumulate into existing output with per-token mask.
        # 使用逐token掩码累加到现有输出。
        offs_token_out = offs_token // ROUTER_TOPK  # 计算输出token偏移（除以路由TopK）
        add_mask = tl.load(add_mask_ptr + offs_token_out, mask=token_mask, other=False)  # 加载加法掩码
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  # 计算输出指针
        c_mask = token_mask[:, None] & add_mask[:, None] & (offs_cn[None, :] < N)  # 计算存储掩码
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)  # 加载已有输出值
        tl.store(c_ptrs, existing + accumulator, mask=c_mask)  # 累加并存储
    elif FUSE_SUM_ALL_REDUCE:  # 融合全归约求和模式
        offs_token_out = offs_token // ROUTER_TOPK  # 计算输出token偏移（除以路由TopK）
        c_ptrs = (  # 计算输出指针
            c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]  # 按输出token偏移
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  # 计算存储掩码
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)  # 原子加法存储
    else:  # 普通输出模式
        if c_sorted:  # C输出已排序
            c_ptrs = (  # 计算输出指针（使用token ID偏移）
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_cn[None, :]
            )
        else:  # C输出未排序
            c_ptrs = (  # 计算输出指针（使用token偏移）
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  # 计算存储掩码
        tl.store(c_ptrs, accumulator, mask=c_mask)  # 存储计算结果


# -----------------------------------------------------------------------------
# TMA allocator: set once per process (avoid per-call triton.set_allocator)
# TMA分配器：每个进程设置一次（避免每次调用triton.set_allocator）
# -----------------------------------------------------------------------------
_TMA_ALLOCATOR_SET = False  # TMA分配器是否已设置的标志


def _set_triton_tma_allocator():  # 设置Triton TMA全局分配器
    """TMA descriptors require a global allocator; set it once to avoid per-call overhead."""  # TMA描述符需要全局分配器；设置一次以避免每次调用的开销
    """TMA描述符需要全局分配器；设置一次以避免每次调用的开销。"""
    global _TMA_ALLOCATOR_SET  # 声明全局变量
    if _TMA_ALLOCATOR_SET:  # 如果已设置
        return  # 直接返回

    # TMA descriptors require a global memory allocation
    # TMA描述符需要全局内存分配
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):  # TMA内存分配函数
        # NOTE: keep this allocation on CUDA device
        # 注意：保持此分配在CUDA设备上
        return torch.empty(size, device="cuda", dtype=torch.int8)  # 在CUDA设备上分配内存

    triton.set_allocator(alloc_fn)  # 设置Triton分配器
    _TMA_ALLOCATOR_SET = True  # 标记已设置


# --- B TensorDescriptor cache (LRU) ---
# --- B张量描述符缓存（LRU） ---
_B_DESC_CACHE_MAX = 64  # B描述符缓存最大容量
_B_DESC_CACHE: "OrderedDict[tuple, TensorDescriptor]" = OrderedDict()  # B描述符LRU缓存


def _get_b_tma_desc_cached(B: torch.Tensor, block_n: int, block_k: int):  # 获取缓存的B矩阵TMA描述符
    """
    Cache TensorDescriptor for constant weight B.
    Keyed by storage ptr + shape/stride/dtype + tile shape.

    缓存常量权重B的张量描述符。
    以存储指针+形状/步长/数据类型+块形状为键。
    """
    key = (  # 构建缓存键
        int(B.data_ptr()),  # 张量数据指针
        tuple(B.shape),  # 张量形状
        tuple(B.stride()),  # 张量步长
        str(B.dtype),  # 张量数据类型
        int(block_n),  # N维度块大小
        int(block_k),  # K维度块大小
    )

    desc = _B_DESC_CACHE.get(key, None)  # 从缓存中查找描述符
    if desc is not None:  # 如果命中缓存
        _B_DESC_CACHE.move_to_end(key)  # 将该键移到末尾（最近使用）
        return desc  # 返回缓存的描述符

    # Create outside lock to reduce lock hold time (ok if duplicated rarely)
    # 在锁外创建以减少锁持有时间（偶尔重复创建可以接受）
    desc = TensorDescriptor(  # 创建新的张量描述符
        B,  # 权重张量
        B.shape,  # 张量形状
        B.stride(),  # 张量步长
        [1, block_n, block_k],  # 块形状（1专家 x block_n x block_k）
    )

    _B_DESC_CACHE[key] = desc  # 将描述符放入缓存
    _B_DESC_CACHE.move_to_end(key)  # 将该键移到末尾（最近使用）
    if len(_B_DESC_CACHE) > _B_DESC_CACHE_MAX:  # 如果缓存超过最大容量
        _B_DESC_CACHE.popitem(last=False)  # 移除最久未使用的项

    return desc  # 返回描述符


def invoke_fused_moe_kernel(  # 调用融合MoE内核的入口函数
    A: torch.Tensor,  # 输入张量A
    B: torch.Tensor,  # 权重张量B
    bias: Optional[torch.Tensor],  # 偏置张量（可选）
    C: torch.Tensor,  # 输出张量C
    A_scale: Optional[torch.Tensor],  # A缩放因子（可选）
    B_scale: Optional[torch.Tensor],  # B缩放因子（可选）
    B_zp: Optional[torch.Tensor],  # B零点（可选）
    topk_weights: torch.Tensor,  # TopK路由权重
    topk_ids: torch.Tensor,  # TopK专家ID
    sorted_token_ids: torch.Tensor,  # 排序后的token ID
    expert_ids: torch.Tensor,  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # 填充后token数量
    mul_routed_weight: bool,  # 是否乘以路由权重
    top_k: int,  # TopK值
    config: Dict[str, Any],  # 内核配置字典
    compute_type: tl.dtype,  # 计算数据类型
    use_fp8_w8a8: bool,  # 是否使用FP8 W8A8量化
    use_int8_w8a8: bool,  # 是否使用INT8 W8A8量化
    use_int8_w8a16: bool,  # 是否使用INT8 W8A16量化
    use_int4_w4a16: bool,  # 是否使用INT4 W4A16量化
    per_channel_quant: bool,  # 是否使用逐通道量化
    block_shape: Optional[List[int]] = None,  # 量化块形状（可选）
    no_combine: bool = False,  # 是否不合并（未使用）
    a_use_tma: bool = False,  # A矩阵是否使用TMA
    b_use_tma: bool = False,  # B矩阵是否使用TMA
    c_sorted: bool = False,  # C输出是否已排序
    filter_expert: bool = True,  # 是否过滤专家
    fuse_sum_all_reduce: bool = False,  # 是否融合全归约求和
    router_topk: int = 1,  # 路由器TopK值
    fuse_add_to_output: bool = False,  # 是否融合加法到输出
    add_output_mask: Optional[torch.Tensor] = None,  # 加法输出掩码（可选）
) -> None:  # 调用融合MoE内核，根据量化类型和配置选择合适的内核
    """根据量化类型和配置调用相应的融合MoE内核函数"""
    assert topk_weights.stride(1) == 1  # 断言：TopK权重的内维度步长为1（连续）
    assert sorted_token_ids.stride(0) == 1  # 断言：排序token ID的步长为1（连续）

    if use_fp8_w8a8:  # 如果使用FP8 W8A8量化
        swap_ab = should_enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])  # 检查是否应启用swap_ab
    else:  # 不使用FP8量化
        swap_ab = False  # 不启用swap_ab

    padded_size = 0  # 初始化填充大小
    if use_fp8_w8a8:  # FP8 W8A8量化模式
        assert B_scale is not None  # 断言：B缩放因子必须存在
        if block_shape is None:  # 无块级量化
            # activation tensor-wise fp8 quantization, dynamic or static
            # 激活值张量级FP8量化，动态或静态
            padded_size = padding_size  # 设置填充大小
            # activations apply per-token quantization when weights apply per-channel quantization by default
            # 当权重使用逐通道量化时，激活值默认使用逐token量化
            A, A_scale = scaled_fp8_quant(  # 对激活值进行FP8量化
                A, A_scale, use_per_token_if_dynamic=per_channel_quant  # 逐通道时使用逐token量化
            )
        else:  # 块级量化
            # activation block-wise fp8 quantization
            # 激活值块级FP8量化
            assert len(block_shape) == 2  # 断言：块形状长度为2
            block_n, block_k = block_shape[0], block_shape[1]  # 获取N和K维度块大小
            if _is_cuda:  # CUDA平台
                A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)  # SGLang专用分组FP8量化
            else:  # 其他平台
                A, A_scale = per_token_group_quant_fp8(A, block_k)  # 通用分组FP8量化
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]  # 断言：A缩放因子K维度对齐
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]  # 断言：B缩放因子N维度对齐
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]  # 断言：B缩放因子K维度对齐
    elif use_int8_w8a8:  # INT8 W8A8量化模式
        assert B_scale is not None  # 断言：B缩放因子必须存在
        if block_shape is None:  # 无块级量化
            # activation channel-wise int8 quantization
            # 激活值逐通道INT8量化
            assert (  # 断言
                per_channel_quant  # 必须使用逐通道量化
            ), "int8 quantization only supports channel-wise quantization except for block-wise quantization"  # INT8量化只支持逐通道量化（除块级量化外）
            A, A_scale = per_token_quant_int8(A)  # 对激活值进行INT8逐token量化
        else:  # 块级量化
            # activation block-wise int8 quantization
            # 激活值块级INT8量化
            assert len(block_shape) == 2  # 断言：块形状长度为2
            block_n, block_k = block_shape[0], block_shape[1]  # 获取N和K维度块大小
            if _is_cuda:  # CUDA平台
                A, A_scale = sglang_per_token_group_quant_int8(A, block_k)  # SGLang专用分组INT8量化
            else:  # 其他平台
                A, A_scale = per_token_group_quant_int8(A, block_k)  # 通用分组INT8量化
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]  # 断言：A缩放因子K维度对齐
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]  # 断言：B缩放因子N维度对齐
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]  # 断言：B缩放因子K维度对齐
    elif use_int8_w8a16 or use_int4_w4a16:  # INT8 W8A16或INT4 W4A16量化模式
        assert B_scale is not None  # 断言：B缩放因子必须存在
        assert block_shape is None or block_shape[0] == 0  # 断言：无块级量化或N维度块大小为0
    else:  # 无量化模式
        assert A_scale is None  # 断言：A缩放因子不存在
        assert B_scale is None  # 断言：B缩放因子不存在

    grid = lambda META: (  # 定义网格大小计算函数
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])  # M维度块数
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),  # 乘以N维度块数
    )

    K = B.shape[2] - padded_size  # 计算实际K维度大小（减去填充）
    if K % config["BLOCK_SIZE_K"] == 0:  # K维度能整除块大小时
        even_Ks = True  # 标记K维度对齐
    else:  # K维度不能整除时
        even_Ks = False  # 标记K维度不对齐

    if fuse_sum_all_reduce:  # 融合全归约求和模式
        assert not c_sorted, "fuse_sum_all_reduce only supports c_sorted=False"  # 断言：不支持c_sorted
    if fuse_add_to_output:  # 融合加法到输出模式
        assert (  # 断言
            not fuse_sum_all_reduce  # 不能同时使用融合全归约
        ), "fuse_add_to_output and fuse_sum_all_reduce are mutually exclusive"  # 融合加法和融合全归约互斥
        assert (  # 断言
            add_output_mask is not None  # 必须提供加法掩码
        ), "add_output_mask required when fuse_add_to_output=True"  # 融合加法模式需要加法掩码

    if (  # 如果使用GPTQ/AWQ量化且有块形状
        (use_int8_w8a16 or use_int4_w4a16)
        and block_shape is not None
        and block_shape[1] > 0
    ):
        assert (  # 断言
            not fuse_sum_all_reduce  # 不支持融合全归约
        ), "fuse_sum_all_reduce is not supported for GPTQ/AWQ kernels"  # GPTQ/AWQ内核不支持融合全归约
        assert B_scale is not None and B_scale.ndim == 3  # 断言：B缩放因子为3维
        assert B_zp is None or B_zp.ndim == 3  # 断言：B零点为None或3维
        assert bias is None  # 断言：无偏置
        fused_moe_kernel_gptq_awq[grid](  # 调用GPTQ/AWQ专用MoE内核
            A,  # 输入张量
            B,  # 权重张量
            C,  # 输出张量
            B_scale,  # B缩放因子
            B_zp,  # B零点
            topk_weights,  # TopK路由权重
            sorted_token_ids,  # 排序后token ID
            expert_ids,  # 专家ID
            num_tokens_post_padded,  # 填充后token数量
            B.shape[1],  # N维度大小
            A.shape[1],  # K维度大小
            sorted_token_ids.shape[0],  # EM维度大小
            topk_ids.numel(),  # 有效token数
            A.stride(0),  # A矩阵M步长
            A.stride(1),  # A矩阵K步长
            B.stride(0),  # B矩阵专家步长
            B.stride(2),  # B矩阵K步长
            B.stride(1),  # B矩阵N步长
            C.stride(-2),  # C矩阵M步长
            C.stride(-1),  # C矩阵N步长
            B_scale.stride(0),  # B缩放因子专家步长
            B_scale.stride(2),  # B缩放因子K步长
            B_scale.stride(1),  # B缩放因子N步长
            B_zp.stride(0) if B_zp is not None else 0,  # B零点专家步长
            B_zp.stride(2) if B_zp is not None else 0,  # B零点K步长
            B_zp.stride(1) if B_zp is not None else 0,  # B零点N步长
            group_size=block_shape[1],  # 量化分组大小
            MUL_ROUTED_WEIGHT=mul_routed_weight,  # 是否乘以路由权重
            top_k=top_k,  # TopK值
            compute_type=compute_type,  # 计算类型
            has_zp=B_zp is not None,  # 是否有零点
            use_int4_w4a16=use_int4_w4a16,  # 是否INT4 W4A16量化
            use_int8_w8a16=use_int8_w8a16,  # 是否INT8 W8A16量化
            even_Ks=even_Ks,  # K维度是否对齐
            filter_expert=filter_expert,  # 是否过滤专家
            **config,  # 其他配置参数
        )

    else:  # 非GPTQ/AWQ量化模式
        if a_use_tma or b_use_tma:  # 如果使用TMA
            _set_triton_tma_allocator()  # 设置TMA全局分配器

        if a_use_tma:  # 如果A使用TMA
            a_desc = TensorDescriptor(  # 创建A矩阵TMA描述符
                A, A.shape, A.stride(), [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]]  # 指定块形状
            )
        else:  # A不使用TMA
            a_desc = None  # 不使用描述符
        if b_use_tma:  # 如果B使用TMA
            # B is constant weights -> cache descriptor
            # B是常量权重 -> 缓存描述符
            b_desc = _get_b_tma_desc_cached(  # 获取缓存的B矩阵TMA描述符
                B,  # 权重张量
                config["BLOCK_SIZE_N"],  # N维度块大小
                config["BLOCK_SIZE_K"],  # K维度块大小
            )
        else:  # B不使用TMA
            b_desc = None  # 不使用描述符

        fused_moe_kernel[grid](  # 调用主MoE内核
            A,  # 输入张量
            a_desc,  # A矩阵TMA描述符
            B,  # 权重张量
            b_desc,  # B矩阵TMA描述符
            bias,  # 偏置
            C,  # 输出张量
            A_scale,  # A缩放因子
            B_scale,  # B缩放因子
            topk_weights,  # TopK路由权重
            sorted_token_ids,  # 排序后token ID
            expert_ids,  # 专家ID
            num_tokens_post_padded,  # 填充后token数量
            add_output_mask,  # 加法输出掩码
            B.shape[1],  # N维度大小
            B.shape[2] - padded_size,  # K维度大小（减去填充）
            sorted_token_ids.shape[0],  # EM维度大小
            topk_ids.numel(),  # 有效token数
            A.stride(0),  # A矩阵M步长
            A.stride(1),  # A矩阵K步长
            B.stride(0),  # B矩阵专家步长
            B.stride(2),  # B矩阵K步长
            B.stride(1),  # B矩阵N步长
            bias.stride(0) if bias is not None else 0,  # 偏置专家步长
            bias.stride(1) if bias is not None else 0,  # 偏置N步长
            C.stride(-2),  # C矩阵M步长
            C.stride(-1),  # C矩阵N步长
            A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,  # A缩放因子M步长
            A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,  # A缩放因子K步长
            B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,  # B缩放因子专家步长
            B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,  # B缩放因子K步长
            B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,  # B缩放因子N步长
            0 if block_shape is None else block_shape[0],  # 量化分组N大小
            0 if block_shape is None else block_shape[1],  # 量化分组K大小
            MUL_ROUTED_WEIGHT=mul_routed_weight,  # 是否乘以路由权重
            top_k=top_k,  # TopK值
            compute_type=compute_type,  # 计算类型
            use_fp8_w8a8=use_fp8_w8a8,  # 是否FP8 W8A8量化
            use_int8_w8a8=use_int8_w8a8,  # 是否INT8 W8A8量化
            use_int8_w8a16=use_int8_w8a16,  # 是否INT8 W8A16量化
            per_channel_quant=per_channel_quant,  # 是否逐通道量化
            even_Ks=even_Ks,  # K维度是否对齐
            c_sorted=c_sorted,  # C输出是否排序
            filter_expert=filter_expert,  # 是否过滤专家
            swap_ab=swap_ab,  # 是否交换AB
            FUSE_ADD_TO_OUTPUT=fuse_add_to_output,  # 是否融合加法到输出
            FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,  # 是否融合全归约
            ROUTER_TOPK=router_topk,  # 路由器TopK值
            **config,  # 其他配置参数
        )


@triton.jit  # Triton JIT编译装饰器
def tanh(x):  # Triton实现的tanh函数
    """Triton实现的tanh激活函数，使用缩放sigmoid近似"""
    return 2 * tl.sigmoid(2 * x) - 1  # tanh(x) = 2*sigmoid(2x) - 1


@triton.jit  # Triton JIT编译装饰器
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):  # 应用激活函数
    """
    Apply activation function based on compile-time constant.

    Args:
        x: Input tensor (converted to float32 inside)
        ACTIVATION_TYPE: Compile-time constant string ("silu" or "gelu")

    Returns:
        Activated output in the same dtype as input

    根据编译时常量应用激活函数。

    参数：
        x: 输入张量（内部转换为float32）
        ACTIVATION_TYPE: 编译时常量字符串（"silu"或"gelu"）

    返回：
        与输入相同数据类型的激活输出
    """
    x = x.to(tl.float32)  # 转换为float32以提高精度
    if ACTIVATION_TYPE == "silu":  # SiLU激活函数
        return x * tl.sigmoid(x)  # SiLU(x) = x * sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":  # GELU激活函数
        kAlpha = 0.7978845608028654  # GELU近似系数 = sqrt(2/pi)
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))  # GELU近似公式
    else:  # 不支持的激活函数
        raise ValueError(f"Unsupported activation: {ACTIVATION_TYPE}")  # 抛出异常


@triton.jit  # Triton JIT编译装饰器
def act_and_mul_kernel(  # 激活函数与乘法融合内核
    gateup_output,  # gate和up拼接的输出张量
    down_input,  # down投影输入张量（输出）
    hidden_size,  # 隐藏层大小
    expert_ids_ptr,  # 专家ID指针
    expert_step: tl.constexpr,  # 专家步长（编译时常量）
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
    ACTIVATION_TYPE: tl.constexpr,  # 激活函数类型（编译时常量）
    SWIGLU_LIMIT: tl.constexpr = 0.0,  # SwiGLU限制值（编译时常量，默认0）
    HAS_SWIGLU_LIMIT: tl.constexpr = False,  # 是否有SwiGLU限制（编译时常量，默认False）
):  # 统一的激活与乘法内核，处理排序和未排序路由，支持SiLU和GELU激活
    """
    Unified activation and multiply kernel that handles both sorted and unsorted routing,
    and both SiLU and GELU activations using compile-time constants.

    统一的激活与乘法内核，使用编译时常量处理排序和未排序路由，
    以及SiLU和GELU激活函数。
    """
    InDtype = gateup_output.dtype.element_ty  # 获取输入数据类型
    OutDtype = down_input.dtype.element_ty  # 获取输出数据类型

    half_hidden_size = hidden_size // 2  # 隐藏层大小的一半（gate和up各占一半）
    pid = tl.program_id(0)  # 获取当前程序ID

    expert_id = tl.load(expert_ids_ptr + pid // expert_step)  # 加载当前专家ID

    if expert_id == -1:  # 如果专家ID为-1（被过滤的专家）
        return  # 直接返回，不处理

    gateup_output_ptr = gateup_output + pid * hidden_size  # 计算gateup输出指针
    down_input_ptr = down_input + pid * half_hidden_size  # 计算down输入指针
    gate_output_ptr = gateup_output_ptr  # gate部分指针（前半部分）
    up_output_ptr = gateup_output_ptr + half_hidden_size  # up部分指针（后半部分）

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):  # 分块处理
        offset = start_offset + tl.arange(0, BLOCK_SIZE)  # 计算偏移
        mask = offset < half_hidden_size  # 生成掩码

        gate_output = tl.load(gate_output_ptr + offset, mask=mask)  # 加载gate输出
        up_output = tl.load(up_output_ptr + offset, mask=mask)  # 加载up输出

        if HAS_SWIGLU_LIMIT:  # 如果有SwiGLU限制
            gate_output = tl.minimum(gate_output, SWIGLU_LIMIT)  # gate限制在[-inf, L]
            up_output = tl.maximum(tl.minimum(up_output, SWIGLU_LIMIT), -SWIGLU_LIMIT)  # up限制在[-L, L]

        gate_output_activated = _apply_activation(gate_output, ACTIVATION_TYPE)  # 对gate应用激活函数
        gate_output_activated = gate_output_activated.to(InDtype)  # 转换回输入数据类型

        act_mul_output = gate_output_activated * up_output  # 激活后的gate与up相乘
        act_mul_output = act_mul_output.to(OutDtype)  # 转换为输出数据类型
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)  # 存储结果


def act_and_mul_triton(  # 使用Triton实现的激活与乘法函数
    gateup_output: torch.Tensor,  # gate和up拼接的输出
    down_input: torch.Tensor,  # down投影输入（输出）
    config: Dict[str, Any],  # 配置字典
    topk_ids: Optional[torch.Tensor] = None,  # TopK专家ID（未排序路由时使用）
    expert_ids: Optional[torch.Tensor] = None,  # 专家ID（排序路由时使用）
    down_moe_use_tma: bool = False,  # 是否使用排序路由布局
    activation: str = "silu",  # 激活函数类型（默认SiLU）
    swiglu_limit: Optional[float] = None,  # SwiGLU限制值（可选）
) -> None:  # 调用Triton激活与乘法内核
    """
    Args:
        gateup_output: Input tensor containing gate and up outputs concatenated
        down_input: Output tensor for the result
        config: Configuration dictionary with BLOCK_SIZE_M and BLOCK_SIZE_N
        topk_ids: Expert IDs for unsorted routing (used when down_moe_use_tma=False)
        expert_ids: Expert IDs for sorted routing (used when down_moe_use_tma=True)
        down_moe_use_tma: Whether to use sorted routing layout
        activation: Activation type ("silu" or "gelu")
        swiglu_limit: if not None, clamp gate to [-inf, L] and up to [-L, L] before activation
                      (compiles a separate kernel variant via tl.constexpr).

    参数：
        gateup_output: 包含gate和up输出拼接的输入张量
        down_input: 结果的输出张量
        config: 包含BLOCK_SIZE_M和BLOCK_SIZE_N的配置字典
        topk_ids: 未排序路由的专家ID（down_moe_use_tma=False时使用）
        expert_ids: 排序路由的专家ID（down_moe_use_tma=True时使用）
        down_moe_use_tma: 是否使用排序路由布局
        activation: 激活类型（"silu"或"gelu"）
        swiglu_limit: 如果非None，在激活前将gate截断到[-inf, L]，up截断到[-L, L]
                      （通过tl.constexpr编译单独的内核变体）。
    """
    grid = (down_input.shape[0],)  # 定义网格大小（token数）
    hidden_size = gateup_output.shape[1]  # 获取隐藏层大小
    expert_ids_row = topk_ids.view(-1) if not down_moe_use_tma else expert_ids  # 根据路由方式选择专家ID
    expert_step = 1 if not down_moe_use_tma else config["BLOCK_SIZE_M"]  # 根据路由方式设置专家步长
    has_swiglu_limit = swiglu_limit is not None  # 检查是否有SwiGLU限制
    act_and_mul_kernel[grid](  # 调用激活与乘法内核
        gateup_output,  # gateup输出
        down_input,  # down输入
        hidden_size,  # 隐藏层大小
        expert_ids_row,  # 专家ID
        expert_step,  # 专家步长
        BLOCK_SIZE=512,  # 块大小
        ACTIVATION_TYPE=activation,  # 激活函数类型
        SWIGLU_LIMIT=float(swiglu_limit) if has_swiglu_limit else 0.0,  # SwiGLU限制值
        HAS_SWIGLU_LIMIT=has_swiglu_limit,  # 是否有SwiGLU限制
    )


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
# _moe_sum_reduce_kernel内核修改自 https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit  # Triton JIT编译装饰器
def _moe_sum_reduce_kernel(  # MoE求和归约内核
    input_ptr,  # 输入张量指针
    input_stride_0,  # 输入第0维步长
    input_stride_1,  # 输入第1维步长
    input_stride_2,  # 输入第2维步长
    output_ptr,  # 输出张量指针
    output_stride_0,  # 输出第0维步长
    output_stride_1,  # 输出第1维步长
    token_num: int,  # token数量
    topk_num: int,  # TopK数量
    hidden_dim: int,  # 隐藏维度
    routed_scaling_factor: tl.constexpr,  # 路由缩放因子（编译时常量）
    BLOCK_M: tl.constexpr,  # M维度块大小（编译时常量）
    BLOCK_DIM: tl.constexpr,  # 维度块大小（编译时常量）
    NUM_STAGE: tl.constexpr,  # 流水线阶段数（编译时常量）
):  # 将多个专家的输出沿TopK维度求和归约
    """MoE专家输出求和归约内核，沿TopK维度累加并乘以路由缩放因子"""
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)  # 转换步长为int64
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)  # 转换步长为int64
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)  # 转换步长为int64

    token_block_id = tl.program_id(0)  # 获取token块ID
    dim_block_id = tl.program_id(1)  # 获取维度块ID

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)  # 计算token偏移
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)  # 计算维度偏移

    mask_token = offs_token < token_num  # 生成token掩码
    mask_dim = offs_dim < hidden_dim  # 生成维度掩码

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]  # 计算基础指针

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)  # 初始化累加器

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):  # 沿TopK维度迭代（带流水线）
        tile = tl.load(  # 加载当前TopK的数据块
            base_ptrs + i * input_stride_1,  # 计算当前TopK的指针偏移
            mask=mask_token[:, None] & mask_dim[None, :],  # 掩码
            other=0.0,  # 越界填充值
        )
        accumulator += tile.to(tl.float32)  # 累加到fp32
    accumulator *= routed_scaling_factor  # 乘以路由缩放因子

    # -------- Write back --------
    # -------- 写回 --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]  # 计算存储指针
    tl.store(  # 存储结果
        store_ptrs,  # 存储指针
        accumulator.to(input_ptr.dtype.element_ty),  # 转换回输入数据类型
        mask=mask_token[:, None] & mask_dim[None, :],  # 掩码
    )


def moe_sum_reduce_triton(  # 使用Triton实现的MoE求和归约
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float  # 输入、输出、路由缩放因子
):  # 将MoE专家输出沿TopK维度求和归约
    """使用Triton内核实现MoE专家输出沿TopK维度的求和归约"""
    assert input.is_contiguous()  # 断言：输入连续
    assert output.is_contiguous()  # 断言：输出连续

    token_num, topk_num, hidden_dim = input.shape  # 获取输入形状
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim  # 断言：输出形状正确

    BLOCK_M = 1  # M维度块大小
    BLOCK_DIM = 2048  # 维度块大小
    NUM_STAGE = 1  # 流水线阶段数
    num_warps = 16  # warp数量

    grid = (  # 定义网格大小
        triton.cdiv(token_num, BLOCK_M),  # token块数
        triton.cdiv(hidden_dim, BLOCK_DIM),  # 维度块数
    )

    _moe_sum_reduce_kernel[grid](  # 调用MoE求和归约内核
        input,  # 输入张量
        *input.stride(),  # 输入步长
        output,  # 输出张量
        *output.stride(),  # 输出步长
        token_num=token_num,  # token数量
        topk_num=topk_num,  # TopK数量
        hidden_dim=hidden_dim,  # 隐藏维度
        routed_scaling_factor=routed_scaling_factor,  # 路由缩放因子
        BLOCK_M=BLOCK_M,  # M维度块大小
        BLOCK_DIM=BLOCK_DIM,  # 维度块大小
        NUM_STAGE=NUM_STAGE,  # 流水线阶段数
        num_warps=num_warps,  # warp数量
    )
    return  # 返回


@triton.jit  # Triton JIT编译装饰器
def _fused_append_shared_experts_kernel(  # 融合追加共享专家内核
    topk_ids_ptr,  # TopK专家ID指针
    topk_weights_ptr,  # TopK权重指针
    out_ids_ptr,  # 输出专家ID指针
    out_weights_ptr,  # 输出权重指针
    N_BASE,  # runtime scalar # 运行时标量，共享专家起始ID
    scale_factor,  # runtime scalar # 运行时标量，缩放因子
    K: tl.constexpr,  # TopK数量（编译时常量）
    S: tl.constexpr,  # 共享专家数量（编译时常量）
):  # 将共享专家信息追加到TopK路由结果中
    """
    for m in range(M):
        for n in range(K):
            fused_ids[m, n] = topk_ids[m, n]
            fused_weights[m, n] = topk_weights[m, n]
        for s in range(S):
            fused_ids[m, K + s] = N + s
            fused_weights[m, K + s] = scale_factor

    对于每个token m：
        复制K个TopK专家ID和权重
        追加S个共享专家ID（从N开始）和缩放因子权重
    """
    pid = tl.program_id(0)  # 获取当前程序ID（token索引）

    ids_row_ptr = pid * K  # 当前token的TopK ID行偏移
    w_row_ptr = pid * K  # 当前token的TopK权重行偏移
    out_ids_row_ptr = pid * (K + S)  # 当前token的输出ID行偏移
    out_w_row_ptr = pid * (K + S)  # 当前token的输出权重行偏移

    offs_k = tl.arange(0, K)  # TopK维度偏移
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k)  # 加载TopK专家ID
    ws = tl.load(topk_weights_ptr + w_row_ptr + offs_k)  # 加载TopK权重

    tl.store(out_ids_ptr + out_ids_row_ptr + offs_k, ids)  # 存储TopK专家ID
    tl.store(out_weights_ptr + out_w_row_ptr + offs_k, ws)  # 存储TopK权重

    offs_s = tl.arange(0, S)  # 共享专家维度偏移

    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)  # 生成共享专家ID（从N_BASE开始）
    shared_ws = tl.full([S], scale_factor, dtype=ws.dtype)  # 生成共享专家权重（统一缩放因子）

    tl.store(out_ids_ptr + out_ids_row_ptr + K + offs_s, shared_ids)  # 存储共享专家ID
    tl.store(out_weights_ptr + out_w_row_ptr + K + offs_s, shared_ws)  # 存储共享专家权重


def fused_append_shared_experts(  # 融合追加共享专家
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N=None  # TopK ID、权重、共享专家数、缩放因子、起始ID
):  # 将共享专家追加到TopK路由结果
    """将共享专家信息追加到TopK路由结果中，共享专家使用统一缩放因子"""
    assert N is not None, "N (shared expert base id) must be provided"  # 断言：必须提供N
    m, k = topk_ids.shape  # 获取token数和TopK数
    s = int(num_fused_shared_experts)  # 获取共享专家数量
    if s <= 0:  # 如果没有共享专家
        return topk_ids, topk_weights  # 直接返回原始结果

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)  # 分配输出ID张量
    out_weights = torch.empty(  # 分配输出权重张量
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device  # 形状为(m, k+s)
    )

    _fused_append_shared_experts_kernel[(m,)](  # 调用内核
        topk_ids,  # TopK专家ID
        topk_weights,  # TopK权重
        out_ids,  # 输出ID
        out_weights,  # 输出权重
        N_BASE=N,  # 共享专家起始ID
        scale_factor=scale_factor,  # 缩放因子
        K=k,  # TopK数量
        S=s,  # 共享专家数量
        num_warps=1,  # warp数量
    )
    return out_ids, out_weights  # 返回合并后的结果


@triton.jit  # Triton JIT编译装饰器
def _fused_append_shared_experts_with_weights_kernel(  # 带权重的融合追加共享专家内核
    topk_ids_ptr,  # TopK专家ID指针
    topk_weights_ptr,  # TopK权重指针
    shared_weights_ptr,  # 共享专家权重指针（逐token不同）
    out_ids_ptr,  # 输出专家ID指针
    out_weights_ptr,  # 输出权重指针
    N_BASE,  # 共享专家起始ID
    K: tl.constexpr,  # TopK数量（编译时常量）
    S: tl.constexpr,  # 共享专家数量（编译时常量）
    BLOCK_K: tl.constexpr,  # K维度块大小（编译时常量）
    BLOCK_S: tl.constexpr,  # S维度块大小（编译时常量）
):  # 将共享专家信息（带逐token权重）追加到TopK路由结果
    """带逐token权重的融合追加共享专家内核"""
    pid = tl.program_id(0)  # 获取当前程序ID（token索引）

    ids_row_ptr = pid * K  # 当前token的TopK ID行偏移
    out_row_ptr = pid * (K + S)  # 当前token的输出行偏移

    offs_k = tl.arange(0, BLOCK_K)  # K维度偏移
    mask_k = offs_k < K  # K维度掩码
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k, mask=mask_k)  # 加载TopK专家ID
    ws = tl.load(topk_weights_ptr + ids_row_ptr + offs_k, mask=mask_k)  # 加载TopK权重

    tl.store(out_ids_ptr + out_row_ptr + offs_k, ids, mask=mask_k)  # 存储TopK专家ID
    tl.store(out_weights_ptr + out_row_ptr + offs_k, ws, mask=mask_k)  # 存储TopK权重

    offs_s = tl.arange(0, BLOCK_S)  # S维度偏移
    mask_s = offs_s < S  # S维度掩码
    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)  # 生成共享专家ID
    shared_ws = tl.load(shared_weights_ptr + pid * S + offs_s, mask=mask_s)  # 加载逐token共享专家权重

    tl.store(out_ids_ptr + out_row_ptr + K + offs_s, shared_ids, mask=mask_s)  # 存储共享专家ID
    tl.store(out_weights_ptr + out_row_ptr + K + offs_s, shared_ws, mask=mask_s)  # 存储共享专家权重


def fused_append_shared_experts_with_weights(  # 带逐token权重的融合追加共享专家
    topk_ids, topk_weights, shared_weights, num_fused_shared_experts, N=None  # TopK ID、权重、共享权重、共享专家数、起始ID
):  # 将共享专家信息（带逐token权重）追加到TopK路由结果
    """Like fused_append_shared_experts but accepts per-token shared weights tensor."""  # 类似fused_append_shared_experts，但接受逐token共享权重张量。
    """类似fused_append_shared_experts，但接受逐token共享权重张量。"""
    assert N is not None, "N (shared expert base id) must be provided"  # 断言：必须提供N
    m, k = topk_ids.shape  # 获取token数和TopK数
    s = int(num_fused_shared_experts)  # 获取共享专家数量
    if s <= 0:  # 如果没有共享专家
        return topk_ids, topk_weights  # 直接返回原始结果

    shared_weights_2d = shared_weights.to(topk_weights.dtype)  # 转换共享权重数据类型
    if shared_weights_2d.ndim == 1:  # 如果是1维张量
        shared_weights_2d = shared_weights_2d.unsqueeze(-1)  # 增加一个维度
    if shared_weights_2d.shape[1] < s:  # 如果共享权重列数少于共享专家数
        shared_weights_2d = shared_weights_2d.expand(m, s)  # 扩展到s列
    shared_weights_2d = shared_weights_2d.contiguous()  # 确保连续存储

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)  # 分配输出ID张量
    out_weights = torch.empty(  # 分配输出权重张量
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device  # 形状为(m, k+s)
    )

    block_k = triton.next_power_of_2(k)  # 获取K的下一个2的幂
    block_s = triton.next_power_of_2(s)  # 获取S的下一个2的幂

    _fused_append_shared_experts_with_weights_kernel[(m,)](  # 调用内核
        topk_ids,  # TopK专家ID
        topk_weights,  # TopK权重
        shared_weights_2d,  # 逐token共享专家权重
        out_ids,  # 输出ID
        out_weights,  # 输出权重
        N_BASE=N,  # 共享专家起始ID
        K=k,  # TopK数量
        S=s,  # 共享专家数量
        BLOCK_K=block_k,  # K维度块大小
        BLOCK_S=block_s,  # S维度块大小
        num_warps=1,  # warp数量
    )
    return out_ids, out_weights  # 返回合并后的结果
