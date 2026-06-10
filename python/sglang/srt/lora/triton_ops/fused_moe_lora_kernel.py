# 融合MoE（混合专家）LoRA内核的Triton实现
# 实现MoE模型中LoRA适配器的收缩和扩展计算，支持多专家多适配器的并行处理

# Temporarily adapted from https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/fused_moe_lora_op.py, will optimize in future refactor
# 临时适配自vLLM项目的融合MoE LoRA操作，未来重构时将进行优化

import torch  # 导入PyTorch张量库
import triton  # 导入Triton编译框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.distributed import (  # 导入分布式通信操作
    tensor_model_parallel_all_gather,  # 张量模型并行全收集操作
    tensor_model_parallel_all_reduce,  # 张量模型并行全规约操作
)
from sglang.srt.utils.common import is_blackwell_supported, is_sm90_supported  # 导入GPU架构检测函数

# Import SGLang's standard PDL support detection
# 导入SGLang的标准PDL支持检测


_LORA_PTR_DICT: dict[tuple[int, ...], torch.Tensor] = {}  # LoRA权重指针缓存字典，用于避免重复创建指针张量


def _get_ptr(lora_weights: list[torch.Tensor], device: torch.device):  # 获取LoRA权重的指针张量
    """
    `_LORA_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    `_LORA_PTR_DICT`在`profile_run`期间收集所需信息，
    之后保持不变，后续使用通过查找表（LUT）进行。
    Refer to:
    参考：
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)  # 使用每个权重张量的数据指针作为缓存键

    if (ptr_tensor := _LORA_PTR_DICT.get(key)) is not None:  # 如果缓存中已存在该键
        return ptr_tensor  # 直接返回缓存的指针张量

    tensor_ptrs = []  # 初始化指针列表
    for lora_weight in lora_weights:  # 遍历所有LoRA权重
        tensor_ptrs.append(lora_weight.data_ptr())  # 收集每个权重张量的数据指针
    ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)  # 创建指针张量，64位无符号整数类型

    _LORA_PTR_DICT[key] = ptr_tensor  # 将指针张量存入缓存
    return _LORA_PTR_DICT.get(key)  # 返回缓存的指针张量


@triton.jit(  # Triton JIT编译装饰器
    do_not_specialize=[  # 不进行特化的参数列表
        "num_valid_tokens",  # 有效token数量
        "EM",  # 扩展的M维度
        "stride_tl",  # sorted_token_ids的步长
        "stride_el",  # expert_ids的步长
        "slice_a_size",  # 输入A切片大小
        "slice_c_size",  # 输出C切片大小
    ]
)
def _fused_moe_lora_kernel(  # 融合MoE LoRA内核函数
    a_ptr,  # 输入矩阵A的指针
    b_ptr,  # 权重矩阵B的指针（LoRA权重）
    c_ptr,  # 输出矩阵C的指针
    topk_weights_ptr,  # top-k路由权重的指针
    sorted_token_ids_ptr,  # 排序后token ID的指针
    expert_ids_ptr,  # 专家ID的指针
    num_tokens_post_padded_ptr,  # 填充后token数量的指针
    # Matrix dimensions  # 矩阵维度
    N,  # 输出维度N
    K,  # 输入维度K
    EM,  # 扩展的M维度（token数 * top_k）
    num_valid_tokens,  # 有效token数量
    num_experts,  # 专家数量
    lora_ids,  # LoRA适配器ID
    adapter_enabled,  # 适配器启用状态
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    # 步长变量表示在特定维度上移动1个元素时指针的增量。例如`stride_am`是
    # `a_ptr`向下移动一行时的增量（A有M行）。
    stride_am,  # A矩阵M维步长
    stride_ak,  # A矩阵K维步长
    stride_bl,  # B矩阵LoRA维步长
    stride_be,  # B矩阵专家维步长
    stride_bk,  # B矩阵K维步长
    stride_bn,  # B矩阵N维步长
    stride_cm,  # C矩阵M维步长
    stride_cn,  # C矩阵N维步长
    stride_tl,  # sorted_token_ids的步长
    stride_el,  # expert_ids的步长
    slice_a_size,  # 输入A切片大小
    slice_c_size,  # 输出C切片大小
    # Meta-parameters  # 元参数
    num_slice_a: tl.constexpr,  # 输入A的切片数
    num_slice_c: tl.constexpr,  # 输出C的切片数
    top_k: tl.constexpr,  # top-k值
    MUL_ROUTED_WEIGHT: tl.constexpr,  # 是否乘以路由权重的标志
    BLOCK_SIZE_M: tl.constexpr,  # M方向的块大小
    BLOCK_SIZE_N: tl.constexpr,  # N方向的块大小
    BLOCK_SIZE_K: tl.constexpr,  # K方向的块大小
    GROUP_SIZE_M: tl.constexpr,  # M方向的分组大小，用于提高L2缓存命中率
    SPLIT_K: tl.constexpr,  # K维度的分割数，用于并行化
    USE_GDC: tl.constexpr,  # 是否使用GDC（GPU依赖链）的标志
    launch_pdl: tl.constexpr,  # 是否启动PDL（持久驱动程序）的标志
    IS_PRIMARY: tl.constexpr,  # 是否为主内核的标志
):
    pid = tl.program_id(axis=0)  # 获取第0轴的程序ID
    slice_id = tl.program_id(axis=1)  # 获取第1轴的程序ID（切片ID）
    lora_idx = tl.program_id(axis=2)  # 获取第2轴的程序ID（LoRA索引）
    lora_id = tl.load(lora_ids + lora_idx)  # 加载当前LoRA适配器的ID

    if lora_id == -1:  # 如果LoRA ID为-1
        # Early exit for the no-lora case.
        # 无LoRA情况下提前退出。
        return  # 直接返回
    moe_enabled = tl.load(adapter_enabled + lora_id)  # 加载当前LoRA适配器的MoE启用状态
    if moe_enabled == 0:  # 如果MoE LoRA未启用
        # Early exit for the no moe lora case.
        # 无MoE LoRA情况下提前退出。
        return  # 直接返回
    max_loras = tl.num_programs(axis=2)  # 获取LoRA维度上的最大程序数
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)  # 计算K维度的网格数

    # calculate pid_m,pid_n
    # 计算pid_m和pid_n
    pid_sk = pid % SPLIT_K  # 计算K分割索引
    pid_m_n = pid // SPLIT_K  # 计算去除K分割后的程序ID
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)  # M维度上的程序数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N维度上的程序数

    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每组中的程序数
    group_id = pid_m_n // num_pid_in_group  # 当前程序所在的组ID
    first_pid_m = group_id * GROUP_SIZE_M  # 当前组中第一个M维度的程序ID
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 当前组的实际大小
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)  # 计算M维度的程序ID
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m  # 计算N维度的程序ID

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)  # 加载当前LoRA填充后的token数量
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:  # 如果当前块的起始位置超出有效token范围
        return  # 直接返回
    # get the expert_id to process curr shard
    # 获取处理当前分片的专家ID
    ind = lora_id * stride_el + pid_m  # 计算专家索引的位置
    expert_id = tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)  # 加载专家ID，越界时返回-1
    if expert_id == -1:  # 如果专家ID无效
        return  # 直接返回

    # get a_ptr,b_ptr,c_ptr
    # 获取a_ptr、b_ptr、c_ptr指针
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size  # 计算当前输入A的指针
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))  # 加载当前权重B的指针
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size  # 计算当前输出C的指针

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N  # 计算N维度的偏移量
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # 计算K维度的偏移量
    # ================================================================= secure  # 安全边界处理

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)  # 计算token ID的偏移量
    token_ind = stride_tl * lora_id + offs_token_id  # 计算sorted_token_ids中的索引
    offs_token = tl.load(  # 加载排序后的token ID
        sorted_token_ids_ptr + token_ind, token_ind < max_loras * stride_tl, 0  # 使用掩码防止越界
    )
    token_mask = offs_token < num_valid_tokens  # 计算有效token的掩码

    # ================================================================= secure  # 安全边界处理

    # get a_ptrs,b_ptrs
    # 获取a_ptrs和b_ptrs指针
    a_ptrs = cur_a_ptr + (  # 计算输入A的指针位置
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak  # token偏移/top_k*行步长 + K偏移*列步长
    )

    b_ptrs = (  # 计算权重B的指针位置
        cur_b_ptr  # 权重B基址
        + lora_id * stride_bl  # LoRA偏移
        + expert_id * stride_be  # 专家偏移
        + offs_k[:, None] * stride_bk  # K维偏移
        + offs_bn[None, :] * stride_bn  # N维偏移
    )

    if USE_GDC and IS_PRIMARY:  # 如果使用GDC且是主内核
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        # GDC启动依赖提示运行时系统启动依赖内核。
        tl.extra.cuda.gdc_launch_dependents()  # 启动GDC依赖内核

    # ================================================================= secure  # 安全边界处理

    # accumulator
    # 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器为零

    # ================================================================= secure  # 安全边界处理

    # GDC wait waits for ALL programs in the prior kernel to complete
    # before continuing.
    # GDC等待前一个内核的所有程序完成后才继续。
    if USE_GDC and not IS_PRIMARY:  # 如果使用GDC且不是主内核
        tl.extra.cuda.gdc_wait()  # 等待GDC依赖完成

    for k in range(0, grid_k):  # 沿K维度迭代
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)  # 计算剩余K维度大小
        # pre-fetch lora weight
        # 预取LoRA权重
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)  # 加载权重B的一个小块
        a = tl.load(  # 加载输入A的一个小块
            a_ptrs,  # A的指针位置
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),  # token掩码和K维掩码
            other=0.0,  # 越界位置填充0
        )
        accumulator += tl.dot(a, b.to(a.dtype))  # 累加矩阵乘法结果
        # Advance the ptrs to the next K block.
        # 将指针前进到下一个K块。
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak  # 移动A指针到下一个K块
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk  # 移动B指针到下一个K块

    if MUL_ROUTED_WEIGHT:  # 如果需要乘以路由权重
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)  # 加载MoE路由权重
        accumulator = accumulator * moe_weight[:, None]  # 将累加结果乘以路由权重
    accumulator = accumulator.to(c_ptr.dtype.element_ty)  # 将累加结果转换为输出数据类型
    # Write back the block of the output
    # 写回输出块
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算输出N维偏移量
    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  # 计算输出指针位置
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  # 计算输出掩码

    if SPLIT_K == 1:  # 如果K维度未分割
        tl.store(c_ptrs, accumulator, mask=c_mask)  # 直接存储结果
    else:  # K维度已分割
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")  # 使用原子加操作合并分割结果


@torch.inference_mode()  # 推理模式装饰器，禁用梯度计算
def _fused_moe_lora_shrink(  # 融合MoE LoRA收缩函数
    a_intermediate_cache1: torch.Tensor,  # 中间缓存张量，存储收缩结果
    # (num_slices, num_tokens, top_k_num, max_lora_rank)  # 形状：(切片数, token数, top_k数, 最大LoRA秩)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)  # 当前隐藏状态，形状：(token数, 输入维度)
    lora_a_stacked: list[  # LoRA A权重堆叠列表
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]  # 形状：[(最大LoRA数, 专家数, 最大秩, K), ...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)  # top-k路由权重
    sorted_token_ids: torch.Tensor,  # (max_loras, _)  # 排序后的token ID
    expert_ids: torch.Tensor,  # (max_loras, _ ,)  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )  # 填充后的token数量
    top_k_num: int,  # top-k值
    lora_ids: torch.Tensor,  # LoRA适配器ID
    adapter_enabled: torch.Tensor,  # 适配器启用状态
    ## adding for kernel  # 内核所需的额外参数
    device: torch.device,  # 计算设备
    N: int,  # 输出维度N
    M: int,  # token数M
    EM: int,  # 扩展的M维度
    K: int,  # 输入维度K
    num_tokens: int,  # 总token数量
    num_experts: int,  # 专家数量
    num_slices: int,  # 切片数量
    block_size_m: int,  # M方向块大小
    block_size_n: int,  # N方向块大小
    block_size_k: int,  # K方向块大小
    group_size_m: int,  # M方向分组大小
    num_warps: int,  # warp数量
    num_stages: int,  # 流水线阶段数
    split_k: int,  # K维度分割数
    top_k_divisor: int = None,  # top-k除数，默认为None
    mul_routed_weight: bool = False,  # 是否乘以路由权重，默认为False
) -> None:  # 无返回值
    w1_lora_a_stacked = lora_a_stacked[0]  # 获取第一个切片的LoRA A权重

    use_gdc = is_sm90_supported() or is_blackwell_supported()  # 检测是否支持GDC（SM90或Blackwell架构）
    shrink_config = {  # 收缩操作的配置字典
        "BLOCK_SIZE_M": block_size_m,  # M方向块大小
        "BLOCK_SIZE_N": block_size_n,  # N方向块大小
        "BLOCK_SIZE_K": block_size_k,  # K方向块大小
        "GROUP_SIZE_M": group_size_m,  # M方向分组大小
        "num_warps": num_warps,  # warp数量
        "num_stages": num_stages,  # 流水线阶段数
        "SPLIT_K": split_k,  # K维度分割数
        "USE_GDC": use_gdc,  # 是否使用GDC
        "launch_pdl": use_gdc,  # triton kernel metadata  # Triton内核元数据，是否启动PDL
    }

    b_ptr = _get_ptr(lora_a_stacked, device)  # 获取LoRA A权重的指针张量

    grid = lambda META: (  # 动态计算内核启动网格的lambda函数
        split_k  # K分割数
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])  # M维度块数
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),  # N维度块数
        len(lora_a_stacked),  # 切片数
        lora_a_stacked[0].shape[0],  # LoRA适配器数量
    )
    _fused_moe_lora_kernel[grid](  # 启动融合MoE LoRA内核
        qcurr_hidden_states,  # 输入隐藏状态
        b_ptr,  # LoRA权重指针
        a_intermediate_cache1,  # 中间缓存（输出）
        topk_weights,  # top-k路由权重
        sorted_token_ids,  # 排序后的token ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后的token数量
        N,  # 输出维度N
        K,  # 输入维度K
        EM,  # 扩展的M维度
        num_tokens,  # 总token数量
        num_experts,  # 专家数量
        lora_ids,  # LoRA适配器ID
        adapter_enabled,  # 适配器启用状态
        qcurr_hidden_states.stride(0),  # 输入第0维步长
        qcurr_hidden_states.stride(1),  # 输入第1维步长
        w1_lora_a_stacked.stride(0),  # 权重A第0维步长（LoRA维度）
        w1_lora_a_stacked.stride(1),  # 权重A第1维步长（专家维度）
        w1_lora_a_stacked.stride(3),  # 权重A第3维步长（K维度）
        w1_lora_a_stacked.stride(2),  # 权重A第2维步长（秩维度）
        a_intermediate_cache1.stride(2),  # 中间缓存第2维步长
        a_intermediate_cache1.stride(3),  # 中间缓存第3维步长
        sorted_token_ids.stride(0),  # sorted_token_ids第0维步长
        expert_ids.stride(0),  # expert_ids第0维步长
        slice_a_size=qcurr_hidden_states.numel(),  # 输入切片大小
        slice_c_size=a_intermediate_cache1.numel() // num_slices,  # 输出切片大小
        num_slice_a=1,  # 输入切片数为1
        num_slice_c=num_slices,  # 输出切片数
        top_k=(  # top_k参数
            top_k_divisor  # 如果指定了除数
            if top_k_divisor is not None  # 使用除数
            else (1 if mul_routed_weight else top_k_num)  # 否则：乘路由权重时为1，否则为top_k_num
        ),
        MUL_ROUTED_WEIGHT=False,  # 收缩阶段不乘路由权重
        IS_PRIMARY=True,  # 收缩阶段是主内核
        **shrink_config,  # 展开收缩配置
    )


@torch.inference_mode()  # 推理模式装饰器，禁用梯度计算
def _fused_moe_lora_expand(  # 融合MoE LoRA扩展函数
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)  # 最终输出张量
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)  # 收缩阶段的中间缓存
    b_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, output_dim_size)  # 扩展阶段的中间缓存
    lora_b_stacked: list[  # LoRA B权重堆叠列表
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]  # 形状说明
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)  # top-k路由权重
    sorted_token_ids: torch.Tensor,  # (max_loras, _)  # 排序后的token ID
    expert_ids: torch.Tensor,  # (max_loras, _ ,)  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )  # 填充后的token数量
    top_k_num: int,  # top-k值
    lora_ids: torch.Tensor,  # LoRA适配器ID
    adapter_enabled: torch.Tensor,  # 适配器启用状态
    ## adding for kernel  # 内核所需的额外参数
    device: torch.device,  # 计算设备
    N: int,  # 输出维度N
    M: int,  # token数M
    EM: int,  # 扩展的M维度
    K: int,  # 输入维度K
    num_tokens: int,  # 总token数量
    num_experts: int,  # 专家数量
    num_slices: int,  # 切片数量
    max_lora_rank: int,  # 最大LoRA秩
    w1_output_dim_size: int,  # 输出维度大小
    block_size_m: int,  # M方向块大小
    block_size_n: int,  # N方向块大小
    block_size_k: int,  # K方向块大小
    group_size_m: int,  # M方向分组大小
    num_warps: int,  # warp数量
    num_stages: int,  # 流水线阶段数
    split_k: int,  # K维度分割数
    mul_routed_weight: bool = False,  # 是否乘以路由权重
    offset: int = 0,  # 输出偏移量
) -> None:  # 无返回值

    b_ptr = _get_ptr(lora_b_stacked, device)  # 获取LoRA B权重的指针张量
    K = max_lora_rank  # K维度设为最大LoRA秩（扩展阶段K是秩）
    N = w1_output_dim_size  # N维度设为输出维度大小

    w1_lora_b_stacked = lora_b_stacked[0]  # 获取第一个切片的LoRA B权重

    a_intermediate_cache1 = a_intermediate_cache1.view(  # 重塑中间缓存的形状
        -1, a_intermediate_cache1.shape[3]  # 合并前三维为行，保留秩维度为列
    )

    use_gdc = is_sm90_supported() or is_blackwell_supported()  # 检测是否支持GDC
    expand_config = {  # 扩展操作的配置字典
        "BLOCK_SIZE_M": block_size_m,  # M方向块大小
        "BLOCK_SIZE_N": block_size_n,  # N方向块大小
        "BLOCK_SIZE_K": block_size_k,  # K方向块大小
        "GROUP_SIZE_M": group_size_m,  # M方向分组大小
        "num_warps": num_warps,  # warp数量
        "num_stages": num_stages,  # 流水线阶段数
        "SPLIT_K": split_k,  # Set split_k = 1 for expand calls  # 扩展调用时设置split_k=1
        "USE_GDC": use_gdc,  # 是否使用GDC
        "launch_pdl": use_gdc,  # triton kernel metadata  # Triton内核元数据
    }

    grid = lambda META: (  # 动态计算内核启动网格的lambda函数
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),  # M和N维度块数的乘积
        len(lora_b_stacked),  # 切片数
        lora_b_stacked[0].shape[0],  # LoRA适配器数量
    )
    _fused_moe_lora_kernel[grid](  # 启动融合MoE LoRA内核
        a_intermediate_cache1,  # 输入（收缩结果）
        b_ptr,  # LoRA B权重指针
        b_intermediate_cache1,  # 中间缓存（扩展输出）
        topk_weights,  # top-k路由权重
        sorted_token_ids,  # 排序后的token ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后的token数量
        N,  # 输出维度N
        K,  # 输入维度K（即LoRA秩）
        EM,  # 扩展的M维度
        num_tokens,  # 总token数量
        num_experts,  # 专家数量
        lora_ids,  # LoRA适配器ID
        adapter_enabled,  # 适配器启用状态
        a_intermediate_cache1.stride(0),  # 输入第0维步长
        a_intermediate_cache1.stride(1),  # 输入第1维步长
        w1_lora_b_stacked.stride(0),  # 权重B第0维步长（LoRA维度）
        w1_lora_b_stacked.stride(1),  # 权重B第1维步长（专家维度）
        w1_lora_b_stacked.stride(3),  # 权重B第3维步长（秩维度）
        w1_lora_b_stacked.stride(2),  # 权重B第2维步长（N维度）
        b_intermediate_cache1.stride(2),  # 中间缓存第2维步长
        b_intermediate_cache1.stride(3),  # 中间缓存第3维步长
        sorted_token_ids.stride(0),  # sorted_token_ids第0维步长
        expert_ids.stride(0),  # expert_ids第0维步长
        slice_a_size=a_intermediate_cache1.numel() // num_slices,  # 输入切片大小
        slice_c_size=b_intermediate_cache1.numel() // num_slices,  # 输出切片大小
        num_slice_a=num_slices,  # 输入切片数
        num_slice_c=num_slices,  # 输出切片数
        top_k=1,  # 扩展阶段top_k固定为1
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # 是否乘以路由权重
        IS_PRIMARY=False,  # 扩展阶段不是主内核
        **expand_config,  # 展开扩展配置
    )
    for i in range(num_slices):  # 遍历每个切片
        output[:, :, i * N + offset : (i + 1) * N + offset] += b_intermediate_cache1[i]  # 将各切片的扩展结果累加到输出


@torch.inference_mode()  # 推理模式装饰器，禁用梯度计算
def _fused_moe_lora(  # 融合MoE LoRA主函数，组合收缩和扩展操作
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)  # 最终输出张量
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)  # 当前隐藏状态
    lora_a_stacked: list[  # LoRA A权重堆叠列表
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]  # 形状说明
    lora_b_stacked: list[  # LoRA B权重堆叠列表
        torch.Tensor
    ],  # [(max_loras, num_experts, N, max_lora_rank,),...]  # 形状说明
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)  # top-k路由权重
    sorted_token_ids: torch.Tensor,  # (max_loras, _)  # 排序后的token ID
    expert_ids: torch.Tensor,  # (max_loras, _ ,)  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )  # 填充后的token数量
    max_lora_rank: int,  # 最大LoRA秩
    top_k_num: int,  # top-k值
    lora_ids: torch.Tensor,  # LoRA适配器ID
    adapter_enabled: torch.Tensor,  # 适配器启用状态
    shrink_block_size_m: int,  # 收缩M方向块大小
    shrink_block_size_n: int,  # 收缩N方向块大小
    shrink_block_size_k: int,  # 收缩K方向块大小
    shrink_group_size_m: int,  # 收缩M方向分组大小
    shrink_num_warps: int,  # 收缩warp数量
    shrink_num_stages: int,  # 收缩流水线阶段数
    shrink_split_k: int,  # 收缩K维度分割数
    expand_block_size_m: int,  # 扩展M方向块大小
    expand_block_size_n: int,  # 扩展N方向块大小
    expand_block_size_k: int,  # 扩展K方向块大小
    expand_group_size_m: int,  # 扩展M方向分组大小
    expand_num_warps: int,  # 扩展warp数量
    expand_num_stages: int,  # 扩展流水线阶段数
    expand_split_k: int,  # 扩展K维度分割数
    mul_routed_weight: bool = False,  # 是否乘以路由权重
    fully_sharded: bool = False,  # 是否完全分片
    offset: int = 0,  # 输出偏移量
) -> None:  # 无返回值
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0  # 断言A和B权重列表长度相同且非空
    assert (  # 断言各张量维度一致性
        sorted_token_ids.dim()  # sorted_token_ids维度
        == expert_ids.dim()  # 等于expert_ids维度
        == topk_weights.dim()  # 等于topk_weights维度
        == qcurr_hidden_states.dim()  # 等于qcurr_hidden_states维度
        == 2  # 均为2维
    )
    assert (  # 断言第0维大小一致
        sorted_token_ids.shape[0]  # sorted_token_ids第0维
        == expert_ids.shape[0]  # 等于expert_ids第0维
        == num_tokens_post_padded.shape[0]  # 等于num_tokens_post_padded第0维
    )
    assert output.shape[0] == topk_weights.shape[0]  # 断言输出和权重的token维度一致
    assert top_k_num == topk_weights.shape[1]  # 断言top_k值与权重第1维一致
    device = qcurr_hidden_states.device  # 获取计算设备
    num_slices = len(lora_a_stacked)  # 切片数量等于A权重列表长度
    w1_lora_b_stacked = lora_b_stacked[0]  # 获取第一个切片的LoRA B权重
    num_experts = lora_a_stacked[0].shape[1]  # 从A权重获取专家数量
    N = max_lora_rank  # 收缩阶段的N维度等于最大LoRA秩
    M = topk_weights.shape[0]  # M维度等于token数量
    EM = sorted_token_ids.shape[1]  # 扩展M维度等于sorted_token_ids第1维
    K = qcurr_hidden_states.shape[1]  # K维度等于隐藏状态维度
    num_tokens = M * top_k_num  # 总token数等于M乘以top_k
    w1_output_dim_size = w1_lora_b_stacked.shape[2]  # 扩展阶段的输出维度大小

    # Detect whether input is already expanded (down path: [M*top_k, dim])
    # or not (gate_up path: [M, dim]). Down path needs divisor=1.
    # 检测输入是否已经扩展（下投影路径：[M*top_k, dim]）还是未扩展
    # （gate_up路径：[M, dim]）。下投影路径需要divisor=1。
    input_is_expanded = qcurr_hidden_states.shape[0] == M * top_k_num  # 判断输入是否已扩展
    shrink_top_k_divisor = 1 if input_is_expanded else top_k_num  # 计算收缩阶段的top_k除数

    a_intermediate_cache1 = torch.zeros(  # 创建收缩阶段的中间缓存
        (num_slices, M, top_k_num, max_lora_rank),  # 形状：(切片数, token数, top_k, 最大秩)
        dtype=output.dtype,  # 与输出相同的数据类型
        device=device,  # 与输入相同的设备
    )

    b_intermediate_cache1 = torch.zeros(  # 创建扩展阶段的中间缓存
        (num_slices, M, top_k_num, w1_output_dim_size),  # 形状：(切片数, token数, top_k, 输出维度)
        dtype=output.dtype,  # 与输出相同的数据类型
        device=device,  # 与输入相同的设备
    )

    _fused_moe_lora_shrink(  # 执行收缩操作
        a_intermediate_cache1,  # 中间缓存（输出）
        qcurr_hidden_states,  # 输入隐藏状态
        lora_a_stacked,  # LoRA A权重
        topk_weights,  # top-k路由权重
        sorted_token_ids,  # 排序后的token ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后的token数量
        top_k_num,  # top-k值
        lora_ids,  # LoRA适配器ID
        adapter_enabled,  # 适配器启用状态
        ## adding for kernel  # 内核所需的额外参数
        device,  # 计算设备
        N,  # 输出维度N
        M,  # token数M
        EM,  # 扩展的M维度
        K,  # 输入维度K
        num_tokens,  # 总token数量
        num_experts,  # 专家数量
        num_slices,  # 切片数量
        shrink_block_size_m,  # 收缩M方向块大小
        shrink_block_size_n,  # 收缩N方向块大小
        shrink_block_size_k,  # 收缩K方向块大小
        shrink_group_size_m,  # 收缩M方向分组大小
        shrink_num_warps,  # 收缩warp数量
        shrink_num_stages,  # 收缩流水线阶段数
        shrink_split_k,  # 收缩K维度分割数
        top_k_divisor=shrink_top_k_divisor,  # 收缩阶段的top_k除数
        mul_routed_weight=False,  # 收缩阶段不乘路由权重
    )

    if fully_sharded:  # 如果完全分片模式
        if max_lora_rank == w1_lora_b_stacked.shape[-1]:  # 如果当前秩等于完整秩
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(  # 对收缩结果进行全规约
                a_intermediate_cache1  # 中间缓存
            )
        else:  # 当前秩不等于完整秩
            a_intermediate_cache1 = tensor_model_parallel_all_gather(  # 对收缩结果进行全收集
                a_intermediate_cache1  # 中间缓存
            )

            # reset max_lora_rank to the full rank after allgather
            # allgather后将max_lora_rank重置为完整秩
            max_lora_rank = a_intermediate_cache1.shape[-1]  # 更新最大LoRA秩

    _fused_moe_lora_expand(  # 执行扩展操作
        output,  # 最终输出张量
        a_intermediate_cache1,  # 收缩阶段的中间缓存（输入）
        b_intermediate_cache1,  # 扩展阶段的中间缓存（输出）
        lora_b_stacked,  # LoRA B权重
        topk_weights,  # top-k路由权重
        sorted_token_ids,  # 排序后的token ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后的token数量
        top_k_num,  # top-k值
        lora_ids,  # LoRA适配器ID
        adapter_enabled,  # 适配器启用状态
        ## adding for kernel  # 内核所需的额外参数
        device,  # 计算设备
        N,  # 输出维度N
        M,  # token数M
        EM,  # 扩展的M维度
        K,  # 输入维度K
        num_tokens,  # 总token数量
        num_experts,  # 专家数量
        num_slices,  # 切片数量
        max_lora_rank,  # 最大LoRA秩
        w1_output_dim_size,  # 输出维度大小
        expand_block_size_m,  # 扩展M方向块大小
        expand_block_size_n,  # 扩展N方向块大小
        expand_block_size_k,  # 扩展K方向块大小
        expand_group_size_m,  # 扩展M方向分组大小
        expand_num_warps,  # 扩展warp数量
        expand_num_stages,  # 扩展流水线阶段数
        expand_split_k,  # 扩展K维度分割数
        mul_routed_weight,  # 是否乘以路由权重
        offset,  # 输出偏移量
    )


def _fused_moe_lora_fake(  # 融合MoE LoRA的假实现，用于torch.compile追踪
    output: torch.Tensor,  # 输出张量
    qcurr_hidden_states: torch.Tensor,  # 当前隐藏状态
    lora_a_stacked: list[torch.Tensor],  # LoRA A权重堆叠列表
    lora_b_stacked: list[torch.Tensor],  # LoRA B权重堆叠列表
    topk_weights: torch.Tensor,  # top-k路由权重
    sorted_token_ids: torch.Tensor,  # 排序后的token ID
    expert_ids: torch.Tensor,  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # 填充后的token数量
    max_lora_rank: int,  # 最大LoRA秩
    top_k_num: int,  # top-k值
    lora_ids: torch.Tensor,  # LoRA适配器ID
    adapter_enabled: torch.Tensor,  # 适配器启用状态
    shrink_block_size_m: int,  # 收缩M方向块大小
    shrink_block_size_n: int,  # 收缩N方向块大小
    shrink_block_size_k: int,  # 收缩K方向块大小
    shrink_group_size_m: int,  # 收缩M方向分组大小
    shrink_num_warps: int,  # 收缩warp数量
    shrink_num_stages: int,  # 收缩流水线阶段数
    shrink_split_k: int,  # 收缩K维度分割数
    expand_block_size_m: int,  # 扩展M方向块大小
    expand_block_size_n: int,  # 扩展N方向块大小
    expand_block_size_k: int,  # 扩展K方向块大小
    expand_group_size_m: int,  # 扩展M方向分组大小
    expand_num_warps: int,  # 扩展warp数量
    expand_num_stages: int,  # 扩展流水线阶段数
    expand_split_k: int,  # 扩展K维度分割数
    mul_routed_weight: bool = False,  # 是否乘以路由权重
    fully_sharded: bool = False,  # 是否完全分片
    offset: int = 0,  # 输出偏移量
) -> None:  # 无返回值
    return  # 假实现直接返回


def _fused_moe_lora_shrink_fake(  # 融合MoE LoRA收缩的假实现
    a_intermediate_cache1: torch.Tensor,  # 中间缓存张量
    qcurr_hidden_states: torch.Tensor,  # 当前隐藏状态
    lora_a_stacked: list[torch.Tensor],  # LoRA A权重堆叠列表
    topk_weights: torch.Tensor,  # top-k路由权重
    sorted_token_ids: torch.Tensor,  # 排序后的token ID
    expert_ids: torch.Tensor,  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # 填充后的token数量
    top_k_num: int,  # top-k值
    lora_ids: torch.Tensor,  # LoRA适配器ID
    adapter_enabled: torch.Tensor,  # 适配器启用状态
    device: torch.device,  # 计算设备
    N: int,  # 输出维度N
    M: int,  # token数M
    EM: int,  # 扩展的M维度
    K: int,  # 输入维度K
    num_tokens: int,  # 总token数量
    num_experts: int,  # 专家数量
    num_slices: int,  # 切片数量
    block_size_m: int,  # M方向块大小
    block_size_n: int,  # N方向块大小
    block_size_k: int,  # K方向块大小
    group_size_m: int,  # M方向分组大小
    num_warps: int,  # warp数量
    num_stages: int,  # 流水线阶段数
    split_k: int,  # K维度分割数
    mul_routed_weight: bool = False,  # 是否乘以路由权重
) -> None:  # 无返回值
    return  # 假实现直接返回


def _fused_moe_lora_expand_fake(  # 融合MoE LoRA扩展的假实现
    output: torch.Tensor,  # 输出张量
    a_intermediate_cache1: torch.Tensor,  # 中间缓存张量
    b_intermediate_cache1: torch.Tensor,  # 扩展阶段中间缓存
    lora_b_stacked: list[torch.Tensor],  # LoRA B权重堆叠列表
    topk_weights: torch.Tensor,  # top-k路由权重
    sorted_token_ids: torch.Tensor,  # 排序后的token ID
    expert_ids: torch.Tensor,  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # 填充后的token数量
    top_k_num: int,  # top-k值
    lora_ids: torch.Tensor,  # LoRA适配器ID
    adapter_enabled: torch.Tensor,  # 适配器启用状态
    device: torch.device,  # 计算设备
    N: int,  # 输出维度N
    M: int,  # token数M
    EM: int,  # 扩展的M维度
    K: int,  # 输入维度K
    num_tokens: int,  # 总token数量
    num_experts: int,  # 专家数量
    num_slices: int,  # 切片数量
    max_lora_rank: int,  # 最大LoRA秩
    w1_output_dim_size: int,  # 输出维度大小
    block_size_m: int,  # M方向块大小
    block_size_n: int,  # N方向块大小
    block_size_k: int,  # K方向块大小
    group_size_m: int,  # M方向分组大小
    num_warps: int,  # warp数量
    num_stages: int,  # 流水线阶段数
    split_k: int,  # K维度分割数
    mul_routed_weight: bool = False,  # 是否乘以路由权重
    offset: int = 0,  # 输出偏移量
) -> None:  # 无返回值
    return  # 假实现直接返回


# Register as SGLang custom ops following the same pattern as other ops
# 按照其他操作的模式注册为SGLang自定义操作
try:  # 尝试注册自定义操作
    from sglang.srt.utils.common import direct_register_custom_op  # 导入直接注册自定义操作的函数

    direct_register_custom_op(  # 注册融合MoE LoRA操作
        op_name="fused_moe_lora",  # 操作名称
        op_func=_fused_moe_lora,  # 操作实现函数
        mutates_args=["output"],  # 可变参数列表
        fake_impl=_fused_moe_lora_fake,  # 假实现函数
    )

    direct_register_custom_op(  # 注册融合MoE LoRA收缩操作
        op_name="fused_moe_lora_shrink",  # 操作名称
        op_func=_fused_moe_lora_shrink,  # 操作实现函数
        mutates_args=["a_intermediate_cache1"],  # 可变参数列表
        fake_impl=_fused_moe_lora_shrink_fake,  # 假实现函数
    )

    direct_register_custom_op(  # 注册融合MoE LoRA扩展操作
        op_name="fused_moe_lora_expand",  # 操作名称
        op_func=_fused_moe_lora_expand,  # 操作实现函数
        mutates_args=["output", "b_intermediate_cache1"],  # 可变参数列表
        fake_impl=_fused_moe_lora_expand_fake,  # 假实现函数
    )

    # Export through torch.ops.sglang namespace
    # 通过torch.ops.sglang命名空间导出
    fused_moe_lora = torch.ops.sglang.fused_moe_lora  # 导出融合MoE LoRA操作
    fused_moe_lora_shrink = torch.ops.sglang.fused_moe_lora_shrink  # 导出融合MoE LoRA收缩操作
    fused_moe_lora_expand = torch.ops.sglang.fused_moe_lora_expand  # 导出融合MoE LoRA扩展操作

except AttributeError:  # 如果注册失败（旧版本PyTorch不支持自定义操作）
    fused_moe_lora = _fused_moe_lora  # 直接使用原始函数
    fused_moe_lora_shrink = _fused_moe_lora_shrink  # 直接使用原始函数
    fused_moe_lora_expand = _fused_moe_lora_expand  # 直接使用原始函数
