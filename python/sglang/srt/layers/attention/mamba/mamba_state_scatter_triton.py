# Mamba状态散射Triton融合核 - 替代update_mamba_state_after_mtp_verify中昂贵的
# 高级索引操作，使用单一融合的gather-scatter核，避免多次index_elementwise_kernel启动
"""
Fused Triton kernel for Mamba state scatter operations. # Mamba状态散射操作的融合Triton核。

This kernel replaces the expensive advanced indexing operations in # 此核替代了
`update_mamba_state_after_mtp_verify` with a single fused gather-scatter kernel, # `update_mamba_state_after_mtp_verify`中昂贵的先进索引操作，使用单一融合gather-scatter核，
avoiding multiple `index_elementwise_kernel` launches. # 避免多次`index_elementwise_kernel`启动。
"""

import torch  # 导入PyTorch深度学习框架 # 导入PyTorch
import triton  # 导入Triton GPU编程框架 # 导入Triton编译器
import triton.language as tl  # 导入Triton语言并简写为tl # 导入Triton语言API


@triton.jit  # Triton JIT编译装饰器 # 使用Triton即时编译装饰器
def _fused_mamba_state_scatter_with_mask_kernel(  # 融合的Mamba状态散射带掩码核函数 - 内建掩码的gather-scatter操作
    src_ptr,  # 源数据指针
    dst_ptr,  # 目标数据指针
    # Raw index arrays (before index_select) # 原始索引数组（index_select之前）
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor # 目标索引原始指针，形状为[total_requests]
    step_indices_raw_ptr,  # [total_requests] - last_correct_step_indices or mamba_steps_to_track # 步骤索引原始指针，形状为[total_requests]
    elem_per_entry: tl.constexpr,  # 每个条目的元素数（编译时常量）
    src_layer_stride,  # 源数据层步长
    src_req_stride,  # 源数据请求步长
    src_step_stride,  # 源数据步骤步长
    dst_layer_stride,  # 目标数据层步长
    dst_req_stride,  # 目标数据请求步长
    src_req_size,  # 源数据请求维度大小
    src_step_size,  # 源数据步骤维度大小
    dst_req_size,  # 目标数据请求维度大小
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    """ # 融合gather-scatter核函数文档
    Fused gather-scatter kernel with built-in masking. # 内建掩码的融合gather-scatter核。

    This kernel fuses the index_select operations by: # 此核通过以下方式融合index_select操作：
    1. Iterating over all requests (pid_req from 0 to total_requests-1) # 1. 遍历所有请求（pid_req从0到total_requests-1）
    2. Checking if step_indices_raw[pid_req] >= 0 (valid mask) # 2. 检查step_indices_raw[pid_req] >= 0（有效掩码）
    3. If valid, performing the scatter: # 3. 如果有效，执行散射：
       dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :] # dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :]

    Grid: (total_requests, num_layers, ceil(elem_per_entry / BLOCK_SIZE)) # 网格：(total_requests, num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    """
    pid_req = tl.program_id(0)  # 获取请求维度的程序ID # 当前处理的请求索引
    pid_layer = tl.program_id(1).to(tl.int64)  # 获取层维度的程序ID并转为int64 # 当前处理的层索引
    pid_block = tl.program_id(2).to(tl.int64)  # 获取块维度的程序ID并转为int64 # 当前处理的块索引

    # Load step index to check validity (step >= 0 means valid) # 加载步骤索引以检查有效性（step >= 0表示有效）
    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)  # 加载步骤索引 # 读取当前请求的步骤索引

    # Early exit if this request is not valid (step < 0) # 如果此请求无效则提前退出（step < 0）
    if step_idx < 0:  # 步骤索引小于0表示无效 # 无效请求直接跳过
        return  # 直接返回 # 跳过无效请求

    # Load destination index # 加载目标索引
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)  # 加载目标索引 # 读取当前请求的目标位置

    # Source index is just the request index itself # 源索引就是请求索引本身
    src_idx = pid_req  # 源请求索引 # 直接使用请求索引

    # Bounds check to avoid illegal memory access # 边界检查以避免非法内存访问
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):  # 检查所有索引是否在有效范围内 # 确保不越界
        return  # 直接返回 # 越界则跳过

    # Compute base offsets # 计算基础偏移量
    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )  # 计算源数据偏移 # 根据层、请求和步骤计算源地址偏移
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride  # 计算目标数据偏移 # 根据层和请求计算目标地址偏移

    # Compute element range for this block # 计算此块的元素范围
    start = pid_block * BLOCK_SIZE  # 块起始位置 # 当前块在元素维度上的起始偏移
    offsets = start + tl.arange(0, BLOCK_SIZE)  # 计算元素偏移 # 生成块内所有元素的偏移
    mask = offsets < elem_per_entry  # 计算元素掩码 # 确保不超出每个条目的元素数

    # Load from source and store to destination # 从源加载并存储到目标
    data = tl.load(src_ptr + src_offset + offsets, mask=mask)  # 从源加载数据 # 读取源位置的数据
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)  # 存储到目标 # 将数据写入目标位置


def fused_mamba_state_scatter_with_mask(  # 融合的Mamba状态散射带掩码函数 - 将多个操作融合为单一核调用
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape] # 目标张量，形状为[层数, 缓存大小, *状态形状]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape] # 源张量，形状为[层数, 推测大小, 草稿token数, *状态形状]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw indices (e.g., state_indices_tensor) # 目标索引原始数组，形状为[总请求数]
    step_indices_raw: torch.Tensor,  # [total_requests] - raw step indices (step >= 0 means valid) # 步骤索引原始数组，形状为[总请求数]，>=0表示有效
):
    """ # 融合gather-scatter带掩码的Mamba状态更新文档
    Fully fused gather-scatter with built-in masking for mamba state updates. # 完全融合的带内建掩码的gather-scatter用于Mamba状态更新。

    This function fuses the following operations into a single kernel: # 此函数将以下操作融合为单一核：
    1. valid_mask = step_indices_raw >= 0 # 1. 有效掩码 = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero() # 2. 有效索引 = 有效掩码的非零位置
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select) # 3. 目标索引 = dst_indices_raw[有效索引]（index_select）
    4. step_indices = step_indices_raw[valid_indices]  (index_select) # 4. 步骤索引 = step_indices_raw[有效索引]（index_select）
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :] # 5. 对每个有效i：dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

    Args: # 参数：
        dst: Destination tensor [num_layers, cache_size, *state_shape] # dst：目标张量 [层数, 缓存大小, *状态形状]
        src: Source tensor [num_layers, spec_size, draft_tokens, *state_shape] # src：源张量 [层数, 推测大小, 草稿token数, *状态形状]
        dst_indices_raw: Raw destination indices for all requests [total_requests] # dst_indices_raw：所有请求的原始目标索引 [总请求数]
        step_indices_raw: Raw step indices; entry >= 0 means valid [total_requests] # step_indices_raw：原始步骤索引；条目>=0表示有效 [总请求数]
    """
    total_requests = step_indices_raw.shape[0]  # 获取总请求数 # 读取请求总数
    if total_requests == 0:  # 如果没有请求 # 空请求直接返回
        return  # 直接返回 # 无请求时无需处理

    if dst.device != src.device:  # 检查设备一致性 # 确保源和目标在同一设备上
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )  # 抛出设备不匹配错误
    if not dst.is_cuda or not src.is_cuda:  # 检查是否为CUDA张量 # 确保都是CUDA张量
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA tensors."
        )  # 抛出不支持非CUDA的错误
    if dst.ndim < 2 or src.ndim < 3:  # 检查张量维度 # 确保维度足够
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")  # 抛出维度错误
    if dst.shape[0] != src.shape[0]:  # 检查层数一致 # 确保层维度匹配
        raise ValueError(
            f"Layer dimension mismatch: {dst.shape[0]=} vs {src.shape[0]=}"
        )  # 抛出层数不匹配错误
    if dst.shape[2:] != src.shape[3:]:  # 检查尾部维度一致 # 确保状态形状匹配
        raise ValueError(
            f"Trailing dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}"
        )  # 抛出尾部维度不匹配错误
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:  # 检查索引为1D # 确保索引是一维的
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )  # 抛出索引维度错误
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:  # 检查索引长度一致 # 确保两个索引数组长度相同
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )  # 抛出索引长度不匹配错误

    num_layers = dst.shape[0]  # 获取层数 # 读取目标张量的第一维
    src_req_size = src.shape[1]  # 获取源请求维度大小 # 读取源张量的请求维度
    src_step_size = src.shape[2]  # 获取源步骤维度大小 # 读取源张量的步骤维度
    dst_req_size = dst.shape[1]  # 获取目标请求维度大小 # 读取目标张量的缓存大小

    # Flatten trailing dimensions: number of elements per (layer, cache_line) entry. # 展平尾部维度：每个(层, 缓存行)条目的元素数。
    elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])  # 计算每个条目的元素数 # 总元素数除以层数×缓存行数

    # Get strides (in elements, not bytes) # 获取步长（以元素为单位，非字节）
    src_layer_stride = src.stride(0)  # 获取源层步长 # 读取源张量的层步长
    src_req_stride = src.stride(1)  # 获取源请求步长 # 读取源张量的请求步长
    src_step_stride = src.stride(2)  # 获取源步骤步长 # 读取源张量的步骤步长
    dst_layer_stride = dst.stride(0)  # 获取目标层步长 # 读取目标张量的层步长
    dst_req_stride = dst.stride(1)  # 获取目标请求步长 # 读取目标张量的缓存行步长

    # Ensure indices are int32 and contiguous # 确保索引为int32且连续
    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()  # 转换目标索引为int32并确保连续 # 确保索引格式正确
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()  # 转换步骤索引为int32并确保连续 # 确保索引格式正确

    # Ensure tensors are contiguous # 确保张量连续
    if not dst.is_contiguous():  # 检查目标张量是否连续
        raise ValueError("dst tensor must be contiguous")  # 抛出不连续错误
    if not src.is_contiguous():  # 检查源张量是否连续
        raise ValueError("src tensor must be contiguous")  # 抛出不连续错误

    # Block size for copying elements # 复制元素的块大小
    BLOCK_SIZE = 1024  # 设置块大小为1024 # 每个线程块处理的元素数

    # Grid over all requests - invalid ones will early-exit in the kernel # 网格覆盖所有请求 - 无效请求将在核中提前退出
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))  # 计算启动网格 # 请求×层数×元素块数

    _fused_mamba_state_scatter_with_mask_kernel[grid](  # 启动融合Mamba状态散射核 # 调用Triton核函数
        src,  # 源数据
        dst,  # 目标数据
        dst_indices_raw,  # 目标索引
        step_indices_raw,  # 步骤索引
        elem_per_entry,  # 每个条目的元素数
        src_layer_stride,  # 源层步长
        src_req_stride,  # 源请求步长
        src_step_stride,  # 源步骤步长
        dst_layer_stride,  # 目标层步长
        dst_req_stride,  # 目标请求步长
        src_req_size,  # 源请求维度大小
        src_step_size,  # 源步骤维度大小
        dst_req_size,  # 目标请求维度大小
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小
    )
