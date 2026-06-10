# 注意力层工具函数模块
# 提供FlashInfer和FlashMLA的KV索引创建、MLA注意力量化与RoPE、序列填充、
# KV缓存重整形、融合QK RoPE与缓存写入等Triton核函数及其启动包装器
import torch  # 导入PyTorch库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.utils import is_cuda  # 导入CUDA检测工具函数

_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096  # FlashMLA创建KV缓存块大小常量
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)  # Triton编译期常量形式的块大小

_is_cuda = is_cuda()  # 检测当前是否为CUDA环境

if _is_cuda:  # 如果是CUDA环境
    from sglang.jit_kernel.concat_mla import concat_mla_absorb_q  # 导入MLA吸收Q拼接核函数

from sglang.jit_kernel.utils import is_arch_support_pdl  # 导入架构PDL支持检测函数


@triton.jit  # Triton JIT编译装饰器
def create_flashinfer_kv_indices_triton(  # 创建FlashInfer KV索引的Triton核函数
    req_to_token_ptr,  # [max_batch, max_context_len] 请求到token映射指针
    req_pool_indices_ptr,  # 请求池索引指针
    page_kernel_lens_ptr,  # 页面内核长度指针
    kv_indptr,  # KV间接指针
    kv_start_idx,  # KV起始索引
    kv_indices_ptr,  # KV索引输出指针
    req_to_token_ptr_stride: tl.constexpr,  # 请求到token映射的步长（编译期常量）
):
    BLOCK_SIZE: tl.constexpr = 512  # 块大小编译期常量
    pid = tl.program_id(axis=0)  # 获取当前程序实例ID

    # find the req pool idx, this is for batch to token  # 查找请求池索引，用于批次到token的映射
    req_pool_index = tl.load(req_pool_indices_ptr + pid)  # 加载当前请求的池索引
    kv_indices_offset = tl.load(kv_indptr + pid)  # 加载当前请求的KV索引偏移

    kv_start = 0  # KV起始位置初始化为0
    kv_end = 0  # KV结束位置初始化为0
    if kv_start_idx:  # 如果提供了KV起始索引
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)  # 加载KV起始索引并转为int32
        kv_end = kv_start  # KV结束位置初始化为起始位置
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)  # 加载页面内核长度并累加到结束位置

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)  # 计算需要循环的次数
    for i in range(num_loop):  # 遍历每个块
        # index into req_to_token_ptr needs to be int64  # req_to_token_ptr的索引需要是int64类型
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE  # 计算当前块的偏移量
        mask = offset < kv_end - kv_start  # 生成有效数据的掩码
        data = tl.load(  # 从req_to_token_ptr加载数据
            req_to_token_ptr  # 请求到token映射基地址
            + req_pool_index * req_to_token_ptr_stride  # 加上请求池索引偏移
            + kv_start  # 加上KV起始偏移
            + offset,  # 加上当前块内偏移
            mask=mask,  # 应用掩码
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)  # 将数据写入KV索引输出


def get_num_page_per_block_flashmla(page_size: int = 64) -> int:  # 获取FlashMLA每个块中的页面数量
    num_page_per_block = _FLASHMLA_CREATE_KV_BLOCK_SIZE // page_size  # 块大小除以页面大小
    return num_page_per_block  # 返回每个块中的页面数


@triton.jit  # Triton JIT编译装饰器
def create_flashmla_kv_indices_triton(  # 创建FlashMLA KV索引的Triton核函数
    req_to_token_ptr,  # [max_batch, max_context_len] 请求到token映射指针
    req_pool_indices_ptr,  # 请求池索引指针
    page_kernel_lens_ptr,  # 页面内核长度指针
    kv_start_idx,  # KV起始索引
    kv_indices_ptr,  # KV索引输出指针
    req_to_token_ptr_stride: tl.constexpr,  # 请求到token映射的步长（编译期常量）
    kv_indices_ptr_stride: tl.constexpr,  # KV索引输出的步长（编译期常量）
    PAGED_SIZE: tl.constexpr = 64,  # 分页大小（编译期常量）
):
    NUM_PAGE_PER_BLOCK: tl.constexpr = (  # 每个块中的页面数（编译期常量）
        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE  # 块大小除以分页大小
    )
    pid = tl.program_id(axis=0)  # 获取当前程序实例ID

    # find the req pool idx, this is for batch to token  # 查找请求池索引，用于批次到token的映射
    req_pool_index = tl.load(req_pool_indices_ptr + pid)  # 加载当前请求的池索引

    kv_start = 0  # KV起始位置初始化为0
    kv_end = 0  # KV结束位置初始化为0
    if kv_start_idx:  # 如果提供了KV起始索引
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)  # 加载KV起始索引并转为int32
        kv_end = kv_start  # KV结束位置初始化为起始位置

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)  # 加载页面内核长度并累加到结束位置

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)  # 计算分页数量
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)  # 计算需要循环的块数

    for i in range(num_pages_loop):  # 遍历每个块
        # index into req_to_token_ptr needs to be int64  # req_to_token_ptr的索引需要是int64类型
        paged_offset = (  # 计算分页偏移量
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK  # 块内页面索引加块偏移
        ) * PAGED_SIZE  # 乘以分页大小
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK  # 计算输出分页偏移

        mask = paged_offset < num_paged * PAGED_SIZE  # 生成输入有效数据的掩码
        mask_out = paged_offset_out < num_paged  # 生成输出有效数据的掩码

        data = tl.load(  # 从req_to_token_ptr加载数据
            req_to_token_ptr  # 请求到token映射基地址
            + req_pool_index * req_to_token_ptr_stride  # 加上请求池索引偏移
            + kv_start  # 加上KV起始偏移
            + paged_offset,  # 加上分页偏移
            mask=mask,  # 应用掩码
        )
        tl.store(  # 将数据写入KV索引输出
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,  # 输出地址
            data // PAGED_SIZE,  # 将token ID转换为页面ID（整除页面大小）
            mask=mask_out,  # 应用输出掩码
        )


@triton.jit  # Triton JIT编译装饰器
def concat_and_cast_mha_k_kernel(  # 拼接MHA（多头注意力）K向量的nope和rope分量的核函数
    k_ptr,  # 输出K张量指针
    k_nope_ptr,  # K的nope（非位置编码）分量指针
    k_rope_ptr,  # K的rope（旋转位置编码）分量指针
    head_cnt: tl.constexpr,  # 注意力头数量（编译期常量）
    k_stride0: tl.constexpr,  # K张量第0维步长（编译期常量）
    k_stride1: tl.constexpr,  # K张量第1维步长（编译期常量）
    nope_stride0: tl.constexpr,  # nope分量第0维步长（编译期常量）
    nope_stride1: tl.constexpr,  # nope分量第1维步长（编译期常量）
    rope_stride0: tl.constexpr,  # rope分量第0维步长（编译期常量）
    nope_dim: tl.constexpr,  # nope维度大小（编译期常量）
    rope_dim: tl.constexpr,  # rope维度大小（编译期常量）
):
    pid_loc = tl.program_id(0)  # 获取当前程序实例ID（token位置）
    head_range = tl.arange(0, head_cnt)  # 生成注意力头索引范围

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1  # 计算K张量每个头的基地址

    nope_offs = tl.arange(0, nope_dim)  # 生成nope维度的偏移范围

    src_nope_ptr = (  # 计算nope源数据指针
        k_nope_ptr  # nope基地址
        + pid_loc * nope_stride0  # 加上token偏移
        + head_range[:, None] * nope_stride1  # 加上头偏移
        + nope_offs[None, :]  # 加上维度偏移
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]  # 计算nope目标指针

    src_nope = tl.load(src_nope_ptr)  # 加载nope源数据
    tl.store(dst_nope_ptr, src_nope)  # 将nope数据存储到K张量前半部分

    rope_offs = tl.arange(0, rope_dim)  # 生成rope维度的偏移范围
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]  # 计算rope源数据指针
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]  # 计算rope目标指针（偏移nope_dim）
    src_rope = tl.load(src_rope_ptr)  # 加载rope源数据
    tl.store(dst_rope_ptr, src_rope)  # 将rope数据存储到K张量后半部分


def concat_and_cast_mha_k_triton(  # 使用Triton核函数拼接MHA K向量的nope和rope分量
    k: torch.Tensor,  # 输出K张量
    k_nope: torch.Tensor,  # K的nope分量
    k_rope: torch.Tensor,  # K的rope分量
):
    # The source data type will be implicitly converted to the target data type.  # 源数据类型将被隐式转换为目标数据类型
    assert (  # 断言检查所有张量维度为3维
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"  # 维度不匹配时报错
    assert (  # 断言检查第0维大小一致
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"  # 形状不匹配时报错
    assert (  # 断言检查第1维大小：k和k_nope头数一致，k_rope头数为1
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"  # 形状不匹配时报错
    assert (  # 断言检查最后一维大小等于nope和rope维度之和
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"  # 形状不匹配时报错

    nope_dim = k_nope.shape[-1]  # 获取nope维度大小
    rope_dim = k_rope.shape[-1]  # 获取rope维度大小
    grid = (k.shape[0],)  # 设置核函数启动网格大小（token数量）

    concat_and_cast_mha_k_kernel[grid](  # 启动拼接核函数
        k,  # 输出K张量
        k_nope,  # nope分量
        k_rope,  # rope分量
        k.shape[1],  # 注意力头数量
        k.stride(0),  # K张量第0维步长
        k.stride(1),  # K张量第1维步长
        k_nope.stride(0),  # nope分量第0维步长
        k_nope.stride(1),  # nope分量第1维步长
        k_rope.stride(0),  # rope分量第0维步长
        nope_dim,  # nope维度大小
        rope_dim,  # rope维度大小
    )


@triton.jit  # Triton JIT编译装饰器
def pad_sequence_with_mask_kernel(  # 带掩码的序列填充核函数
    input_ptr,  # (total_tokens, hidden) 输入嵌入指针
    offsets_ptr,  # (B,) 偏移量指针
    lengths_ptr,  # (B,) 序列长度指针
    output_ptr,  # (B, max_len, hidden) 输出指针
    mask_ptr,  # (B, max_len) 掩码指针
    max_len,  # 最大序列长度
    hidden_dim,  # 隐藏维度大小
    BLOCK_M: tl.constexpr,  # seq block  # 序列块大小（编译期常量）
    BLOCK_D: tl.constexpr,  # hidden block  # 隐藏维度块大小（编译期常量）
):
    b = tl.program_id(0)  # batch index  # 批次索引
    m = tl.program_id(1)  # seq block index  # 序列块索引

    offset = tl.load(offsets_ptr + b)  # 加载当前批次的偏移量
    length = tl.load(lengths_ptr + b)  # 加载当前批次的序列长度

    seq_ids = m * BLOCK_M + tl.arange(0, BLOCK_M)  # 计算序列位置索引
    hid_ids = tl.arange(0, BLOCK_D)  # 生成隐藏维度索引

    seq_mask = seq_ids < max_len  # 生成序列边界掩码
    valid_token = seq_ids < length  # 生成有效token掩码

    # input index  # 输入索引
    in_token = offset + seq_ids  # 计算输入token索引
    in_ptr = input_ptr + in_token[:, None] * hidden_dim + hid_ids[None, :]  # 计算输入数据地址

    # output index  # 输出索引
    out_ptr = (  # 计算输出数据地址
        output_ptr  # 输出基地址
        + b * max_len * hidden_dim  # 加上批次偏移
        + seq_ids[:, None] * hidden_dim  # 加上序列位置偏移
        + hid_ids[None, :]  # 加上隐藏维度偏移
    )

    values = tl.load(  # 加载输入数据
        in_ptr,  # 输入地址
        mask=valid_token[:, None] & (hid_ids[None, :] < hidden_dim),  # 应用联合掩码
        other=0.0,  # 无效位置填充0
    )

    tl.store(  # 将数据写入输出
        out_ptr,  # 输出地址
        values,  # 数据值
        mask=seq_mask[:, None] & (hid_ids[None, :] < hidden_dim),  # 应用联合掩码
    )

    # attention mask  # 注意力掩码
    if tl.program_id(2) == 0:  # 只在第3个维度为0时处理掩码
        mask_out_ptr = mask_ptr + b * max_len + seq_ids  # 计算掩码输出地址
        tl.store(mask_out_ptr, valid_token, mask=seq_mask)  # 写入注意力掩码


def pad_sequence_with_mask(  # 对变长序列进行填充并生成注意力掩码
    input_emb,  # (total_tokens, hidden) 输入嵌入
    offsets,  # (B,) 每个序列的偏移量
    lengths,  # (B,) 每个序列的长度
    max_len,  # 填充后的最大序列长度
):
    B = offsets.shape[0]  # 获取批次大小
    hidden_dim = input_emb.shape[1]  # 获取隐藏维度大小

    output = torch.zeros(  # 创建填充后的输出张量
        (B, max_len, hidden_dim),  # 形状为(批次, 最大长度, 隐藏维度)
        device=input_emb.device,  # 设备与输入一致
        dtype=input_emb.dtype,  # 数据类型与输入一致
    )
    attn_mask = torch.empty(  # 创建注意力掩码张量
        (B * max_len),  # 形状为(批次*最大长度)
        device=input_emb.device,  # 设备与输入一致
        dtype=torch.bool,  # 布尔类型
    )

    BLOCK_D = triton.next_power_of_2(hidden_dim)  # 计算隐藏维度的2的幂次块大小
    BLOCK_M = triton.next_power_of_2(max_len)  # 计算最大长度的2的幂次块大小

    grid = (  # 设置核函数启动网格
        B,  # 批次维度
        triton.cdiv(max_len, BLOCK_M),  # 序列块维度
        1,  # 掩码处理维度
    )

    pad_sequence_with_mask_kernel[grid](  # 启动填充核函数
        input_emb,  # 输入嵌入
        offsets,  # 偏移量
        lengths,  # 序列长度
        output,  # 输出张量
        attn_mask,  # 注意力掩码
        max_len,  # 最大序列长度
        hidden_dim,  # 隐藏维度大小
        BLOCK_M=BLOCK_M,  # 序列块大小
        BLOCK_D=BLOCK_D,  # 隐藏维度块大小
    )

    return B, output, attn_mask  # 返回批次大小、填充输出和注意力掩码


@triton.jit  # Triton JIT编译装饰器
def seqlens_expand_kernel(  # 序列长度展开核函数，将序列长度扩展为每个token的KV位置
    extend_seq_lens_ptr,  # [N] 扩展序列长度指针
    seq_lens_ptr,  # [N] 完整序列长度指针
    offsets_ptr,  # [N+1] 偏移量指针
    output_ptr,  # [sum(extend_seq_lens)] 输出指针
    N,  # 序列数量
    BLOCK: tl.constexpr,  # 块大小（编译期常量）
):
    pid = tl.program_id(0)  # 获取当前程序实例ID

    if pid >= N:  # 如果超出序列数量范围
        return  # 直接返回

    qo_len = tl.load(extend_seq_lens_ptr + pid)  # 加载当前序列的查询/输出长度
    kv_len = tl.load(seq_lens_ptr + pid)  # 加载当前序列的KV长度

    start = kv_len - qo_len + 1  # 计算起始位置
    out_offset = tl.load(offsets_ptr + pid)  # 加载输出偏移量

    offs = tl.arange(0, BLOCK)  # 生成块内偏移范围
    mask = offs < qo_len  # 生成有效数据掩码

    values = start + offs  # 计算每个token的KV位置值
    tl.store(output_ptr + out_offset + offs, values, mask=mask)  # 写入输出


def seqlens_expand_triton(  # 使用Triton核函数扩展序列长度为每个token的KV位置
    extend_seq_lens: torch.Tensor,  # 扩展序列长度
    seq_lens: torch.Tensor,  # 完整序列长度
    total_len: int,  # 总长度
    max_q_len: int,  # 最大查询长度
):
    """
    extend_seq_lens: [N], int32, CUDA  # 扩展序列长度，形状[N]，int32类型，CUDA设备
    seq_lens:        [N], int32, CUDA  # 完整序列长度，形状[N]，int32类型，CUDA设备
    """
    assert extend_seq_lens.is_cuda  # 断言扩展序列长度在CUDA设备上
    assert seq_lens.is_cuda  # 断言完整序列长度在CUDA设备上

    N = extend_seq_lens.numel()  # 获取序列数量

    offsets = torch.zeros(N + 1, device=extend_seq_lens.device, dtype=torch.int32)  # 创建偏移量张量
    offsets[1:] = torch.cumsum(extend_seq_lens, dim=0)  # 计算累积和作为偏移量
    output = torch.empty(total_len, device=extend_seq_lens.device, dtype=torch.int32)  # 创建输出张量

    BLOCK = triton.next_power_of_2(max_q_len)  # 计算最大查询长度的2的幂次块大小
    grid = (N,)  # 设置核函数启动网格大小

    seqlens_expand_kernel[grid](  # 启动序列长度展开核函数
        extend_seq_lens,  # 扩展序列长度
        seq_lens,  # 完整序列长度
        offsets,  # 偏移量
        output,  # 输出
        N,  # 序列数量
        BLOCK=BLOCK,  # 块大小
    )

    return output  # 返回展开后的输出


# When num_kv_heads=1, we have tensors with degenerate strides,  # 当num_kv_heads=1时，张量会有退化步长
# For example, as below, where we have stride[-3] == stride[-2]:  # 例如下面的情况，stride[-3] == stride[-2]
# - shape: [num_pages, 1, 64, 128]  # 形状: [num_pages, 1, 64, 128]
# - stride: [8192, 128, 128, 1]  # 步长: [8192, 128, 128, 1]
# This will cause TMA desc validation fail in flashinfer (trtllm-mha backend).  # 这会导致flashinfer（trtllm-mha后端）中TMA描述符验证失败
#
# See: https://github.com/flashinfer-ai/flashinfer/issues/2232  # 参见: https://github.com/flashinfer-ai/flashinfer/issues/2232
def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:  # 规范化张量步长，修复退化步长问题
    """
    Adjust degenerate strides for a tensor, make it canonical.  # 调整张量的退化步长，使其规范化
    """
    sizes = tensor.size()  # 获取张量形状
    strides = tensor.stride()  # 获取张量步长
    ndim = tensor.dim()  # 获取张量维度数

    need_fix = any(  # 检查是否需要修复
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)  # 查找退化步长条件
    )

    if not need_fix:  # 如果不需要修复
        return tensor  # 直接返回原张量

    # canonicalize the stride  # 规范化步长
    # Example:  # 示例
    # - shape: [num_pages, 1, 64, 128]  # 形状: [num_pages, 1, 64, 128]
    # - stride: [8192, 128, 128, 1] (wrong!)  # 步长: [8192, 128, 128, 1] (错误！)
    # Gives new stride: [8192, 8192, 128 ,1] (correct!)  # 新步长: [8192, 8192, 128, 1] (正确！)
    new_strides = [0] * ndim  # 创建新步长列表
    new_strides[-1] = 1  # 最内维步长为1
    for i in range(ndim - 2, -1, -1):  # 从倒数第二维向前遍历
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]  # 计算规范化步长

    return tensor.as_strided(sizes, new_strides)  # 返回使用新步长的张量视图


def mla_quantize_and_rope_for_fp8(  # MLA注意力的FP8量化与RoPE应用函数
    q_nope: torch.Tensor,  # 查询的非位置编码分量
    q_rope: torch.Tensor,  # 查询的旋转位置编码分量
    k_nope: torch.Tensor,  # 键的非位置编码分量
    k_rope: torch.Tensor,  # 键的旋转位置编码分量
    pos_ids: torch.Tensor,  # 位置索引
    cos_sin_cache: torch.Tensor,  # 预计算的余弦/正弦缓存
    is_neox: bool,  # 是否使用NeoX风格的RoPE
    kv_lora_rank: int,  # KV LoRA秩（非位置编码维度）
    qk_rope_head_dim: int,  # QK旋转位置编码头维度
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回量化后的元组
    import flashinfer.rope  # 导入flashinfer的RoPE模块

    """Quantize and apply RoPE for FP8 attention path.  # 为FP8注意力路径执行量化和RoPE应用

        This function handles the FP8 quantization and RoPE application for MLA attention.  # 此函数处理MLA注意力的FP8量化和RoPE应用
        It takes separate query/key nope and rope components, applies RoPE to the rope parts,  # 接收分离的查询/键nope和rope分量，对rope部分应用RoPE
        quantizes all components to FP8, and merges the query components into a single tensor.  # 将所有分量量化为FP8，并将查询分量合并为单个张量

        Args:  # 参数
            q_nope: Query no-position-encoding component [seq_len, num_heads, kv_lora_rank]  # 查询非位置编码分量 [序列长度, 头数, kv_lora_rank]
                - expected dtype: torch.bfloat16  # 预期数据类型: torch.bfloat16
            q_rope: Query RoPE component [seq_len, num_heads, qk_rope_head_dim]  # 查询RoPE分量 [序列长度, 头数, qk_rope_head_dim]
                - expected dtype: torch.bfloat16  # 预期数据类型: torch.bfloat16
            k_nope: Key no-position-encoding component [seq_len, num_heads, kv_lora_rank]  # 键非位置编码分量 [序列长度, 头数, kv_lora_rank]
                - expected dtype: torch.bfloat16  # 预期数据类型: torch.bfloat16
            k_rope: Key RoPE component [seq_len, num_heads, qk_rope_head_dim]  # 键RoPE分量 [序列长度, 头数, qk_rope_head_dim]
                - expected dtype: torch.bfloat16  # 预期数据类型: torch.bfloat16
            pos_ids: Position indices for each token  # 每个token的位置索引
                - expected dtype: torch.int64 or torch.int32  # 预期数据类型: torch.int64或torch.int32
            cos_sin_cache: Precomputed cosine/sine cache for RoPE  # 预计算的RoPE余弦/正弦缓存
                - expected dtype: matches q_/k_ input dtype (torch.bfloat16)  # 预期数据类型: 与q_/k_输入类型匹配(torch.bfloat16)
            is_neox: Whether to use NeoX-style RoPE (interleaved) or GPT-style (half rotation)  # 是否使用NeoX风格RoPE（交错）或GPT风格（半旋转）
            kv_lora_rank: Dimension of the no-position-encoding component  # 非位置编码分量的维度
            qk_rope_head_dim: Dimension of the RoPE component  # RoPE分量的维度

        Returns:  # 返回值
            tuple: (merged_q_out, k_nope_out, k_rope_out) quantized to FP8  # 元组: (合并查询输出, 键nope输出, 键rope输出)，量化为FP8
                - merged_q_out: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim], dtype=torch.float8_e4m3fn  # 合并查询输出
                - k_nope_out:   [seq_len, num_heads, kv_lora_rank], dtype=torch.float8_e4m3fn  # 键nope输出
                - k_rope_out:   [seq_len, num_heads, qk_rope_head_dim], dtype=torch.float8_e4m3fn  # 键rope输出
        """
    attn_dtype = torch.float8_e4m3fn  # 设置注意力计算数据类型为FP8
    q_len, num_heads = q_rope.shape[0], q_rope.shape[1]  # 获取查询序列长度和头数

    # Allocate output tensors with FP8 dtype  # 分配FP8数据类型的输出张量
    # Query output will contain merged nope + rope components  # 查询输出将包含合并的nope + rope分量
    q_out = q_rope.new_empty(  # 创建查询输出张量
        q_len,  # 序列长度
        num_heads,  # 头数
        kv_lora_rank + qk_rope_head_dim,  # 总维度=nope维度+rope维度
        dtype=attn_dtype,  # FP8数据类型
    )

    # Key outputs maintain original shapes but with FP8 dtype  # 键输出保持原始形状但使用FP8数据类型
    k_rope_out = k_rope.new_empty(k_rope.shape, dtype=attn_dtype)  # 创建键rope输出张量
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)  # 创建键nope输出张量

    # Apply RoPE and quantize all components in a single fused kernel call  # 在单个融合核函数调用中应用RoPE并量化所有分量
    # This kernel handles:  # 此核函数处理
    # 1. RoPE application to q_rope and k_rope using cos_sin_cache and positions  # 1. 使用cos_sin_cache和位置对q_rope和k_rope应用RoPE
    # 2. Quantization of all components to FP8 format  # 2. 将所有分量量化为FP8格式
    # 3. Output placement into pre-allocated tensors  # 3. 将输出放入预分配的张量
    flashinfer.rope.mla_rope_quantize_fp8(  # 调用flashinfer的MLA RoPE量化FP8核函数
        q_rope=q_rope,  # 查询rope分量
        k_rope=k_rope,  # 键rope分量
        q_nope=q_nope,  # 查询nope分量
        k_nope=k_nope,  # 键nope分量
        cos_sin_cache=cos_sin_cache,  # 余弦/正弦缓存
        pos_ids=pos_ids,  # 位置索引
        is_neox=is_neox,  # 是否NeoX风格
        quantize_dtype=attn_dtype,  # 量化目标数据类型
        # Output tensor slicing: q_out contains [nope_part, rope_part]  # 输出张量切片: q_out包含[nope部分, rope部分]
        q_rope_out=q_out[..., kv_lora_rank:],  # RoPE part goes to end  # RoPE部分放在末尾
        k_rope_out=k_rope_out,  # 键rope输出
        q_nope_out=q_out[..., :kv_lora_rank],  # Nope part goes to beginning  # Nope部分放在开头
        k_nope_out=k_nope_out,  # 键nope输出
        # Quantization scales (set to 1.0 for no additional scaling)  # 量化缩放因子（设为1.0表示无额外缩放）
        quant_scale_q=1.0,  # 查询量化缩放因子
        quant_scale_kv=1.0,  # 键值量化缩放因子
        enable_pdl=is_arch_support_pdl(),  # 是否启用PDL
    )

    return q_out, k_nope_out, k_rope_out  # 返回量化后的查询、键nope、键rope


def concat_mla_absorb_q_general(q_nope, q_rope):  # MLA吸收Q拼接的通用函数，根据条件选择优化路径
    if _is_cuda and q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64:  # CUDA环境下且维度匹配时使用优化核
        return concat_mla_absorb_q(q_nope, q_rope)  # 使用优化的CUDA核函数
    else:  # 否则使用通用方法
        return torch.cat([q_nope, q_rope], dim=-1)  # 在最后一维拼接nope和rope分量


@triton.jit  # Triton JIT编译装饰器
def reshape_and_cache_flash(  # 将token级K/V张量重整形并写入分页KV缓存的Triton核函数
    key_ptr,  # 源键张量指针
    value_ptr,  # 源值张量指针
    key_cache_ptr,  # 目标键缓存指针
    value_cache_ptr,  # 目标值缓存指针
    slot_mapping_ptr,  # token到缓存槽的映射指针
    swa_slot_mapping_ptr,  # SWA模式的二级槽映射指针
    k_scale_ptr,  # 键缩放因子指针
    v_scale_ptr,  # 值缩放因子指针
    block_stride,  # 缓存块间步长
    key_stride,  # 源键token间步长
    value_stride,  # 源值token间步长
    num_heads,  # 注意力头数
    head_size,  # 每个头的隐藏维度
    block_size,  # 每个缓存块的槽位数
    HEAD_BLOCK: tl.constexpr,  # 每个程序处理的头块数
    BLOCK_D: tl.constexpr,  # 向量化维度大小（2的幂填充）
    HAS_SWA: tl.constexpr,  # 是否启用SWA重映射
    USE_SCALE: tl.constexpr,  # 是否在存储前启用缩放除法
):
    """
    Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.  # 将逐token的K/V张量重整形为分页KV缓存布局的Triton核函数

    Source layout:  # 源布局
        key/value: [num_tokens, num_heads, head_size]  # 键/值: [token数, 头数, 头维度]

    Target cache layout:  # 目标缓存布局
        cache: [num_blocks, block_size, num_heads, head_size]  # 缓存: [块数, 块大小, 头数, 头维度]

    Each Triton program instance handles:  # 每个Triton程序实例处理
        - one token (program_id(0))  # 一个token（program_id(0)）
        - one block of heads (program_id(1))  # 一个头块（program_id(1)）

    Features:  # 功能
        - optional SWA slot remapping  # 可选的SWA槽位重映射
        - optional FP8 scale dequantization before cache write  # 可选的缓存写入前FP8缩放反量化

    Args:  # 参数
        key_ptr: Pointer to source key tensor.  # 源键张量指针
        value_ptr: Pointer to source value tensor.  # 源值张量指针
        key_cache_ptr: Pointer to destination key cache tensor.  # 目标键缓存指针
        value_cache_ptr: Pointer to destination value cache tensor.  # 目标值缓存指针
        slot_mapping_ptr: Maps token -> cache slot.  # token到缓存槽的映射
        swa_slot_mapping_ptr: Optional second-stage slot remap for SWA mode.  # SWA模式的可选二级槽位重映射
        k_scale_ptr: Optional key scaling factor pointer.  # 可选的键缩放因子指针
        v_scale_ptr: Optional value scaling factor pointer.  # 可选的值缩放因子指针
        block_stride: Stride between cache blocks.  # 缓存块间步长
        key_stride: Stride between source key tokens.  # 源键token间步长
        value_stride: Stride between source value tokens.  # 源值token间步长
        num_heads: Number of attention heads.  # 注意力头数
        head_size: Hidden dimension per head.  # 每个头的隐藏维度
        block_size: Number of slots per cache block.  # 每个缓存块的槽位数
        HEAD_BLOCK: Number of heads processed per program.  # 每个程序处理的头数
        BLOCK_D: Vectorized dimension size (power-of-2 padded).  # 向量化维度大小（2的幂填充）
        HAS_SWA: Enable SWA remapping.  # 启用SWA重映射
        USE_SCALE: Enable scale division before storing.  # 存储前启用缩放除法
    """

    # ----------------------------------  # ----------------------------------
    # program ids  # 程序ID
    # pid0 = token  # pid0 = token索引
    # pid1 = head block  # pid1 = 头块索引
    # ----------------------------------  # ----------------------------------
    token_idx = tl.program_id(0)  # 获取token索引
    head_block_idx = tl.program_id(1)  # 获取头块索引

    # ----------------------------------  # ----------------------------------
    # slot mapping  # 槽位映射
    # ----------------------------------  # ----------------------------------
    slot_idx = tl.load(slot_mapping_ptr + token_idx)  # 加载当前token的缓存槽索引

    if HAS_SWA:  # 如果启用SWA
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)  # 通过SWA映射重新获取槽索引

    if slot_idx < 0:  # 如果槽索引无效
        return  # 直接返回（跳过此token）

    block_idx = slot_idx // block_size  # 计算缓存块索引
    block_offset = slot_idx % block_size  # 计算块内偏移

    # ----------------------------------  # ----------------------------------
    # head range  # 头范围
    # ----------------------------------  # ----------------------------------
    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)  # 计算当前处理的头索引

    head_mask = head_idx < num_heads  # 生成有效头的掩码

    dim_idx = tl.arange(0, BLOCK_D)  # 生成维度索引范围

    # shape = [HEAD_BLOCK, BLOCK_D]  # 形状 = [头块大小, 维度块大小]
    offs = head_idx[:, None] * head_size + dim_idx[None, :]  # 计算头维度偏移

    mask = head_mask[:, None] & (dim_idx[None, :] < head_size)  # 生成联合掩码

    # ----------------------------------  # ----------------------------------
    # source load  # 源数据加载
    # ----------------------------------  # ----------------------------------
    src_key = token_idx * key_stride + offs  # 计算源键数据地址
    src_value = token_idx * value_stride + offs  # 计算源值数据地址

    k = tl.load(key_ptr + src_key, mask=mask)  # 加载键数据
    v = tl.load(value_ptr + src_value, mask=mask)  # 加载值数据

    # ----------------------------------  # ----------------------------------
    # optional scale  # 可选缩放
    # ----------------------------------  # ----------------------------------
    if USE_SCALE:  # 如果需要缩放
        k_scale = tl.load(k_scale_ptr)  # 加载键缩放因子
        v_scale = tl.load(v_scale_ptr)  # 加载值缩放因子

        k = k / k_scale  # 键除以缩放因子（反量化）
        v = v / v_scale  # 值除以缩放因子（反量化）

    # ----------------------------------  # ----------------------------------
    # target layout  # 目标布局
    # [block_idx, block_offset, head, dim]  # [块索引, 块内偏移, 头, 维度]
    # ----------------------------------  # ----------------------------------
    tgt = block_idx * block_stride + block_offset * num_heads * head_size + offs  # 计算目标缓存地址

    tl.store(key_cache_ptr + tgt, k, mask=mask)  # 将键数据写入缓存
    tl.store(value_cache_ptr + tgt, v, mask=mask)  # 将值数据写入缓存


def launch_reshape_and_cache_flash(  # 启动reshape_and_cache_flash Triton核函数的包装器
    key,  # 源键张量 [num_tokens, num_heads, head_size]
    value,  # 源值张量 [num_tokens, num_heads, head_size]
    key_cache,  # 目标键缓存 [num_blocks, block_size, num_heads, head_size]
    value_cache,  # 目标值缓存 [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # token到缓存槽的映射
    swa_slot_mapping=None,  # 可选的SWA重映射表
    k_scale=None,  # 可选的键缩放因子
    v_scale=None,  # 可选的值缩放因子
):
    """
    Launch wrapper for reshape_and_cache_flash Triton kernel.  # reshape_and_cache_flash Triton核函数的启动包装器

    This wrapper prepares launch configuration and dispatches the Triton kernel  # 此包装器准备启动配置并调度Triton核函数
    that writes token-major K/V tensors into paged KV cache layout.  # 将token级K/V张量写入分页KV缓存布局

    Args:  # 参数
        key: Source key tensor [num_tokens, num_heads, head_size]  # 源键张量 [token数, 头数, 头维度]
        value: Source value tensor [num_tokens, num_heads, head_size]  # 源值张量 [token数, 头数, 头维度]
        key_cache: Destination key cache [num_blocks, block_size, num_heads, head_size]  # 目标键缓存
        value_cache: Destination value cache [num_blocks, block_size, num_heads, head_size]  # 目标值缓存
        slot_mapping: Token-to-cache slot mapping  # token到缓存槽的映射
        swa_slot_mapping: Optional SWA remapping table  # 可选的SWA重映射表
        k_scale: Optional key scaling factor  # 可选的键缩放因子
        v_scale: Optional value scaling factor  # 可选的值缩放因子
    """

    num_tokens = key.shape[0]  # 获取token数量
    num_heads = key.shape[1]  # 获取头数
    head_size = key.shape[2]  # 获取头维度大小

    HEAD_BLOCK = 4  # 每个程序处理的头块大小

    BLOCK_D = triton.next_power_of_2(head_size)  # 计算头维度的2的幂次块大小

    grid = (  # 设置核函数启动网格
        num_tokens,  # token维度
        triton.cdiv(num_heads, HEAD_BLOCK),  # 头块维度
    )

    reshape_and_cache_flash[grid](  # 启动重整形缓存核函数
        key,  # 源键张量
        value,  # 源值张量
        key_cache,  # 目标键缓存
        value_cache,  # 目标值缓存
        slot_mapping,  # 槽位映射
        swa_slot_mapping,  # SWA映射
        k_scale if k_scale is not None else key,  # 键缩放因子（不存在时用key占位）
        v_scale if v_scale is not None else key,  # 值缩放因子（不存在时用key占位）
        key_cache.stride(0),  # 缓存块间步长
        key.stride(0),  # 键token间步长
        value.stride(0),  # 值token间步长
        num_heads,  # 头数
        head_size,  # 头维度
        key_cache.shape[1],  # 缓存块大小
        HEAD_BLOCK=HEAD_BLOCK,  # 头块大小
        BLOCK_D=BLOCK_D,  # 维度块大小
        HAS_SWA=(swa_slot_mapping is not None),  # 是否有SWA映射
        USE_SCALE=(k_scale is not None),  # 是否有缩放因子
    )


@triton.jit  # Triton JIT编译装饰器
def _get_gptj_rotated_x(  # 获取GPT-J风格的旋转向量（相邻维度配对旋转）
    x,  # 输入向量
    x_rotated_mask,  # 旋转掩码
    BLOCK_D: tl.constexpr,  # 维度块大小（编译期常量）
    BLOCK_D_HALF: tl.constexpr,  # 半维度块大小（编译期常量）
):
    # GPT-J rotary layout:  # GPT-J旋转位置编码布局
    # Pair adjacent dimensions and apply:  # 将相邻维度配对并应用旋转
    # [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]  # [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]

    # Apply sign inversion on odd positions.  # 对奇数位置应用符号反转
    x_rotated = tl.where(x_rotated_mask, x, -x)  # 根据掩码选择正或负
    # Reshape into (D/2, 2) pairs.  # 重塑为(D/2, 2)对
    x_rotated = tl.reshape(x_rotated, (BLOCK_D_HALF, 2))  # 重塑为半维度对
    # Swap each pair.  # 交换每对元素
    x_rotated = tl.flip(x_rotated, 1)  # 沿对维度翻转
    # Flatten back to original shape.  # 展平回原始形状
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))  # 展平为一维
    return x_rotated  # 返回旋转后的向量


@triton.jit  # Triton JIT编译装饰器
def _get_neox_rotated_x(  # 获取GPT-NeoX风格的旋转向量（前后半部分配对旋转）
    x,  # 输入向量
    x_rotated_mask,  # 旋转掩码
    BLOCK_D: tl.constexpr,  # 维度块大小（编译期常量）
    BLOCK_D_HALF: tl.constexpr,  # 半维度块大小（编译期常量）
):
    # GPT-NeoX rotary layout:  # GPT-NeoX旋转位置编码布局
    # Split head dimension into two halves:  # 将头维度分成两半
    # [x0, x1, x2, x3] -> [-x2, -x3, x0, x1]  # [x0, x1, x2, x3] -> [-x2, -x3, x0, x1]

    # Keep first half positive, second half negative.  # 保持前半部分为正，后半部分为负
    x_rotated = tl.where(x_rotated_mask, x, -x)  # 根据掩码选择正或负
    # Reshape into (2, D/2).  # 重塑为(2, D/2)
    x_rotated = tl.reshape(x_rotated, (2, BLOCK_D_HALF))  # 重塑为两行半维度
    # Reverse each half.  # 反转每一半
    x_rotated = tl.flip(x_rotated, 1)  # 沿半维度翻转
    # Flatten and reverse full vector.  # 展平并反转整个向量
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))  # 展平为一维
    x_rotated = tl.flip(x_rotated, 0)  # 反转整个向量
    return x_rotated  # 返回旋转后的向量


@triton.jit  # Triton JIT编译装饰器
def _unit_rope(  # 单个注意力头向量的RoPE变换
    x_ptrs,  # 输入向量指针
    cos,  # 余弦值
    sin,  # 正弦值
    d_pe_offs,  # 位置编码维度偏移
    IS_NEOX: tl.constexpr,  # 是否NeoX风格（编译期常量）
    BLOCK_D_pe: tl.constexpr,  # 位置编码维度块大小（编译期常量）
    BLOCK_D_HALF_pe: tl.constexpr,  # 半位置编码维度块大小（编译期常量）
):
    # Load one full attention head vector.  # 加载一个完整的注意力头向量
    x_pe = tl.load(x_ptrs)  # 加载位置编码输入

    # Stage 1: Build rotated vector according to rotary layout.  # 阶段1：根据旋转布局构建旋转向量
    if IS_NEOX:  # 如果使用NeoX风格
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe  # 前半部分为True
        x_pe_rotated = _get_neox_rotated_x(  # 获取NeoX旋转向量
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:  # 使用GPT-J风格
        x_rotated_mask = d_pe_offs % 2 == 0  # 偶数位置为True
        x_pe_rotated = _get_gptj_rotated_x(  # 获取GPT-J旋转向量
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    # Stage 2: Apply RoPE transform:  # 阶段2：应用RoPE变换
    # x' = x*cos + rotate(x)*sin  # x' = x*cos + rotate(x)*sin
    x_pe = x_pe * cos + x_pe_rotated * sin  # 计算RoPE变换结果

    return x_pe  # 返回RoPE变换后的向量


@triton.jit  # Triton JIT编译装饰器
def _load_cos_sin(  # 加载余弦和正弦值
    cos_sin_ptr,  # 余弦/正弦缓存指针
    pos,  # 位置索引
    d_cos_offs,  # 余弦维度偏移
    stride_t,  # 时间步步长
    stride_d,  # 维度步长
    freq_dim,  # 频率维度大小
):
    base = pos * stride_t  # 计算基础偏移
    cos = tl.load(cos_sin_ptr + base + d_cos_offs * stride_d)  # 加载余弦值
    sin = tl.load(cos_sin_ptr + base + (d_cos_offs + freq_dim) * stride_d)  # 加载正弦值（偏移freq_dim）
    return cos, sin  # 返回余弦和正弦值


@triton.jit  # Triton JIT编译装饰器
def _fused_qk_rope_reshape_and_cache_kernel(  # 融合QK RoPE、重整形和缓存写入的Triton核函数
    q_ptr,  # 查询张量指针
    k_ptr,  # 键张量指针
    v_ptr,  # 值张量指针
    pos_ptr,  # 位置索引指针
    cos_sin_ptr,  # 余弦/正弦缓存指针
    offs_ptr,  # 偏移量指针
    key_cache_ptr,  # 键缓存指针
    value_cache_ptr,  # 值缓存指针
    slot_mapping_ptr,  # 槽位映射指针
    swa_slot_mapping_ptr,  # SWA槽位映射指针
    q_out_ptr,  # 查询输出指针
    k_out_ptr,  # 键输出指针
    zeros_out_ptr,  # 零输出指针
    T,  # 解码token数
    T_slot,  # 槽位映射token数
    q_stride_t,  # 查询时间步步长
    q_stride_h,  # 查询头步长
    q_stride_d,  # 查询维度步长
    k_stride_t,  # 键时间步步长
    k_stride_h,  # 键头步长
    k_stride_d,  # 键维度步长
    v_stride_t,  # 值时间步步长
    v_stride_h,  # 值头步长
    v_stride_d,  # 值维度步长
    cos_sin_stride_t,  # 余弦/正弦时间步步长
    cos_sin_stride_d,  # 余弦/正弦维度步长
    q_out_stride_t,  # 查询输出时间步步长
    q_out_stride_h,  # 查询输出头步长
    q_out_stride_d,  # 查询输出维度步长
    k_out_stride_t,  # 键输出时间步步长
    k_out_stride_h,  # 键输出头步长
    k_out_stride_d,  # 键输出维度步长
    key_cache_stride_t,  # 键缓存时间步步长
    key_cache_stride_h,  # 键缓存头步长
    key_cache_stride_d,  # 键缓存维度步长
    key_cache_stride_b,  # 键缓存块步长
    key_cache_stride_x,  # 键缓存x步长
    value_cache_stride_t,  # 值缓存时间步步长
    value_cache_stride_h,  # 值缓存头步长
    value_cache_stride_d,  # 值缓存维度步长
    value_cache_stride_b,  # 值缓存块步长
    value_cache_stride_slot_chunk,  # 值缓存槽块步长
    value_cache_stride_x,  # 值缓存x步长
    zeros_out_stride_t,  # 零输出时间步步长
    zeros_out_stride_h,  # 零输出头步长
    zeros_out_stride_d,  # 零输出维度步长
    k_scale_ptr,  # 键缩放因子指针
    v_scale_ptr,  # 值缩放因子指针
    QH_PER_KH: tl.constexpr,  # 每个KV头对应的查询头数（编译期常量）
    QH: tl.constexpr,  # 查询头数（编译期常量）
    KH: tl.constexpr,  # 键头数（编译期常量）
    REUSE_FREQS_FRONT_PART: tl.constexpr,  # 是否复用前半部分频率（编译期常量）
    IS_NEOX: tl.constexpr,  # 是否NeoX风格（编译期常量）
    BLOCK_D_pe: tl.constexpr,  # 位置编码维度块大小（编译期常量）
    BLOCK_D_HALF_pe: tl.constexpr,  # 半位置编码维度块大小（编译期常量）
    BLOCK_SIZE: tl.constexpr,  # 缓存块大小（编译期常量）
    X_SIZE: tl.constexpr,  # x维度大小（编译期常量）
    FLASH_LAYOUT: tl.constexpr,  # 是否使用Flash布局（编译期常量）
    VALUE_SHUFFLE_LAYOUT: tl.constexpr = False,  # 是否使用值洗牌布局（编译期常量）
    HAVE_POS: tl.constexpr = False,  # 是否有位置偏移（编译期常量）
    HAVE_K_SCALE: tl.constexpr = False,  # 是否有键缩放（编译期常量）
    HAVE_V_SCALE: tl.constexpr = False,  # 是否有值缩放（编译期常量）
    HAVE_ZEROS: tl.constexpr = False,  # 是否有零输出（编译期常量）
    HAS_SWA: tl.constexpr = False,  # 是否有SWA映射（编译期常量）
):
    # ============================================================  # ============================================================
    # Stage 0: Static stride assumptions for Triton compiler  # 阶段0：为Triton编译器设置静态步长假设
    #
    # These assumptions help Triton optimize pointer arithmetic and  # 这些假设帮助Triton优化指针算术和
    # simplify generated address calculations.  # 简化生成的地址计算
    # ============================================================  # ============================================================

    tl.assume(q_stride_t >= 0)  # 假设查询时间步步长非负
    tl.assume(q_stride_h >= 0)  # 假设查询头步长非负
    tl.assume(q_stride_d >= 0)  # 假设查询维度步长非负
    tl.assume(k_stride_t >= 0)  # 假设键时间步步长非负
    tl.assume(k_stride_h >= 0)  # 假设键头步长非负
    tl.assume(k_stride_d >= 0)  # 假设键维度步长非负
    tl.assume(v_stride_t >= 0)  # 假设值时间步步长非负
    tl.assume(v_stride_h >= 0)  # 假设值头步长非负
    tl.assume(v_stride_d >= 0)  # 假设值维度步长非负
    tl.assume(cos_sin_stride_t >= 0)  # 假设余弦/正弦时间步步长非负
    tl.assume(cos_sin_stride_d >= 0)  # 假设余弦/正弦维度步长非负
    tl.assume(q_out_stride_t >= 0)  # 假设查询输出时间步步长非负
    tl.assume(q_out_stride_h >= 0)  # 假设查询输出头步长非负
    tl.assume(q_out_stride_d >= 0)  # 假设查询输出维度步长非负
    tl.assume(k_out_stride_t >= 0)  # 假设键输出时间步步长非负
    tl.assume(k_out_stride_h >= 0)  # 假设键输出头步长非负
    tl.assume(k_out_stride_d >= 0)  # 假设键输出维度步长非负
    tl.assume(key_cache_stride_t >= 0)  # 假设键缓存时间步步长非负
    tl.assume(key_cache_stride_h >= 0)  # 假设键缓存头步长非负
    tl.assume(key_cache_stride_d >= 0)  # 假设键缓存维度步长非负
    tl.assume(key_cache_stride_b >= 0)  # 假设键缓存块步长非负
    tl.assume(key_cache_stride_x >= 0)  # 假设键缓存x步长非负
    tl.assume(value_cache_stride_t >= 0)  # 假设值缓存时间步步长非负
    tl.assume(value_cache_stride_h >= 0)  # 假设值缓存头步长非负
    tl.assume(value_cache_stride_d >= 0)  # 假设值缓存维度步长非负
    tl.assume(value_cache_stride_b >= 0)  # 假设值缓存块步长非负
    tl.assume(value_cache_stride_slot_chunk >= 0)  # 假设值缓存槽块步长非负
    tl.assume(value_cache_stride_x >= 0)  # 假设值缓存x步长非负
    tl.assume(zeros_out_stride_t >= 0)  # 假设零输出时间步步长非负
    tl.assume(zeros_out_stride_h >= 0)  # 假设零输出头步长非负
    tl.assume(zeros_out_stride_d >= 0)  # 假设零输出维度步长非负

    # ============================================================  # ============================================================
    # Stage 1: Program instance mapping  # 阶段1：程序实例映射
    #
    # Each program handles:  # 每个程序处理
    #   - one (token, q_head) for Q path  # 一个(token, 查询头)用于Q路径
    #   - selected KV ownership for cache write path  # 选定的KV所有权用于缓存写入路径
    #
    # pid layout:  # pid布局
    #   [0, T*QH)            -> decode Q path  # [0, T*QH) -> 解码Q路径
    #   [T*QH, extra KV)     -> KV-only path  # [T*QH, 额外KV) -> 仅KV路径
    # ============================================================  # ============================================================

    pid = tl.program_id(0)  # 获取当前程序实例ID
    tl.assume(pid >= 0)  # 假设pid非负

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)  # 生成位置编码维度偏移范围

    # ============================================================  # ============================================================
    # Stage 2: Main decode path (Q always active)  # 阶段2：主解码路径（Q始终活跃）
    # ============================================================  # ============================================================

    if pid < T * QH:  # 如果在Q路径范围内
        pid_t = pid // QH  # 计算token索引
        pid_hq = pid % QH  # 计算查询头索引

        # --------------------------------------------------------  # --------------------------------------------------------
        # Stage 2.1: Compute rotary frequency offsets  # 阶段2.1：计算旋转频率偏移
        #
        # RoPE frequencies may be stored as:  # RoPE频率可能存储为
        #   D/2 frequencies (shared front-half)  # D/2个频率（共享前半部分）
        #   D frequencies (full explicit)  # D个频率（完整显式）
        # --------------------------------------------------------  # --------------------------------------------------------

        if REUSE_FREQS_FRONT_PART:  # 如果复用前半部分频率
            if IS_NEOX:  # NeoX风格
                d_cos_offs = d_pe_offs  # 初始化余弦偏移
                d_cos_offs = tl.where(  # 调整后半部分的偏移
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),  # 后半部分条件
                    d_cos_offs - BLOCK_D_HALF_pe,  # 映射到前半部分
                    d_cos_offs,  # 前半部分保持不变
                ).to(d_cos_offs.dtype)  # 转换回原始类型
                # d_cos_mask = d_cos_offs < BLOCK_D_pe  # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:  # GPT-J风格
                d_cos_offs = d_pe_offs // 2  # 每对维度共享一个频率
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe  # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:  # 不复用频率
            d_cos_offs = d_pe_offs  # 直接使用位置编码偏移
            # d_cos_mask = d_cos_offs < BLOCK_D_pe  # d_cos_mask = d_cos_offs < BLOCK_D_pe

        # --------------------------------------------------------  # --------------------------------------------------------
        # Stage 2.2: Load token position and optional offset  # 阶段2.2：加载token位置和可选偏移
        #
        # offs_ptr is used by chunked prefill / sliding-window decode.  # offs_ptr用于分块预填充/滑动窗口解码
        # --------------------------------------------------------  # --------------------------------------------------------
        pos = tl.load(pos_ptr + pid_t)  # 加载当前token的位置索引
        if HAVE_POS:  # 如果有位置偏移
            offset = tl.load(offs_ptr + pid_t)  # 加载偏移量
            pos = pos + offset  # 位置加上偏移

        # --------------------------------------------------------  # --------------------------------------------------------
        # Stage 2.3: Load cosine / sine table  # 阶段2.3：加载余弦/正弦表
        # --------------------------------------------------------  # --------------------------------------------------------
        # cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d  # cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        # cos = tl.load(cos_ptr + cos_offs)  # cos = tl.load(cos_ptr + cos_offs)
        # sin = tl.load(sin_ptr + cos_offs)  # sin = tl.load(sin_ptr + cos_offs)

        freq_dim = BLOCK_D_HALF_pe if REUSE_FREQS_FRONT_PART else BLOCK_D_pe  # 确定频率维度大小

        cos, sin = _load_cos_sin(  # 加载余弦和正弦值
            cos_sin_ptr,  # 余弦/正弦缓存指针
            pos,  # 位置索引
            d_cos_offs,  # 余弦维度偏移
            cos_sin_stride_t,  # 时间步步长
            cos_sin_stride_d,  # 维度步长
            freq_dim,  # 频率维度
        )

        # --------------------------------------------------------  # --------------------------------------------------------
        # Stage 2.4: Apply RoPE to Q  # 阶段2.4：对查询应用RoPE
        # --------------------------------------------------------  # --------------------------------------------------------
        q_ptrs = (  # 计算查询数据指针
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d  # 基地址加各维度偏移
        )
        q_pe = _unit_rope(  # 对查询应用RoPE变换
            q_ptrs,  # 查询指针
            cos,  # 余弦值
            sin,  # 正弦值
            d_pe_offs,  # 位置编码偏移
            IS_NEOX,  # 是否NeoX风格
            BLOCK_D_pe,  # 位置编码维度块大小
            BLOCK_D_HALF_pe,  # 半位置编码维度块大小
        )

        # Store rotated Q output.  # 存储旋转后的查询输出
        q_out_ptrs = (  # 计算查询输出指针
            q_out_ptr  # 查询输出基地址
            + pid_t * q_out_stride_t  # 加上token偏移
            + pid_hq * q_out_stride_h  # 加上头偏移
            + d_pe_offs * q_out_stride_d  # 加上维度偏移
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))  # 将RoPE后的查询写入输出

        if HAVE_ZEROS:  # 如果需要零输出
            z = tl.zeros((BLOCK_D_pe,), dtype=zeros_out_ptr.dtype.element_ty)  # 创建零向量
            zeros_out_ptrs = (  # 计算零输出指针
                zeros_out_ptr  # 零输出基地址
                + pid_t * zeros_out_stride_t  # 加上token偏移
                + pid_hq * zeros_out_stride_h  # 加上头偏移
                + d_pe_offs * zeros_out_stride_d  # 加上维度偏移
            )
            tl.store(zeros_out_ptrs, z)  # 写入零值

        # ========================================================  # ========================================================
        # Stage 3: KV ownership path  # 阶段3：KV所有权路径
        #
        # Only one Q group leader writes KV:  # 只有一个Q组领导者写入KV
        #   pid_hq % QH_PER_KH == 0  # pid_hq % QH_PER_KH == 0
        #
        # This prevents duplicated KV cache writes.  # 这防止重复的KV缓存写入
        # ========================================================  # ========================================================

        if pid_hq % QH_PER_KH == 0:  # 如果是Q组领导者
            # ----------------------------------------------------  # ----------------------------------------------------
            # Stage 3.1: Resolve cache slot  # 阶段3.1：解析缓存槽位
            # ----------------------------------------------------  # ----------------------------------------------------
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)  # 加载当前token的缓存槽
            if HAS_SWA:  # 如果有SWA映射
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)  # 通过SWA映射重新获取槽

            # ------------------------------------------------  # ------------------------------------------------
            # Stage 3.2: Apply RoPE to K  # 阶段3.2：对键应用RoPE
            # ------------------------------------------------  # ------------------------------------------------
            if pid_slot >= 0:  # 如果槽位有效
                pid_t_slot = pid_slot // BLOCK_SIZE  # 计算缓存块索引
                pid_b = pid_slot % BLOCK_SIZE  # 计算块内偏移
                pid_hk = pid_hq // QH_PER_KH  # 计算键头索引
                if HAVE_K_SCALE:  # 如果有键缩放
                    k_scale = tl.load(k_scale_ptr)  # 加载键缩放因子
                else:  # 否则
                    k_scale = 1  # 缩放因子为1
                k_ptrs = (  # 计算键数据指针
                    k_ptr  # 键基地址
                    + pid_t * k_stride_t  # 加上token偏移
                    + pid_hk * k_stride_h  # 加上头偏移
                    + d_pe_offs * k_stride_d  # 加上维度偏移
                )
                k_pe = _unit_rope(  # 对键应用RoPE变换
                    k_ptrs,  # 键指针
                    cos,  # 余弦值
                    sin,  # 正弦值
                    d_pe_offs,  # 位置编码偏移
                    IS_NEOX,  # 是否NeoX风格
                    BLOCK_D_pe,  # 位置编码维度块大小
                    BLOCK_D_HALF_pe,  # 半位置编码维度块大小
                )

                k_out_ptrs = (  # 计算键输出指针
                    k_out_ptr  # 键输出基地址
                    + pid_t * k_out_stride_t  # 加上token偏移
                    + pid_hk * k_out_stride_h  # 加上头偏移
                    + d_pe_offs * k_out_stride_d  # 加上维度偏移
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))  # 将RoPE后的键写入输出

                # ------------------------------------------------  # ------------------------------------------------
                # Stage 3.3: Optional fp8 scaling before cache  # 阶段3.3：缓存前可选的FP8缩放
                # ------------------------------------------------  # ------------------------------------------------

                k_scale_rcprl = 1 / k_scale  # 计算缩放因子的倒数
                k_pe = k_pe * k_scale_rcprl  # 将键乘以缩放倒数

                # ------------------------------------------------  # ------------------------------------------------
                # Stage 3.4: Write K cache  # 阶段3.4：写入键缓存
                #
                # Two layouts supported:  # 支持两种布局
                #   FLASH_LAYOUT  # Flash布局
                #   paged KV layout  # 分页KV布局
                # ------------------------------------------------  # ------------------------------------------------

                if FLASH_LAYOUT:  # 如果使用Flash布局
                    k_out_ptrs = (  # 计算Flash布局的键缓存地址
                        key_cache_ptr  # 键缓存基地址
                        + pid_t_slot * key_cache_stride_t  # 加上块偏移
                        + pid_b * key_cache_stride_b  # 加上块内偏移
                        + pid_hk * key_cache_stride_h  # 加上头偏移
                        + d_pe_offs * key_cache_stride_d  # 加上维度偏移
                    )
                else:  # 使用分页KV布局
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))  # 重塑键数据为2D
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)  # 生成维度x偏移
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)  # 生成x偏移
                    k_out_ptrs = (  # 计算分页布局的键缓存地址
                        key_cache_ptr  # 键缓存基地址
                        + pid_t_slot * key_cache_stride_t  # 加上块偏移
                        + pid_hk * key_cache_stride_h  # 加上头偏移
                        + dx_offs[:, None] * key_cache_stride_d  # 加上维度偏移
                        + pid_b * key_cache_stride_b  # 加上块内偏移
                        + x_offs[None, :] * key_cache_stride_x  # 加上x偏移
                    )

                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))  # 将键数据写入缓存

                # ------------------------------------------------  # ------------------------------------------------
                # Stage 3.5: Write V cache  # 阶段3.5：写入值缓存
                #
                # Supports:  # 支持
                #   normal layout  # 普通布局
                #   shuffle layout  # 洗牌布局
                # ------------------------------------------------  # ------------------------------------------------

                v_ptrs = (  # 计算值数据指针
                    v_ptr  # 值基地址
                    + pid_t * v_stride_t  # 加上token偏移
                    + pid_hk * v_stride_h  # 加上头偏移
                    + d_pe_offs * v_stride_d  # 加上维度偏移
                )
                if HAVE_V_SCALE:  # 如果有值缩放
                    v_scale = tl.load(v_scale_ptr)  # 加载值缩放因子
                else:  # 否则
                    v_scale = 1  # 缩放因子为1
                v_scale_rcprl = 1 / v_scale  # 计算缩放因子的倒数
                v = tl.load(v_ptrs) * v_scale_rcprl  # 加载值数据并乘以缩放倒数
                if VALUE_SHUFFLE_LAYOUT:  # 如果使用值洗牌布局
                    slot_chunk = pid_b // X_SIZE  # 计算槽块索引
                    x_off = pid_b % X_SIZE  # 计算x偏移
                    v_out_ptrs = (  # 计算洗牌布局的值缓存地址
                        value_cache_ptr  # 值缓存基地址
                        + pid_t_slot * value_cache_stride_t  # 加上块偏移
                        + pid_hk * value_cache_stride_h  # 加上头偏移
                        + slot_chunk * value_cache_stride_slot_chunk  # 加上槽块偏移
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d  # 加上维度偏移
                        + x_off * value_cache_stride_x  # 加上x偏移
                    )
                else:  # 使用普通布局
                    v_out_ptrs = (  # 计算普通布局的值缓存地址
                        value_cache_ptr  # 值缓存基地址
                        + pid_t_slot * value_cache_stride_t  # 加上块偏移
                        + pid_hk * value_cache_stride_h  # 加上头偏移
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d  # 加上维度偏移
                        + pid_b * value_cache_stride_b  # 加上块内偏移
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))  # 将值数据写入缓存
    # ============================================================  # ============================================================
    # Stage 4: Extra KV-only path  # 阶段4：额外的仅KV路径
    #
    # Handles tokens that only require cache update:  # 处理只需缓存更新的token
    #   T_slot > T  # T_slot > T
    #
    # No Q / no RoPE on Q branch.  # Q分支无查询/RoPE
    # ============================================================  # ============================================================
    else:  # 不在Q路径范围内
        pid = pid - T * QH + T * KH  # 重新映射pid到KV-only路径
        if pid < T_slot * KH:  # 如果在KV-only范围内
            pid_t = pid // KH  # 计算token索引
            pid_hk = pid % KH  # 计算键头索引
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)  # 加载缓存槽
            if HAS_SWA:  # 如果有SWA映射
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)  # 通过SWA映射重新获取槽

            if pid_slot >= 0:  # 如果槽位有效
                pid_t_slot = pid_slot // BLOCK_SIZE  # 计算缓存块索引
                pid_b = pid_slot % BLOCK_SIZE  # 计算块内偏移
                if HAVE_K_SCALE:  # 如果有键缩放
                    k_scale = tl.load(k_scale_ptr)  # 加载键缩放因子
                else:  # 否则
                    k_scale = 1  # 缩放因子为1
                k_ptrs = (  # 计算键数据指针
                    k_ptr  # 键基地址
                    + pid_t * k_stride_t  # 加上token偏移
                    + pid_hk * k_stride_h  # 加上头偏移
                    + d_pe_offs * k_stride_d  # 加上维度偏移
                )

                k_pe = tl.load(k_ptrs)  # 加载键数据（不需要RoPE）

                k_out_ptrs = (  # 计算键输出指针
                    k_out_ptr  # 键输出基地址
                    + pid_t * k_out_stride_t  # 加上token偏移
                    + pid_hk * k_out_stride_h  # 加上头偏移
                    + d_pe_offs * k_out_stride_d  # 加上维度偏移
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))  # 将键数据写入输出

                k_scale_rcprl = 1 / k_scale  # 计算缩放因子的倒数
                k_pe = k_pe * k_scale_rcprl  # 将键乘以缩放倒数

                if FLASH_LAYOUT:  # 如果使用Flash布局
                    k_out_ptrs = (  # 计算Flash布局的键缓存地址
                        key_cache_ptr  # 键缓存基地址
                        + pid_t_slot * key_cache_stride_t  # 加上块偏移
                        + d_pe_offs * key_cache_stride_d  # 加上维度偏移
                        + pid_b * key_cache_stride_b  # 加上块内偏移
                        + pid_hk * key_cache_stride_h  # 加上头偏移
                    )
                else:  # 使用分页KV布局
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))  # 重塑键数据为2D
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)  # 生成维度x偏移
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)  # 生成x偏移
                    k_out_ptrs = (  # 计算分页布局的键缓存地址
                        key_cache_ptr  # 键缓存基地址
                        + pid_t_slot * key_cache_stride_t  # 加上块偏移
                        + pid_hk * key_cache_stride_h  # 加上头偏移
                        + dx_offs[:, None] * key_cache_stride_d  # 加上维度偏移
                        + pid_b * key_cache_stride_b  # 加上块内偏移
                        + x_offs[None, :] * key_cache_stride_x  # 加上x偏移
                    )
                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))  # 将键数据写入缓存

                v_ptrs = (  # 计算值数据指针
                    v_ptr  # 值基地址
                    + pid_t * v_stride_t  # 加上token偏移
                    + pid_hk * v_stride_h  # 加上头偏移
                    + d_pe_offs * v_stride_d  # 加上维度偏移
                )
                if HAVE_V_SCALE:  # 如果有值缩放
                    v_scale = tl.load(v_scale_ptr)  # 加载值缩放因子
                else:  # 否则
                    v_scale = 1  # 缩放因子为1
                v_scale_rcprl = 1 / v_scale  # 计算缩放因子的倒数
                v = tl.load(v_ptrs) * v_scale_rcprl  # 加载值数据并乘以缩放倒数
                if VALUE_SHUFFLE_LAYOUT:  # 如果使用值洗牌布局
                    slot_chunk = pid_b // X_SIZE  # 计算槽块索引
                    x_off = pid_b % X_SIZE  # 计算x偏移
                    v_out_ptrs = (  # 计算洗牌布局的值缓存地址
                        value_cache_ptr  # 值缓存基地址
                        + pid_t_slot * value_cache_stride_t  # 加上块偏移
                        + pid_hk * value_cache_stride_h  # 加上头偏移
                        + slot_chunk * value_cache_stride_slot_chunk  # 加上槽块偏移
                        + d_pe_offs * value_cache_stride_d  # 加上维度偏移
                        + x_off * value_cache_stride_x  # 加上x偏移
                    )
                else:  # 使用普通布局
                    v_out_ptrs = (  # 计算普通布局的值缓存地址
                        value_cache_ptr  # 值缓存基地址
                        + pid_t_slot * value_cache_stride_t  # 加上块偏移
                        + pid_hk * value_cache_stride_h  # 加上头偏移
                        + d_pe_offs * value_cache_stride_d  # 加上维度偏移
                        + pid_b * value_cache_stride_b  # 加上块内偏移
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))  # 将值数据写入缓存


def fused_qk_rope_reshape_and_cache(  # 融合QK RoPE、重整形和缓存写入的主入口函数
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    v: torch.Tensor,  # 值张量
    key_cache: torch.Tensor,  # 键缓存张量
    value_cache: torch.Tensor,  # 值缓存张量
    slot_mapping: torch.Tensor,  # 槽位映射张量
    pos: torch.Tensor,  # 位置索引张量
    cos_sin: torch.Tensor,  # 余弦/正弦缓存张量
    k_scale: torch.Tensor,  # 键缩放因子张量
    v_scale: torch.Tensor,  # 值缩放因子张量
    is_neox: bool,  # 是否NeoX风格
    flash_layout: bool,  # 是否使用Flash布局
    apply_scale: bool = True,  # 是否应用缩放（默认True）
    offs: torch.Tensor = None,  # 可选的位置偏移张量
    q_out: torch.Tensor = None,  # 可选的查询输出张量
    k_out: torch.Tensor = None,  # 可选的键输出张量
    output_zeros: bool = True,  # 是否输出零张量（默认True）
    zeros_out: torch.Tensor = None,  # 可选的零输出张量
    swa_slot_mapping=None,  # 可选的SWA槽位映射
):
    """
    Perform RoPE on q and k and along the last dimension and copy k and v in to key_cache and value_cache inplace  # 对q和k在最后一维应用RoPE，并将k和v原地复制到key_cache和value_cache

    Key parameters:  # 关键参数
    - q: shape (T, QH, D).  # 查询形状 (T, QH, D)
    - k: shape (T_slot, KH, D).  # 键形状 (T_slot, KH, D)
    - v: shape (T_slot, KH, D).  # 值形状 (T_slot, KH, D)
    - if flash_layout:  # 如果使用Flash布局
    -     key_cache: shape (T_cache, block_size, KH, D).  # 键缓存形状 (T_cache, block_size, KH, D)
    -     value_cache: shape (T_cache, block_size, KH, D).  # 值缓存形状 (T_cache, block_size, KH, D)
    - else:  # 否则
    -     key_cache: shape (T_cache, KH, D // x, block_size, x).  # 键缓存形状 (T_cache, KH, D // x, block_size, x)
    -     value_cache: shape (T_cache, KH, D, block_size).  # 值缓存形状 (T_cache, KH, D, block_size)
    - slot_mapping: shape (T_slot, ).  # 槽位映射形状 (T_slot, )

    T is the number of decode tokens, T_cahce * block_size is the max number of tokens of kv_cache  # T是解码token数，T_cache * block_size是kv缓存的最大token数
    QH must be multiple of KH  # QH必须是KH的倍数

    Returns:  # 返回值
    - q_out: same shape as input q.  # 与输入q相同形状
    - k_out: same shape as input k.  # 与输入k相同形状
    - key_cache: same shape as input key_cache (inplace).  # 与输入key_cache相同形状（原地）
    - value_cache: same shape as input value_cache (inplace).  # 与输入value_cache相同形状（原地）
    - zeros_out: same shape as input q.  # 与输入q相同形状
    """

    t, qh, d = q.shape  # 获取查询的形状
    tk, kh, dk = k.shape  # 获取键的形状
    tv, vh, dv = v.shape  # 获取值的形状
    if flash_layout:  # 如果使用Flash布局
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape  # 获取键缓存的形状
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape  # 获取值缓存的形状
        value_shuffle_layout = False  # Flash布局不使用值洗牌
    else:  # 使用分页KV布局
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape  # 获取键缓存的形状
        if value_cache.ndim == 5:  # 如果值缓存是5维
            # value_cache shuffle: (num_blocks, num_kv_heads, block_size // x, head_size, x)  # 值缓存洗牌: (块数, KV头数, 块大小//x, 头维度, x)
            t_cache_v, vh_cache, slot_chunk_v, dv_cache, x_v = value_cache.shape  # 获取值缓存的形状
            value_shuffle_layout = True  # 使用值洗牌布局
            block_size_v = slot_chunk_v * x_v  # 计算块大小
            assert block_size_v == block_size and x_v == x_cache, (  # 断言块大小和x大小匹配
                f"value_cache shuffle (T,KH,block_size//x,D,x) must match key: "  # 值缓存洗牌形状必须与键匹配
                f"{block_size_v=} {block_size=} {x_v=} {x_cache=}"  # 输出不匹配的值
            )
        else:  # 值缓存是4维
            t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape  # 获取值缓存的形状
            value_shuffle_layout = False  # 不使用值洗牌布局
    (t_slot,) = slot_mapping.shape  # 获取槽位映射的形状

    assert (  # 断言token数一致
        t == tk == tv and t_slot <= tk
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"  # token数应一致
    assert (  # 断言块大小一致
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"  # 块大小应一致
    assert (  # 断言KV头数一致
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"  # KV头数应一致
    assert (  # 断言缓存块数一致
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"  # 缓存token数应一致
    if flash_layout:  # Flash布局
        assert (  # 断言维度一致
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"  # D维度应一致
    else:  # 分页布局
        assert (  # 断言维度一致
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"  # D维度应一致
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"  # x_size应为2的幂

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"  # D维度应为2的幂
    assert block_size == triton.next_power_of_2(  # 块大小应为2的幂
        block_size
    ), "block_size should be power of 2"  # 块大小应为2的幂
    assert qh % kh == 0, "Q heads must be multiple of H heads"  # Q头数必须是KV头数的倍数
    d_freq = cos_sin.shape[-1] // 2  # 计算频率维度
    assert (d_freq == d // 2) or (  # 断言频率维度正确
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"  # cos/sin最后维度应等于或半于qk最后维度
    reuse_freqs_front_part = d_freq == d // 2  # 是否复用前半部分频率

    if q_out is None:  # 如果没有提供查询输出
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)  # 创建查询输出张量

    if k_out is None:  # 如果没有提供键输出
        k_out = torch.empty((tk, kh, dk), dtype=k.dtype, device=q.device)  # 创建键输出张量

    if zeros_out is not None:  # 如果提供了零输出张量
        tz, qhz, dz = zeros_out.shape  # 获取零输出的形状
        assert (  # 断言形状匹配
            t == tz and qh == qhz and d == dz
        ), f"q and zeros shape mismatch {q.shape=} {zeros_out.shape=}"  # q和零输出形状不匹配
        output_zeros = True  # 启用零输出
    elif output_zeros:  # 如果需要零输出但未提供
        zeros_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)  # 创建零输出张量
    else:  # 不需要零输出
        zeros_out = None  # 设为None

    n_pid = t * qh + (t_slot - t) * kh if t_slot >= t else t * qh  # 计算总程序实例数
    grid = (n_pid, 1, 1)  # 设置核函数启动网格
    _fused_qk_rope_reshape_and_cache_kernel[grid](  # 启动融合核函数
        q,  # 查询张量
        k,  # 键张量
        v,  # 值张量
        pos,  # 位置索引
        cos_sin,  # 余弦/正弦缓存
        offs,  # 位置偏移
        key_cache,  # 键缓存
        value_cache,  # 值缓存
        slot_mapping,  # 槽位映射
        swa_slot_mapping,  # SWA映射
        q_out,  # 查询输出
        k_out,  # 键输出
        zeros_out,  # 零输出
        t,  # 解码token数
        t_slot,  # 槽位token数
        *q.stride(),  # 查询步长
        *k.stride(),  # 键步长
        *v.stride(),  # 值步长
        cos_sin.stride(0),  # 余弦/正弦时间步步长
        cos_sin.stride(-1),  # 余弦/正弦维度步长
        *q_out.stride(),  # 查询输出步长
        *k_out.stride(),  # 键输出步长
        key_cache.stride(0) if not flash_layout else key_cache.stride(0),  # 键缓存块步长
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),  # 键缓存头步长/块内步长
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),  # 键缓存维度步长/头步长
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),  # 键缓存块内步长/头步长
        key_cache.stride(4) if not flash_layout else 0,  # 键缓存x步长（Flash布局为0）
        value_cache.stride(0) if not flash_layout else value_cache.stride(0),  # 值缓存块步长
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),  # 值缓存头步长/块内步长
        (  # 值缓存维度步长
            value_cache.stride(3)  # 洗牌布局的维度步长
            if (not flash_layout and value_shuffle_layout)  # 非Flash且洗牌
            else (value_cache.stride(2) if not flash_layout else value_cache.stride(3))  # 其他情况
        ),
        (  # 值缓存块步长
            0  # 洗牌布局时为0
            if (not flash_layout and value_shuffle_layout)  # 非Flash且洗牌
            else (value_cache.stride(3) if not flash_layout else value_cache.stride(1))  # 其他情况
        ),
        value_cache.stride(2) if (not flash_layout and value_shuffle_layout) else 0,  # 值缓存槽块步长
        value_cache.stride(4) if (not flash_layout and value_shuffle_layout) else 0,  # 值缓存x步长
        zeros_out.stride(0) if zeros_out is not None else 0,  # 零输出时间步步长
        zeros_out.stride(1) if zeros_out is not None else 0,  # 零输出头步长
        zeros_out.stride(2) if zeros_out is not None else 0,  # 零输出维度步长
        k_scale_ptr=k_scale,  # 键缩放因子
        v_scale_ptr=v_scale,  # 值缩放因子
        QH_PER_KH=qh // kh,  # 每个KV头对应的查询头数
        QH=qh,  # 查询头数
        KH=kh,  # 键头数
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,  # 是否复用前半部分频率
        IS_NEOX=is_neox,  # 是否NeoX风格
        BLOCK_D_pe=d,  # 位置编码维度块大小
        BLOCK_D_HALF_pe=d // 2,  # 半位置编码维度块大小
        BLOCK_SIZE=block_size,  # 缓存块大小
        X_SIZE=x_cache if not flash_layout else 0,  # x维度大小
        FLASH_LAYOUT=flash_layout,  # 是否Flash布局
        VALUE_SHUFFLE_LAYOUT=value_shuffle_layout,  # 是否值洗牌布局
        HAVE_POS=(offs is not None),  # 是否有位置偏移
        HAVE_K_SCALE=(k_scale is not None and apply_scale),  # 是否有键缩放
        HAVE_V_SCALE=(v_scale is not None and apply_scale),  # 是否有值缩放
        HAVE_ZEROS=output_zeros,  # 是否有零输出
        HAS_SWA=(swa_slot_mapping is not None),  # 是否有SWA映射
        num_warps=1,  # 每个块的warp数
    )

    if zeros_out is not None:  # 如果有零输出
        return q_out.view(-1, qh * d), k_out, key_cache, value_cache, zeros_out  # 返回包含零输出的结果
    return q_out.view(-1, qh * d), k_out, key_cache, value_cache  # 返回不含零输出的结果
