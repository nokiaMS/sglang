# 文件说明：FP8量化+分页KV缓存写入融合Triton内核实现
# 本文件实现了将K/V张量从BF16/FP16量化为FP8并写入分页KV缓存的融合内核
# 主要用于TRTLLM MHA后端，消除了中间FP8张量的内存开销，减少内核启动开销

"""
Fused FP8 quantization + paged KV cache write kernel for TRTLLM MHA backend. # TRTLLM MHA后端的融合FP8量化+分页KV缓存写入内核

This kernel fuses the following operations: # 本内核融合了以下操作：
1. FP8 quantization of K and V tensors (from BF16/FP16 to FP8) # 1. K和V张量的FP8量化（从BF16/FP16转为FP8）
2. Per-token or per-page scale computation # 2. 每token或每页的缩放计算
3. Writing quantized K/V to paged KV cache layout # 3. 将量化后的K/V写入分页KV缓存布局

Performance benefits: # 性能优势：
- Eliminates intermediate FP8 tensors in memory # - 消除内存中的中间FP8张量
- Reduces kernel launch overhead # - 减少内核启动开销
- Better memory bandwidth utilization # - 更好的内存带宽利用率
"""

import logging  # 导入日志模块 # 日志记录
from typing import Optional  # 导入可选类型 # 类型提示

import torch  # 导入PyTorch库 # 深度学习框架
import triton  # 导入Triton库 # GPU内核编写框架
import triton.language as tl  # 导入Triton语言并别名为tl # Triton编程语言

logger = logging.getLogger(__name__)  # 创建模块级日志器 # 模块日志记录器


@triton.jit  # Triton JIT编译装饰器 # 将函数编译为GPU内核
def _process_kv_tensor(  # 处理单个K或V张量块的函数 # 量化并写入单个KV张量块
    token_id,  # token索引 # 当前token ID
    head_block_id,  # 头块索引 # 当前头块ID
    page_id,  # 页索引 # 当前页ID
    page_offset,  # 页内偏移 # 页内偏移量
    input_ptr,  # 输入数据指针 # 输入张量基地址
    cache_ptr,  # 缓存数据指针 # KV缓存基地址
    inv_scale,  # 逆缩放因子 # 1/scale
    use_provided_scale: tl.constexpr,  # 是否使用提供的缩放因子 # 缩放因子标志
    num_kv_heads: tl.constexpr,  # KV头数量 # KV头总数
    head_dim: tl.constexpr,  # 头维度大小 # 头维度
    input_stride_token: tl.constexpr,  # 输入的token步长 # 输入token步长
    input_stride_head: tl.constexpr,  # 输入的head步长 # 输入head步长
    input_stride_dim: tl.constexpr,  # 输入的dim步长 # 输入dim步长
    cache_stride_page: tl.constexpr,  # 缓存的page步长 # 缓存page步长
    cache_stride_offset: tl.constexpr,  # 缓存的offset步长 # 缓存offset步长
    cache_stride_head: tl.constexpr,  # 缓存的head步长 # 缓存head步长
    cache_stride_dim: tl.constexpr,  # 缓存的dim步长 # 缓存dim步长
    BLOCK_HEAD: tl.constexpr,  # 头维度块大小 # 头维度分块大小
    BLOCK_DIM: tl.constexpr,  # 维度块大小 # 维度分块大小
):
    """Process a block of heads for a single K or V tensor.""" # 处理单个K或V张量的一组头 """为单个K或V张量处理一组头的块"""
    head_idx = head_block_id * BLOCK_HEAD  # 计算当前头块的起始头索引 # 头块起始索引
    num_heads_in_block = min(BLOCK_HEAD, num_kv_heads - head_idx)  # 当前块中实际有效的头数 # 有效头数量

    for dim_idx in range(0, head_dim, BLOCK_DIM):  # 按维度块遍历 # 按维度分块遍历
        num_dims_in_block = min(BLOCK_DIM, head_dim - dim_idx)  # 当前块中实际有效的维度数 # 有效维度数量

        head_offsets = head_idx + tl.arange(0, BLOCK_HEAD)  # 头偏移索引 # 头索引范围
        dim_offsets = dim_idx + tl.arange(0, BLOCK_DIM)  # 维度偏移索引 # 维度索引范围

        head_mask = head_offsets < (head_idx + num_heads_in_block)  # 头有效性掩码 # 头范围掩码
        dim_mask = dim_offsets < (dim_idx + num_dims_in_block)  # 维度有效性掩码 # 维度范围掩码

        # Load from input using 3D strides # 使用3D步长从输入加载 # 按三维步长读取输入
        input_offsets = (  # 计算输入偏移 # 输入地址偏移
            token_id * input_stride_token  # token维度偏移 # token步长偏移
            + head_offsets[:, None] * input_stride_head  # head维度偏移 # head步长偏移
            + dim_offsets[None, :] * input_stride_dim  # dim维度偏移 # dim步长偏移
        )
        mask = head_mask[:, None] & dim_mask[None, :]  # 组合掩码 # 头和维度掩码的与运算

        block = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)  # 加载输入数据块 # 读取数据块

        # Quantize to FP8 # 量化为FP8 # 转换为FP8格式
        if use_provided_scale:  # 如果使用提供的缩放因子 # 有缩放因子时
            block_fp8 = (block * inv_scale).to(tl.float8e4nv)  # 乘以逆缩放后转为FP8 # 量化：乘以1/scale后转FP8
        else:  # 不使用缩放因子 # 无缩放因子时
            block_fp8 = block.to(tl.float8e4nv)  # 直接转为FP8 # 直接类型转换

        # Write to cache at [page_id, page_offset, head, dim] # 写入缓存，布局为[页ID, 页内偏移, 头, 维度] # 按分页布局写入
        cache_offsets = (  # 计算缓存偏移 # 缓存地址偏移
            page_id * cache_stride_page  # page维度偏移 # 页步长偏移
            + page_offset * cache_stride_offset  # offset维度偏移 # 页内偏移步长
            + head_offsets[:, None] * cache_stride_head  # head维度偏移 # 头步长偏移
            + dim_offsets[None, :] * cache_stride_dim  # dim维度偏移 # 维度步长偏移
        )

        tl.store(cache_ptr + cache_offsets, block_fp8, mask=mask)  # 将FP8数据写入缓存 # 写入量化后的数据


@triton.jit  # Triton JIT编译装饰器 # 将函数编译为GPU内核
def _fused_fp8_set_kv_buffer_kernel(  # 融合FP8量化+分页KV缓存写入内核 # 量化并写入KV缓存的主内核
    # Input tensors (post-RoPE K and V in FP16/BF16) # 输入张量（RoPE后的K和V，FP16/BF16格式） # 输入K/V张量
    k_ptr,  # [num_tokens, num_kv_heads, head_dim] # Key输入指针
    v_ptr,  # [num_tokens, num_kv_heads, head_dim] # Value输入指针
    # Output KV cache buffers (FP8 paged layout) # 输出KV缓存缓冲区（FP8分页布局） # KV缓存输出
    k_cache_ptr,  # [total_slots, num_kv_heads, head_dim] # Key缓存指针
    v_cache_ptr,  # [total_slots, num_kv_heads, head_dim] # Value缓存指针
    # Cache location indices # 缓存位置索引 # token到缓存位置的映射
    cache_loc_ptr,  # [num_tokens] -> token to cache location mapping # 缓存位置索引指针
    # Pointers to scalar inverse scales (computed on GPU in wrapper) # 标量逆缩放因子指针（在包装器中GPU上计算） # 逆缩放因子
    inv_k_scale_ptr,  # pointer to 0-D tensor on GPU # K的逆缩放因子指针
    inv_v_scale_ptr,  # pointer to 0-D tensor on GPU # V的逆缩放因子指针
    use_provided_scale: tl.constexpr,  # whether to use provided scale # 是否使用提供的缩放因子
    # Tensor dimensions # 张量维度 # 各维度大小
    num_kv_heads: tl.constexpr,  # KV头数量 # KV头总数
    head_dim: tl.constexpr,  # 头维度大小 # 头维度
    page_size: tl.constexpr,  # 页大小 # 每页token数
    # Strides for K input [num_tokens, num_kv_heads, head_dim] # K输入的步长 # K输入步长
    k_stride_token: tl.constexpr,  # K的token步长 # K token步长
    k_stride_head: tl.constexpr,  # K的head步长 # K head步长
    k_stride_dim: tl.constexpr,  # K的dim步长 # K dim步长
    # Strides for K cache [total_slots, num_kv_heads, head_dim] (logically paged) # K缓存的步长（逻辑上分页） # K缓存步长
    k_cache_stride_page: tl.constexpr,  # K缓存的page步长 # K缓存page步长
    k_cache_stride_offset: tl.constexpr,  # K缓存的offset步长 # K缓存offset步长
    k_cache_stride_head: tl.constexpr,  # K缓存的head步长 # K缓存head步长
    k_cache_stride_dim: tl.constexpr,  # K缓存的dim步长 # K缓存dim步长
    # Strides for V input [num_tokens, num_kv_heads, head_dim] # V输入的步长 # V输入步长
    v_stride_token: tl.constexpr,  # V的token步长 # V token步长
    v_stride_head: tl.constexpr,  # V的head步长 # V head步长
    v_stride_dim: tl.constexpr,  # V的dim步长 # V dim步长
    # Strides for V cache [total_slots, num_kv_heads, head_dim] (logically paged) # V缓存的步长（逻辑上分页） # V缓存步长
    v_cache_stride_page: tl.constexpr,  # V缓存的page步长 # V缓存page步长
    v_cache_stride_offset: tl.constexpr,  # V缓存的offset步长 # V缓存offset步长
    v_cache_stride_head: tl.constexpr,  # V缓存的head步长 # V缓存head步长
    v_cache_stride_dim: tl.constexpr,  # V缓存的dim步长 # V缓存dim步长
    # Block sizes # 块大小 # 分块参数
    BLOCK_HEAD: tl.constexpr,  # Number of heads per block # 每个块处理的头数
    BLOCK_DIM: tl.constexpr,  # Head dimension block size # 头维度分块大小
):
    """
    Fused FP8 quantization + paged KV cache write kernel. # 融合FP8量化+分页KV缓存写入内核

    Each program processes one token-head_block-kv combination, quantizing and writing # 每个程序处理一个token-头块-KV组合，进行量化和写入
    to the appropriate page in the KV cache. # 到KV缓存中的相应页面

    Grid: (num_tokens, num_head_blocks, 2) where dim2: 0=K, 1=V # 网格：(num_tokens, num_head_blocks, 2)，dim2: 0=K, 1=V
    """
    # Get program IDs # 获取程序ID # 读取当前线程块的ID
    token_id = tl.program_id(0)  # 当前token索引 # token ID
    head_block_id = tl.program_id(1)  # 当前头块索引 # 头块ID
    kv_idx = tl.program_id(2)  # 0 for K, 1 for V # KV选择索引：0为K，1为V

    # Get cache location for this token # 获取此token的缓存位置 # 读取缓存位置
    cache_loc = tl.load(cache_loc_ptr + token_id)  # 加载缓存位置 # 读取token对应的缓存槽位

    # Compute page_id and offset within page # 计算页ID和页内偏移 # 分解为页号和页内偏移
    page_id = cache_loc // page_size  # 页ID = 缓存位置整除页大小 # 页号
    page_offset = cache_loc % page_size  # 页内偏移 = 缓存位置取模页大小 # 页内偏移

    # Select K or V based on kv_idx # 根据kv_idx选择K或V # 按索引选择处理K或V
    if kv_idx == 0:  # 处理K张量 # Key处理分支
        # Process K tensor # 处理K张量 # 处理Key
        if use_provided_scale:  # 如果使用提供的缩放因子 # 检查缩放因子
            inv_scale = tl.load(inv_k_scale_ptr)  # 加载K的逆缩放因子 # 读取K的1/scale
        else:  # 不使用缩放因子 # 无缩放因子
            inv_scale = 1.0  # 设为1.0（不缩放） # 不缩放
        _process_kv_tensor(  # 调用KV处理函数处理K # 处理K张量块
            token_id,  # token索引 # token ID
            head_block_id,  # 头块索引 # 头块ID
            page_id,  # 页索引 # 页号
            page_offset,  # 页内偏移 # 页内偏移
            k_ptr,  # K输入指针 # Key输入
            k_cache_ptr,  # K缓存指针 # Key缓存
            inv_scale,  # 逆缩放因子 # K的逆缩放因子
            use_provided_scale,  # 缩放因子标志 # 缩放标志
            num_kv_heads,  # KV头数量 # KV头数
            head_dim,  # 头维度 # 头维度
            k_stride_token,  # K的token步长 # K token步长
            k_stride_head,  # K的head步长 # K head步长
            k_stride_dim,  # K的dim步长 # K dim步长
            k_cache_stride_page,  # K缓存的page步长 # K缓存page步长
            k_cache_stride_offset,  # K缓存的offset步长 # K缓存offset步长
            k_cache_stride_head,  # K缓存的head步长 # K缓存head步长
            k_cache_stride_dim,  # K缓存的dim步长 # K缓存dim步长
            BLOCK_HEAD,  # 头维度块大小 # 头分块
            BLOCK_DIM,  # 维度块大小 # 维度分块
        )
    else:  # 处理V张量 # Value处理分支
        # Process V tensor # 处理V张量 # 处理Value
        if use_provided_scale:  # 如果使用提供的缩放因子 # 检查缩放因子
            inv_scale = tl.load(inv_v_scale_ptr)  # 加载V的逆缩放因子 # 读取V的1/scale
        else:  # 不使用缩放因子 # 无缩放因子
            inv_scale = 1.0  # 设为1.0（不缩放） # 不缩放
        _process_kv_tensor(  # 调用KV处理函数处理V # 处理V张量块
            token_id,  # token索引 # token ID
            head_block_id,  # 头块索引 # 头块ID
            page_id,  # 页索引 # 页号
            page_offset,  # 页内偏移 # 页内偏移
            v_ptr,  # V输入指针 # Value输入
            v_cache_ptr,  # V缓存指针 # Value缓存
            inv_scale,  # 逆缩放因子 # V的逆缩放因子
            use_provided_scale,  # 缩放因子标志 # 缩放标志
            num_kv_heads,  # KV头数量 # KV头数
            head_dim,  # 头维度 # 头维度
            v_stride_token,  # V的token步长 # V token步长
            v_stride_head,  # V的head步长 # V head步长
            v_stride_dim,  # V的dim步长 # V dim步长
            v_cache_stride_page,  # V缓存的page步长 # V缓存page步长
            v_cache_stride_offset,  # V缓存的offset步长 # V缓存offset步长
            v_cache_stride_head,  # V缓存的head步长 # V缓存head步长
            v_cache_stride_dim,  # V缓存的dim步长 # V缓存dim步长
            BLOCK_HEAD,  # 头维度块大小 # 头分块
            BLOCK_DIM,  # 维度块大小 # 维度分块
        )


def fused_fp8_set_kv_buffer(  # 融合FP8 KV缓存写入函数 # 量化K/V并写入分页KV缓存的主接口
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] or [num_tokens, num_kv_heads * head_dim] # Key输入张量
    v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] or [num_tokens, num_kv_heads * head_dim] # Value输入张量
    k_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim] # Key缓存
    v_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim] # Value缓存
    cache_loc: torch.Tensor,  # [num_tokens], dtype=int32 # 缓存位置索引
    k_scale: Optional[  # Key缩放因子（可选） # K的缩放因子
        float
    ] = None,  # Scalar scale (matching original set_kv_buffer signature) # 标量缩放（与原始set_kv_buffer签名一致）
    v_scale: Optional[float] = None,  # Value缩放因子（可选） # V的缩放因子
    page_size: int = 16,  # 页大小（默认16） # 每页token数
    use_triton: bool = True,  # Whether to use Triton kernel (set to False to force naive fallback) # 是否使用Triton内核（设为False强制朴素回退）
) -> None:
    """
    Python wrapper for the fused FP8 quantization + paged KV cache write kernel. # 融合FP8量化+分页KV缓存写入内核的Python包装器

    This function replicates the exact behavior of the original set_kv_buffer but with # 此函数复制了原始set_kv_buffer的确切行为，但使用了
    a fused kernel that combines FP8 quantization and cache write. # 融合内核，将FP8量化和缓存写入结合

    Args: # 参数：
        k: Key tensor after RoPE, can be 2D or 3D # RoPE后的Key张量，可以是2D或3D
        v: Value tensor, can be 2D or 3D # Value张量，可以是2D或3D
        k_cache: Paged K cache buffer in FP8 # FP8格式的分页K缓存缓冲区
        v_cache: Paged V cache buffer in FP8 # FP8格式的分页V缓存缓冲区
        cache_loc: Cache location for each token, shape [num_tokens] # 每个token的缓存位置，形状[num_tokens]
        k_scale: Optional scalar scale for K (matching original set_kv_buffer) # K的可选标量缩放（与原始set_kv_buffer一致）
        v_scale: Optional scalar scale for V (matching original set_kv_buffer) # V的可选标量缩放（与原始set_kv_buffer一致）
        page_size: Number of tokens per page # 每页的token数
        use_triton: Whether to use optimized Triton kernel # 是否使用优化的Triton内核
    """
    num_tokens = k.shape[0]  # 获取token数量 # token数

    # Step 1: Infer num_kv_heads and head_dim from cache shape # 步骤1：从缓存形状推断num_kv_heads和head_dim # 从缓存推断维度
    if k_cache.ndim == 3:  # 3D缓存布局 # 3维缓存格式
        # 3D cache layout: [total_slots, num_kv_heads, head_dim] # 3D缓存布局：[总槽位, KV头数, 头维度]
        total_slots, num_kv_heads, head_dim = k_cache.shape  # 解析缓存形状 # 读取各维度大小
        assert (  # 断言检查 # 验证页大小整除性
            total_slots % page_size == 0
        ), f"total_slots ({total_slots}) must be divisible by page_size ({page_size})"  # 总槽位数必须能被页大小整除 # 整除验证
        num_pages = total_slots // page_size  # 计算页数 # 总页数
    elif k_cache.ndim == 4:  # 4D缓存布局 # 4维缓存格式
        # 4D cache layout: [num_pages, page_size, num_kv_heads, head_dim] # 4D缓存布局：[页数, 页大小, KV头数, 头维度]
        num_pages, ps, num_kv_heads, head_dim = k_cache.shape  # 解析缓存形状 # 读取各维度大小
        assert (  # 断言检查 # 验证页大小一致性
            ps == page_size
        ), f"page_size mismatch: cache has {ps}, expected {page_size}"  # 页大小不匹配 # 错误信息
        total_slots = num_pages * page_size  # 计算总槽位数 # 总槽位数
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}, expected 3 or 4")  # 不支持的缓存维度 # 抛出异常

    # Step 2: Validate k, v shapes and normalize # 步骤2：验证k、v形状并规范化 # 验证输入形状
    # Store original 3D shape for Triton path # 保存原始3D形状供Triton路径使用 # 3D视图缓存
    k_3d = None  # K的3D视图 # K的3D视图
    v_3d = None  # V的3D视图 # V的3D视图

    if k.ndim == 3:  # 3D输入 # 3维输入格式
        # Input is [num_tokens, num_kv_heads, head_dim] # 输入为[num_tokens, num_kv_heads, head_dim]
        assert (  # 断言检查 # 验证KV头数
            k.shape[1] == num_kv_heads
        ), f"num_kv_heads mismatch: k.shape[1]={k.shape[1]} vs cache={num_kv_heads}"  # KV头数不匹配 # 错误信息
        assert (  # 断言检查 # 验证头维度
            k.shape[2] == head_dim
        ), f"head_dim mismatch: k.shape[2]={k.shape[2]} vs cache={head_dim}"  # 头维度不匹配 # 错误信息
        assert v.shape[1] == num_kv_heads and v.shape[2] == head_dim, "v shape mismatch"  # 验证V形状 # V形状验证

        # Keep 3D for Triton kernel # 保留3D供Triton内核使用 # 3D视图
        k_3d = k  # K的3D视图 # K 3D
        v_3d = v  # V的3D视图 # V 3D
        # Create 2D view for naive fallback (will be used only if use_triton=False) # 创建2D视图供朴素回退使用（仅在use_triton=False时使用）
        k_2d = k.reshape(num_tokens, num_kv_heads * head_dim)  # 重塑K为2D # K 2D视图
        v_2d = v.reshape(num_tokens, num_kv_heads * head_dim)  # 重塑V为2D # V 2D视图
    elif k.ndim == 2:  # 2D输入 # 2维输入格式
        # Input is already [num_tokens, num_kv_heads * head_dim] # 输入已经是[num_tokens, num_kv_heads * head_dim]
        assert (  # 断言检查 # 验证展平后的维度
            k.shape[1] == num_kv_heads * head_dim
        ), f"k.shape[1]={k.shape[1]} != {num_kv_heads * head_dim}"  # K展平维度不匹配 # 错误信息
        assert (  # 断言检查 # 验证V展平后的维度
            v.shape[1] == num_kv_heads * head_dim
        ), f"v.shape[1]={v.shape[1]} != {num_kv_heads * head_dim}"  # V展平维度不匹配 # 错误信息

        # Create 3D view for Triton kernel # 创建3D视图供Triton内核使用 # 3D视图
        k_3d = k.view(num_tokens, num_kv_heads, head_dim)  # K的3D视图 # K 3D
        v_3d = v.view(num_tokens, num_kv_heads, head_dim)  # V的3D视图 # V 3D
        # Keep 2D for naive # 保留2D供朴素方法使用 # 2D视图
        k_2d = k  # K的2D视图 # K 2D
        v_2d = v  # V的2D视图 # V 2D
    else:
        raise ValueError(f"Unsupported k.ndim={k.ndim}, expected 2 or 3")  # 不支持的输入维度 # 抛出异常

    # Step 3: Compute cache strides based on layout # 步骤3：根据布局计算缓存步长 # 计算缓存步长
    if k_cache.ndim == 3:  # 3D缓存布局 # 3维缓存格式
        # 3D cache: [total_slots, num_kv_heads, head_dim] # 3D缓存：[总槽位, KV头数, 头维度]
        stride_slot = k_cache.stride(0)  # 槽位步长 # slot步长
        stride_head = k_cache.stride(1)  # 头步长 # head步长
        stride_dim = k_cache.stride(2)  # 维度步长 # dim步长

        k_cache_stride_page = stride_slot * page_size  # K缓存的page步长 # K page步长
        k_cache_stride_offset = stride_slot  # K缓存的offset步长 # K offset步长
        k_cache_stride_head = stride_head  # K缓存的head步长 # K head步长
        k_cache_stride_dim = stride_dim  # K缓存的dim步长 # K dim步长

        v_stride_slot = v_cache.stride(0)  # V的槽位步长 # V slot步长
        v_stride_head = v_cache.stride(1)  # V的头步长 # V head步长
        v_stride_dim = v_cache.stride(2)  # V的维度步长 # V dim步长

        v_cache_stride_page = v_stride_slot * page_size  # V缓存的page步长 # V page步长
        v_cache_stride_offset = v_stride_slot  # V缓存的offset步长 # V offset步长
        v_cache_stride_head = v_stride_head  # V缓存的head步长 # V head步长
        v_cache_stride_dim = v_stride_dim  # V缓存的dim步长 # V dim步长
    else:  # 4D缓存布局 # 4维缓存格式
        # 4D cache: [num_pages, page_size, num_kv_heads, head_dim] # 4D缓存：[页数, 页大小, KV头数, 头维度]
        k_cache_stride_page = k_cache.stride(0)  # K缓存的page步长 # K page步长
        k_cache_stride_offset = k_cache.stride(1)  # K缓存的offset步长 # K offset步长
        k_cache_stride_head = k_cache.stride(2)  # K缓存的head步长 # K head步长
        k_cache_stride_dim = k_cache.stride(3)  # K缓存的dim步长 # K dim步长

        v_cache_stride_page = v_cache.stride(0)  # V缓存的page步长 # V page步长
        v_cache_stride_offset = v_cache.stride(1)  # V缓存的offset步长 # V offset步长
        v_cache_stride_head = v_cache.stride(2)  # V缓存的head步长 # V head步长
        v_cache_stride_dim = v_cache.stride(3)  # V缓存的dim步长 # V dim步长

    # Decide whether to use provided scale # 决定是否使用提供的缩放因子 # 缩放因子判断
    use_provided_scale = k_scale is not None and v_scale is not None  # 两个缩放因子都提供时才使用 # 缩放因子标志

    if use_triton and num_tokens > 0:  # 使用优化Triton内核且token数>0 # Triton内核路径
        # Use optimized Triton kernel # 使用优化的Triton内核 # Triton路径
        # Compute input strides for 3D k, v: [num_tokens, num_kv_heads, head_dim] # 计算3D k、v的输入步长：[num_tokens, num_kv_heads, head_dim]
        k_stride_token = k_3d.stride(0)  # K的token步长 # K token步长
        k_stride_head = k_3d.stride(1)  # K的head步长 # K head步长
        k_stride_dim = k_3d.stride(2)  # K的dim步长 # K dim步长

        v_stride_token = v_3d.stride(0)  # V的token步长 # V token步长
        v_stride_head = v_3d.stride(1)  # V的head步长 # V head步长
        v_stride_dim = v_3d.stride(2)  # V的dim步长 # V dim步长

        # Block sizes for tiling (tunable) # 分块大小（可调优） # 分块参数
        BLOCK_HEAD = min(num_kv_heads, 8)  # Process up to 8 heads at once # 一次处理最多8个头
        BLOCK_DIM = min(head_dim, 128)  # Process up to 128 dims at once # 一次处理最多128个维度

        # Compute number of head blocks # 计算头块数量 # 头分块数
        num_head_blocks = (num_kv_heads + BLOCK_HEAD - 1) // BLOCK_HEAD  # 向上取整 # 头块数量

        # Grid: (num_tokens, num_head_blocks, 2) # 网格：(num_tokens, num_head_blocks, 2)
        # - dim 0: tokens # - 维度0：token
        # - dim 1: head blocks # - 维度1：头块
        # - dim 2: K/V (0=K, 1=V) # - 维度2：K/V（0=K，1=V）
        grid = (num_tokens, num_head_blocks, 2)  # 设置内核网格 # 内核启动网格

        device = k_3d.device  # 获取设备 # 计算设备

        def _to_tensor_scale(scale):  # 将缩放值转换为0维CUDA张量 # 标量转GPU张量
            """Convert scale to 0-D CUDA tensor (accepts Python float or Tensor).""" # 将缩放值转换为0维CUDA张量（接受Python浮点数或张量）
            if isinstance(scale, torch.Tensor):  # 如果已经是张量 # 张量类型检查
                return scale.to(device=device, dtype=torch.float32)  # 转换设备和数据类型 # 设备和类型转换
            else:  # Python浮点数或numpy标量 # 标量类型
                # Python float / np scalar # Python浮点数/numpy标量 # 标量值
                return torch.tensor(float(scale), device=device, dtype=torch.float32)  # 创建0维GPU张量 # 构造GPU标量张量

        # Compute inverse scales on GPU to avoid GPU→CPU sync in CUDA graph capture. # 在GPU上计算逆缩放因子，避免CUDA图捕获时的GPU→CPU同步。
        # Previously we used float(k_scale) which triggers synchronization and fails # 之前使用float(k_scale)会触发同步并失败
        # during CUDA graph capture with cudaErrorStreamCaptureUnsupported. # 在CUDA图捕获期间出现cudaErrorStreamCaptureUnsupported错误。
        if use_provided_scale:  # 如果使用缩放因子 # 有缩放因子
            k_scale_tensor = _to_tensor_scale(k_scale)  # 将K缩放因子转为GPU张量 # K缩放GPU张量
            v_scale_tensor = _to_tensor_scale(v_scale)  # 将V缩放因子转为GPU张量 # V缩放GPU张量

            # Pure GPU scalar operation, safe for CUDA graph # 纯GPU标量操作，对CUDA图安全 # GPU端计算逆缩放
            inv_k_scale = (1.0 / k_scale_tensor).to(device=device, dtype=torch.float32)  # 计算K的逆缩放因子 # K 1/scale
            inv_v_scale = (1.0 / v_scale_tensor).to(device=device, dtype=torch.float32)  # 计算V的逆缩放因子 # V 1/scale

            inv_k_scale_ptr = inv_k_scale  # K逆缩放因子指针 # K逆缩放指针
            inv_v_scale_ptr = inv_v_scale  # V逆缩放因子指针 # V逆缩放指针
        else:  # 不使用缩放因子 # 无缩放因子
            # When use_provided_scale=False, kernel uses constant 1.0 for inv_scale. # 当use_provided_scale=False时，内核使用常量1.0作为inv_scale。
            # Triton will optimize away the tl.load() calls via constant folding. # Triton将通过常量折叠优化掉tl.load()调用。
            # We pass dummy pointers (k_3d) which won't be accessed in the kernel. # 传入虚拟指针（k_3d），内核不会访问。
            # This avoids creating new GPU tensors during CUDA graph capture. # 避免在CUDA图捕获期间创建新的GPU张量。
            inv_k_scale_ptr = k_3d  # 使用k_3d作为虚拟指针 # 虚拟K指针
            inv_v_scale_ptr = k_3d  # 使用k_3d作为虚拟指针 # 虚拟V指针

        # Launch Triton kernel # 启动Triton内核 # 调用融合内核
        _fused_fp8_set_kv_buffer_kernel[grid](  # 启动FP8融合内核 # 调用内核
            k_3d,  # K输入（3D） # Key输入
            v_3d,  # V输入（3D） # Value输入
            k_cache,  # K缓存 # Key缓存
            v_cache,  # V缓存 # Value缓存
            cache_loc,  # 缓存位置索引 # 缓存位置
            inv_k_scale_ptr,  # K逆缩放因子指针 # K逆缩放
            inv_v_scale_ptr,  # V逆缩放因子指针 # V逆缩放
            use_provided_scale,  # 缩放因子标志 # 缩放标志
            num_kv_heads,  # KV头数量 # KV头数
            head_dim,  # 头维度 # 头维度
            page_size,  # 页大小 # 页大小
            k_stride_token,  # K的token步长 # K token步长
            k_stride_head,  # K的head步长 # K head步长
            k_stride_dim,  # K的dim步长 # K dim步长
            k_cache_stride_page,  # K缓存的page步长 # K page步长
            k_cache_stride_offset,  # K缓存的offset步长 # K offset步长
            k_cache_stride_head,  # K缓存的head步长 # K head步长
            k_cache_stride_dim,  # K缓存的dim步长 # K dim步长
            v_stride_token,  # V的token步长 # V token步长
            v_stride_head,  # V的head步长 # V head步长
            v_stride_dim,  # V的dim步长 # V dim步长
            v_cache_stride_page,  # V缓存的page步长 # V page步长
            v_cache_stride_offset,  # V缓存的offset步长 # V offset步长
            v_cache_stride_head,  # V缓存的head步长 # V head步长
            v_cache_stride_dim,  # V缓存的dim步长 # V dim步长
            BLOCK_HEAD=BLOCK_HEAD,  # 头维度块大小 # 头分块
            BLOCK_DIM=BLOCK_DIM,  # 维度块大小 # 维度分块
        )
    else:  # 朴素回退实现 # 朴素路径
        # Fallback to naive implementation # 回退到朴素实现 # 使用朴素方法
        _naive_fp8_set_kv_buffer(  # 调用朴素FP8 KV缓存写入 # 朴素实现
            k_2d, v_2d, k_cache, v_cache, cache_loc, k_scale, v_scale, page_size  # 所有参数 # 全部参数
        )


def _naive_fp8_set_kv_buffer(  # 朴素FP8 KV缓存写入函数 # 不使用融合内核的简单实现
    k: torch.Tensor,  # Key张量 # Key
    v: torch.Tensor,  # Value张量 # Value
    k_cache: torch.Tensor,  # Key缓存 # Key缓存
    v_cache: torch.Tensor,  # Value缓存 # Value缓存
    cache_loc: torch.Tensor,  # 缓存位置索引 # 缓存位置
    k_scale: Optional[float],  # Key缩放因子 # K缩放
    v_scale: Optional[float],  # Value缩放因子 # V缩放
    page_size: int,  # 页大小 # 每页token数
) -> None:
    """
    Naive fallback implementation that mimics the original set_kv_buffer logic. # 模仿原始set_kv_buffer逻辑的朴素回退实现

    This directly replicates the behavior of MHATokenToKVPool.set_kv_buffer: # 直接复制MHATokenToKVPool.set_kv_buffer的行为：
    1. Apply scale (if k.dtype != cache.dtype and scale is provided) # 1. 应用缩放（如果k.dtype != cache.dtype且提供了scale）
    2. Convert to FP8 # 2. 转换为FP8
    3. Write to cache at cache_loc # 3. 在cache_loc处写入缓存

    Args: # 参数：
        k: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D # [num_tokens, num_kv_heads * head_dim]，已重塑为2D
        v: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D # [num_tokens, num_kv_heads * head_dim]，已重塑为2D
        k_cache: [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim] # Key缓存
        v_cache: Same shape as k_cache # 与k_cache形状相同 # Value缓存
        cache_loc: [num_tokens] # 缓存位置索引
        k_scale: Optional scale for K # K的可选缩放
        v_scale: Optional scale for V # V的可选缩放
        page_size: Tokens per page # 每页token数
    """
    num_tokens = k.shape[0]  # 获取token数量 # token数

    # Infer dimensions from cache # 从缓存推断维度 # 从缓存读取维度
    if k_cache.ndim == 3:  # 3D缓存 # 3维缓存
        num_kv_heads = k_cache.shape[1]  # KV头数量 # KV头数
        head_dim = k_cache.shape[2]  # 头维度 # 头维度
    elif k_cache.ndim == 4:  # 4D缓存 # 4维缓存
        num_kv_heads = k_cache.shape[2]  # KV头数量 # KV头数
        head_dim = k_cache.shape[3]  # 头维度 # 头维度
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}")  # 不支持的维度 # 抛出异常

    # Determine target dtype and storage dtype # 确定目标数据类型和存储数据类型 # 判断数据类型
    # See: python/sglang/srt/mem_cache/memory_pool.py:445-449 # 参见：memory_pool.py第445-449行
    store_dtype = k_cache.dtype  # 获取缓存的存储数据类型 # 缓存实际存储类型
    if store_dtype == torch.uint8:  # 如果缓存存储为uint8 # uint8存储格式
        # Cache is stored as uint8 for FP8 (due to index_put limitation) # 缓存以uint8存储FP8（因为index_put的限制）
        dtype = torch.float8_e4m3fn  # Logical dtype # 逻辑数据类型为FP8 E4M3
    else:
        dtype = store_dtype  # Cache dtype is the logical dtype # 缓存数据类型即逻辑数据类型

    # Replicate the original set_kv_buffer behavior # 复制原始set_kv_buffer的行为 # 模拟原始逻辑
    # See: python/sglang/srt/mem_cache/memory_pool.py:777-799 # 参见：memory_pool.py第777-799行
    if k.dtype != dtype:  # 如果输入类型与目标类型不同 # 需要量化
        # Need quantization - clone first to avoid modifying input # 需要量化 - 先克隆以避免修改输入
        k = k.clone()  # 克隆K # 复制K避免修改原数据
        v = v.clone()  # 克隆V # 复制V避免修改原数据

        if k_scale is not None:  # 如果提供了K缩放因子 # K缩放处理
            k.div_(k_scale)  # In-place division # 原地除法 # K除以缩放因子
        if v_scale is not None:  # 如果提供了V缩放因子 # V缩放处理
            v.div_(v_scale)  # In-place division # 原地除法 # V除以缩放因子

        k = k.to(dtype)  # 转换K为目标数据类型 # K类型转换
        v = v.to(dtype)  # 转换V为目标数据类型 # V类型转换

    # View FP8 as uint8 if needed (for index_put compatibility) # 如需要将FP8视为uint8（为了index_put兼容性）
    if store_dtype == torch.uint8 and dtype in (torch.float8_e5m2, torch.float8_e4m3fn):  # 缓存为uint8且逻辑类型为FP8 # FP8转uint8视图
        k = k.view(torch.uint8)  # 将K的FP8视图转为uint8 # K视图转换
        v = v.view(torch.uint8)  # 将V的FP8视图转为uint8 # V视图转换

    # Reshape from [T, H*D] to [T, H, D] # 从[T, H*D]重塑为[T, H, D] # 重塑维度
    k = k.view(num_tokens, num_kv_heads, head_dim)  # 重塑K为3D # K 3D重塑
    v = v.view(num_tokens, num_kv_heads, head_dim)  # 重塑V为3D # V 3D重塑

    # Write to cache using advanced indexing (same as original) # 使用高级索引写入缓存（与原始方法相同） # 写入缓存
    if k_cache.ndim == 3:  # 3D缓存格式 # 3维缓存写入
        # 3D cache: [total_slots, H, D] # 3D缓存：[总槽位, H, D]
        k_cache[cache_loc] = k  # 按索引写入K缓存 # K写入
        v_cache[cache_loc] = v  # 按索引写入V缓存 # V写入
    else:  # 4D缓存格式 # 4维缓存写入
        # 4D cache: [num_pages, page_size, H, D] # 4D缓存：[页数, 页大小, H, D]
        # Decompose loc into page_id and page_offset (vectorized) # 将位置分解为页ID和页内偏移（向量化）
        page_ids = cache_loc // page_size  # 计算页ID # 页号
        page_offsets = cache_loc % page_size  # 计算页内偏移 # 页内偏移
        k_cache[page_ids, page_offsets] = k  # 按页索引写入K缓存 # K写入
        v_cache[page_ids, page_offsets] = v  # 按页索引写入V缓存 # V写入
