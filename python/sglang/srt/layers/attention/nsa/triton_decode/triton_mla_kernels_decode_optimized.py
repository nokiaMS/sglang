# DeepSeek V4 的优化 Triton MLA 解码内核模块
# 本模块提供优化的稀疏注意力解码，减少 Python 开销
# 关键优化：
# 1. 融合的 gather+反量化+注意力内核（消除中间缓冲区）
# 2. Split-K 用于小批量下更好的 GPU 并行性
# 3. 预分配的缓冲池用于 splitk 中间结果
# 4. 预计算的步长以减少张量元数据操作
# 注意：此实现假设 KV 缓存始终是 FP8 量化的
"""
Optimized Triton MLA Decode Kernels for DeepSeek V4. # DeepSeek V4 的优化 Triton MLA 解码内核

This module provides optimized sparse attention decode with reduced Python overhead. # 本模块提供优化的稀疏注意力解码，减少 Python 开销

Key optimizations: # 关键优化：
1. Fused gather+dequant+attention kernels (eliminates intermediate buffers) # 1. 融合的 gather+反量化+注意力内核（消除中间缓冲区）
2. Split-K for better GPU parallelism on small batches # 2. Split-K 用于小批量下更好的 GPU 并行性
3. Pre-allocated buffer pool for splitk intermediate results # 3. 预分配的缓冲池用于 splitk 中间结果
4. Pre-computed strides to reduce tensor metadata operations # 4. 预计算的步长以减少张量元数据操作

Note: This implementation assumes KV cache is always FP8 quantized. # 注意：此实现假设 KV 缓存始终是 FP8 量化的
"""

from typing import Optional, Tuple # 导入类型提示

import torch # 导入 PyTorch
import triton # 导入 Triton

from .triton_mla_kernels_decode_common import ( # 从公共模块导入
    _bucket_total_tokens, # 分桶函数
    _unified_sparse_decode_kernel, # 统一稀疏解码内核
    compute_token_ranges, # 计算令牌范围
)
from .triton_mla_kernels_decode_dsv4 import ( # 从 DSV4 模块导入
    DSV4_D_QK, # DSV4 QK 维度常量
    fused_gather_dequant_fp8_dsv4, # 融合 gather+反量化函数
)
from .triton_mla_kernels_decode_fused import ( # 从融合模块导入
    fused_gather_attn_decode_dsv4, # 单作用域融合注意力
    fused_gather_attn_decode_dsv4_dual_scope, # 双作用域融合注意力
    fused_gather_attn_decode_dsv4_dual_scope_low_overhead, # 低开销双作用域融合注意力
)


def triton_sparse_attn_decode( # 优化的稀疏注意力解码入口函数
    q: torch.Tensor, # 查询张量
    kv_scope, # KV 作用域
    extra_kv_scope, # 额外 KV 作用域
    sm_scale: float, # softmax 缩放因子
    d_v: int = 512, # 值维度，默认512
    attn_sink: Optional[torch.Tensor] = None, # 可选的注意力汇聚值
) -> Tuple[torch.Tensor, torch.Tensor]: # 返回输出和 LSE
    """Optimized sparse attention decode for DeepSeek V4 (d_qk=512).""" # DeepSeek V4 (d_qk=512) 的优化稀疏注意力解码
    d_qk = q.shape[-1] # 获取 QK 维度

    if d_qk != DSV4_D_QK: # 验证 QK 维度
        raise ValueError( # 抛出错误
            f"Unsupported d_qk: {d_qk}. Expected {DSV4_D_QK} (DeepSeek V4)" # 不支持的 QK 维度
        )

    return _triton_sparse_attn_decode_dsv4( # 调用 DSV4 实现
        q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
    )


def _should_use_fused_dual_scope(total_tokens: int, h_q: int, total_topk: int) -> bool: # 判断是否应为双作用域使用融合内核
    """Determine whether to use fused kernel for dual-scope cases. # 判断是否应为双作用域使用融合内核

    Returns True if the fused kernel (with splitk for small bs) should be used. # 如果应使用融合内核（小批量带 splitk），返回 True
    For large batch sizes (>= 256), use _should_use_fused_nosplitk instead. # 大批量（>= 256）时使用 _should_use_fused_nosplitk

    The thresholds below were determined empirically on MI355X (256 CUs). # 以下阈值在 MI355X（256 CUs）上经验确定
    """
    if total_tokens <= 4: # 极小批量
        return True # 始终使用融合
    if h_q <= 64 and total_topk <= 800: # 少头少 topk
        return total_tokens <= 256 # 中小批量使用融合
    if h_q <= 64 and total_topk >= 1024: # 少头大 topk
        return total_tokens <= 128 # 小批量使用融合
    # h_q > 64 (e.g. h_q=128 when q is padded to full n_heads). # 多头情况（如 h_q=128 时查询填充到完整头数）
    if h_q > 64: # 多头
        if total_topk >= 400: # 大 topk
            return total_tokens <= 32 # 极小批量
        else: # 小 topk
            return total_tokens <= 128 # 小批量
    return True # 默认使用融合


def _should_use_fused_nosplitk(total_tokens: int, h_q: int, total_topk: int) -> bool: # 判断是否应为大批量使用无 splitk 的融合内核
    """Determine whether to use the fused no-splitk kernel for large batches. # 判断是否应为大批量使用无 splitk 的融合内核

    Kernel-level benchmarking on MI355X shows that for large batch sizes # MI355X 上的内核级基准测试表明，对于大批量
    (total_tokens >= 256), the fused dual-scope kernel WITHOUT split-K # （total_tokens >= 256），无 Split-K 的融合双作用域内核
    is ~10% faster than the separate gather+attention path: # 比分离的 gather+注意力路径快约 10%：

      total_tokens=256:  fused-noSK=169us vs separate=194us (14% faster) # 256令牌：融合-无SK=169us vs 分离=194us（快14%）
      total_tokens=512:  fused-noSK=350us vs separate=408us (14% faster) # 512令牌：融合-无SK=350us vs 分离=408us（快14%）
      total_tokens=1024: fused-noSK=700us vs separate=777us (10% faster) # 1024令牌：融合-无SK=700us vs 分离=777us（快10%）
      total_tokens=4096: fused-noSK=2761us vs separate=3063us (10% faster) # 4096令牌：融合-无SK=2761us vs 分离=3063us（快10%）

    The fused no-splitk kernel avoids: # 融合无 splitk 内核避免了：
    1. Materializing the large intermediate gathered_kv buffer # 1. 物化大型中间 gathered_kv 缓冲区
    2. The separate gather kernel launch # 2. 分离的 gather 内核启动
    3. The split-K combine overhead # 3. Split-K 合并开销

    For total_tokens < 256, the separate path is faster because the # total_tokens < 256 时，分离路径更快，因为
    fused kernel has insufficient parallelism. # 融合内核并行度不足

    For extend (total_tokens >= 1024), the fused kernel always wins # 扩展时（total_tokens >= 1024），融合内核始终胜出
    regardless of h_q or total_topk because: # 无论 h_q 或 total_topk 如何，因为：
    - The grid already has thousands of blocks (good GPU utilization) # - 网格已有数千个块（良好的 GPU 利用率）
    - It eliminates 1.5-5 GB gathered_kv buffer allocation # - 消除了 1.5-5 GB 的 gathered_kv 缓冲区分配
    - It eliminates 2x gather_dequant kernel launches (~414 us) # - 消除了 2 次 gather_dequant 内核启动（约 414 us）
    - It avoids chunking that TP>1 configs require with the separate path # - 避免了分离路径在 TP>1 配置下需要的分块
    """
    if total_tokens >= 1024: # 扩展模式
        return True # 始终使用融合
    if h_q <= 64: # 少头
        return False  # Not benchmarked for h_q <= 64 # 未对 h_q <= 64 进行基准测试
    if total_topk < 200: # 小 topk
        return False  # Small topk doesn't benefit # 小 topk 无收益
    # For h_q > 64 and total_topk >= 200: # h_q > 64 且 total_topk >= 200 时
    # Fused no-splitk wins for total_tokens >= 256 # total_tokens >= 256 时融合无 splitk 胜出
    return total_tokens >= 256 # 返回是否使用


def _triton_sparse_attn_decode_dsv4( # DSV4 优化稀疏注意力解码的内部实现
    q: torch.Tensor, # 查询张量
    kv_scope, # KV 作用域
    extra_kv_scope, # 额外 KV 作用域
    sm_scale: float, # softmax 缩放因子
    d_v: int, # 值维度
    attn_sink: Optional[torch.Tensor], # 可选的注意力汇聚值
) -> Tuple[torch.Tensor, torch.Tensor]: # 返回输出和 LSE
    """Optimized sparse attention decode for DeepSeek V4 (d_qk=512).""" # DeepSeek V4 (d_qk=512) 的优化稀疏注意力解码
    b, s_q, h_q, d_qk = q.shape # 获取查询维度
    total_tokens = b * s_q # 总令牌数
    device = q.device # 设备

    topk_main = kv_scope.indices_in_kvcache.shape[-1] # 主 topk 数量
    kv_quantized_main = kv_scope.blocked_k_quantized # 主量化 KV
    block_size_main = kv_scope.blocked_k.shape[1] # 主块大小

    # Single scope case # 单作用域情况
    if extra_kv_scope is None: # 无额外 KV 作用域
        if topk_main < 8192: # 小 topk
            q_reshaped = q.reshape(total_tokens, h_q, d_qk) # 重塑查询
            if not q_reshaped.is_contiguous(): # 确保连续
                q_reshaped = q_reshaped.contiguous() # 使其连续

            indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main) # 重塑主索引
            if not indices_main.is_contiguous(): # 确保连续
                indices_main = indices_main.contiguous() # 使其连续

            output, lse = fused_gather_attn_decode_dsv4( # 调用单作用域融合注意力
                q_reshaped, # 查询
                kv_quantized_main, # 量化 KV
                indices_main, # 索引
                block_size_main, # 块大小
                sm_scale, # softmax 缩放
                topk_length=kv_scope.topk_length, # topk 长度
                attn_sink=attn_sink, # 注意力汇聚
                s_q=s_q, # 序列长度
            )
            return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2) # 返回
        else: # 大 topk
            from .triton_mla_kernels_decode_dsv4 import triton_sparse_attn_decode_dsv4 # 导入 DSV4 解码

            return triton_sparse_attn_decode_dsv4( # 回退到 DSV4 原始实现
                q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
            )

    # Dual scope case # 双作用域情况
    topk_extra = extra_kv_scope.indices_in_kvcache.shape[-1] # 额外 topk 数量
    total_topk = topk_main + topk_extra # 总 topk 数量

    # For large batch sizes, use fused no-splitk kernel (10% faster than separate). # 大批量使用融合无 splitk 内核（比分離快10%）
    # This check is BEFORE the chunking check because the fused kernel does NOT # 此检查在分块检查之前，因为融合内核不会
    # allocate the intermediate gathered_kv buffer, so buffer size limits don't apply. # 分配中间 gathered_kv 缓冲区，因此缓冲区大小限制不适用
    if _should_use_fused_nosplitk(total_tokens, h_q, total_topk): # 判断是否使用无 splitk 融合
        q_reshaped = q.reshape(total_tokens, h_q, d_qk).contiguous() # 重塑并确保连续

        indices_main = kv_scope.indices_in_kvcache.reshape( # 重塑主索引
            total_tokens, topk_main
        ).contiguous() # 确保连续

        block_size_extra = extra_kv_scope.blocked_k.shape[1] # 额外块大小
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape( # 重塑额外索引
            total_tokens, topk_extra
        ).contiguous() # 确保连续

        output, lse = fused_gather_attn_decode_dsv4_dual_scope( # 调用双作用域融合注意力
            q_reshaped, # 查询
            kv_quantized_main, # 主量化 KV
            indices_main, # 主索引
            block_size_main, # 主块大小
            extra_kv_scope.blocked_k_quantized, # 额外量化 KV
            indices_extra, # 额外索引
            block_size_extra, # 额外块大小
            sm_scale, # softmax 缩放
            topk_length_main=kv_scope.topk_length, # 主 topk 长度
            topk_length_extra=extra_kv_scope.topk_length, # 额外 topk 长度
            attn_sink=attn_sink, # 注意力汇聚
            s_q=s_q, # 序列长度
            force_no_splitk=True, # 强制不使用 splitk
        )
        return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2) # 返回

    # Check if chunking needed for separate path (fall back to original implementation) # 检查分离路径是否需要分块（回退到原始实现）
    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk) # 计算令牌范围
    if len(token_ranges) > 1: # 需要分块
        from .triton_mla_kernels_decode_dsv4 import triton_sparse_attn_decode_dsv4 # 导入 DSV4 解码

        return triton_sparse_attn_decode_dsv4( # 回退到 DSV4 原始实现
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
        )

    # Use fused dual-scope kernel with low-overhead buffer pool # 使用带低开销缓冲池的融合双作用域内核
    if _should_use_fused_dual_scope(total_tokens, h_q, total_topk): # 判断是否使用融合双作用域
        q_reshaped = q.reshape(total_tokens, h_q, d_qk).contiguous() # 重塑并确保连续

        indices_main = kv_scope.indices_in_kvcache.reshape( # 重塑主索引
            total_tokens, topk_main
        ).contiguous() # 确保连续

        block_size_extra = extra_kv_scope.blocked_k.shape[1] # 额外块大小
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape( # 重塑额外索引
            total_tokens, topk_extra
        ).contiguous() # 确保连续

        output, lse = fused_gather_attn_decode_dsv4_dual_scope_low_overhead( # 调用低开销双作用域融合注意力
            q_reshaped, # 查询
            kv_quantized_main, # 主量化 KV
            indices_main, # 主索引
            block_size_main, # 主块大小
            extra_kv_scope.blocked_k_quantized, # 额外量化 KV
            indices_extra, # 额外索引
            block_size_extra, # 额外块大小
            sm_scale, # softmax 缩放
            topk_length_main=kv_scope.topk_length, # 主 topk 长度
            topk_length_extra=extra_kv_scope.topk_length, # 额外 topk 长度
            attn_sink=attn_sink, # 注意力汇聚
            s_q=s_q, # 序列长度
        )
        return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2) # 返回

    # Fallback: Separate gather + attention path # 回退：分离的 gather + 注意力路径
    return _fallback_gather_attention( # 调用回退函数
        q, # 查询
        kv_scope, # KV 作用域
        extra_kv_scope, # 额外 KV 作用域
        sm_scale, # softmax 缩放
        d_v, # 值维度
        attn_sink, # 注意力汇聚
        total_tokens, # 总令牌数
        h_q, # 头数
        d_qk, # QK 维度
        topk_main, # 主 topk
        topk_extra, # 额外 topk
        block_size_main, # 主块大小
        kv_quantized_main, # 主量化 KV
        fused_gather_dequant_fp8_dsv4, # 融合 gather+反量化函数
    )


def _fallback_gather_attention( # 回退路径：分离的 gather + 注意力内核
    q: torch.Tensor, # 查询张量
    kv_scope, # KV 作用域
    extra_kv_scope, # 额外 KV 作用域
    sm_scale: float, # softmax 缩放因子
    d_v: int, # 值维度
    attn_sink: Optional[torch.Tensor], # 注意力汇聚
    total_tokens: int, # 总令牌数
    h_q: int, # 头数
    d_qk: int, # QK 维度
    topk_main: int, # 主 topk 数
    topk_extra: int, # 额外 topk 数
    block_size_main: int, # 主块大小
    kv_quantized_main, # 主量化 KV
    fused_gather_fn, # 融合 gather 函数
) -> Tuple[torch.Tensor, torch.Tensor]: # 返回输出和 LSE
    """Fallback path: separate gather + attention kernels.""" # 回退路径：分离的 gather + 注意力内核
    b = q.shape[0] # 批次大小
    s_q = q.shape[1] # 序列长度
    device = q.device # 设备
    total_topk = topk_main + topk_extra # 总 topk 数

    gathered_kv = torch.empty( # 分配收集后的 KV 缓冲区
        total_tokens, total_topk, d_qk, dtype=torch.bfloat16, device=device # BF16 类型
    )
    invalid_mask = torch.empty( # 分配无效掩码缓冲区
        total_tokens, total_topk, dtype=torch.bool, device=device # bool 类型
    )
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device) # 输出缓冲区
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device) # LSE 缓冲区

    indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main) # 重塑主索引
    block_size_extra = extra_kv_scope.blocked_k.shape[1] # 额外块大小
    indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra) # 重塑额外索引

    fused_gather_fn( # 调用融合 gather+反量化
        kv_quantized_main, # 主量化 KV
        indices_main, # 主索引
        block_size_main, # 主块大小
        kv_scope.topk_length, # 主 topk 长度
        extra_kv_scope.blocked_k_quantized, # 额外量化 KV
        indices_extra, # 额外索引
        block_size_extra, # 额外块大小
        extra_kv_scope.topk_length, # 额外 topk 长度
        gathered_kv, # 输出 KV
        invalid_mask, # 输出掩码
        s_q, # 序列长度
    )

    if q.dtype == torch.bfloat16 and q.is_contiguous(): # 如果已是 BF16 且连续
        q_reshaped = q.view(total_tokens, h_q, d_qk) # 直接视图
    else: # 否则需要转换
        q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk) # 转换并重塑
        if not q_reshaped.is_contiguous(): # 确保连续
            q_reshaped = q_reshaped.contiguous() # 使其连续

    HAS_ATTN_SINK = attn_sink is not None # 是否有注意力汇聚
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1] # 汇聚张量或占位

    grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"])) # 网格大小
    _unified_sparse_decode_kernel[grid]( # 启动统一稀疏解码内核
        q_reshaped, # 查询
        gathered_kv, # 收集的 KV
        invalid_mask, # 无效掩码
        attn_sink_tensor, # 注意力汇聚
        output, # 输出
        lse, # LSE
        sm_scale, # softmax 缩放
        total_tokens, # 总令牌数
        _bucket_total_tokens(total_tokens), # 分桶令牌数
        h_q, # 头数
        total_topk, # 总 topk
        d_qk, # QK 维度
        d_v, # 值维度
        q_reshaped.stride(0), # 查询令牌步长
        q_reshaped.stride(1), # 查询头步长
        q_reshaped.stride(2), # 查询维度步长
        gathered_kv.stride(0), # KV 令牌步长
        gathered_kv.stride(1), # KV topk 步长
        gathered_kv.stride(2), # KV 维度步长
        invalid_mask.stride(0), # 掩码令牌步长
        invalid_mask.stride(1), # 掩码 topk 步长
        output.stride(0), # 输出令牌步长
        output.stride(1), # 输出头步长
        output.stride(2), # 输出维度步长
        lse.stride(0), # LSE令牌步长
        lse.stride(1), # LSE头步长
        HAS_ATTN_SINK=HAS_ATTN_SINK, # 是否有注意力汇聚
    )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2) # 重塑并返回
