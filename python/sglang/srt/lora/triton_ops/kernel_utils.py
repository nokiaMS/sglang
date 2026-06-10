# LoRA Triton内核的工具函数
# 提供token位置解析等辅助功能，供其他LoRA Triton内核共享使用

import triton  # 导入Triton编译框架
import triton.language as tl  # 导入Triton语言模块


@triton.jit  # Triton JIT编译装饰器
def _resolve_token_positions(  # 解析token物理位置的辅助函数
    sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER: tl.constexpr  # 参数：排序token ID、段起始、偏移、段长度、排序标志
):
    """Map logical segment offsets to physical token positions.
    将逻辑段偏移映射到物理token位置。

    When SORTED_BY_ADAPTER is True, segments are grouped by adapter and
    sorted_token_ids provides the indirection to the original token rows.
    When False, tokens are already contiguous starting at seg_start.
    当SORTED_BY_ADAPTER为True时，段按适配器分组，
    sorted_token_ids提供到原始token行的间接映射。
    当为False时，token已经从seg_start开始连续排列。
    """
    if SORTED_BY_ADAPTER:  # 如果按适配器排序
        return tl.load(  # 通过查找表获取物理位置
            sorted_token_ids + seg_start + s_offset, mask=s_offset < seg_len  # 使用掩码防止越界访问
        ).to(tl.int64)  # 转换为64位整数类型
    return (seg_start + s_offset).to(tl.int64)  # 未排序时直接计算物理位置并转换为64位整数
