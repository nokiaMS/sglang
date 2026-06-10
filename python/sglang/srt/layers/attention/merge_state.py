# 注意力层状态合并模块
# 用于合并前缀注意力（prefix）和后缀注意力（suffix）的输出状态，
# 支持CUDA优化核函数和Triton回退核函数两种实现路径
from typing import Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch库
from sgl_kernel import merge_state_v2  # 导入CUDA优化的状态合并核函数v2

from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton  # 导入Triton回退的状态合并核函数
from sglang.srt.utils import is_cuda  # 导入CUDA检测工具函数

_is_cuda = is_cuda()  # 检测当前是否为CUDA环境


# Automatically fallback to the Triton kernel in some cases  # 在某些情况下自动回退到Triton核函数
# (e.g., for AMD GPUs, when the head dimension is not a multiple  # （例如，AMD GPU、头维度不是4或8的倍数、
# of 4 or 8, and in FP8 precision)  # 以及FP8精度时）
def _supported_dtypes(o: torch.Tensor) -> bool:  # 检查数据类型是否被CUDA核函数支持
    return o.dtype in [torch.float32, torch.half, torch.bfloat16]  # 支持float32、float16、bfloat16


def _supported_headdim(o: torch.Tensor) -> bool:  # 检查头维度是否被CUDA核函数支持
    headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]  # 获取头维度大小
    if o.dtype == torch.float32:  # 如果是float32类型
        return headdim % 4 == 0  # 头维度需要是4的倍数
    return headdim % 8 == 0  # 其他类型头维度需要是8的倍数


def merge_state(  # 合并前缀和后缀注意力的输出状态
    prefix_output: torch.Tensor,  # 前缀注意力输出张量
    prefix_lse: torch.Tensor,  # 前缀注意力对数softmax输出
    suffix_output: torch.Tensor,  # 后缀注意力输出张量
    suffix_lse: torch.Tensor,  # 后缀注意力对数softmax输出
    output: Optional[torch.Tensor] = None,  # 可选的输出张量
    output_lse: Optional[torch.Tensor] = None,  # 可选的输出对数softmax张量
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 返回合并后的输出和对数softmax
    if (  # 如果满足CUDA核函数条件
        _is_cuda  # 是CUDA环境
        and _supported_dtypes(prefix_output)  # 数据类型受支持
        and _supported_headdim(prefix_output)  # 头维度受支持
    ):
        return merge_state_v2(  # 使用CUDA优化的合并核函数
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    else:  # 否则回退到Triton核函数
        # Fallback to Triton kernel  # 回退到Triton核函数
        return merge_state_triton(  # 使用Triton实现的合并核函数
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
