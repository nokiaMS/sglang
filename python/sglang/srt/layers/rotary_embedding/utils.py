# 旋转位置编码基础操作工具模块
# 提供NeoX/GPT-J风格旋转、旋转嵌入应用、以及各平台（NPU/CPU/CUDA）的旋转位置编码应用函数
"""Primitive rotary embedding ops: _rotate_neox, _rotate_gptj, _apply_rotary_emb,
apply_rotary_pos_emb variants."""  # 基础旋转位置编码操作：_rotate_neox、_rotate_gptj、_apply_rotary_emb、apply_rotary_pos_emb变体

from __future__ import annotations  # 启用延迟注解评估

from typing import Tuple  # 类型提示：元组类型

import torch  # PyTorch深度学习框架

from sglang.srt.utils import cpu_has_amx_support, get_compiler_backend, is_cpu, is_npu  # 导入平台检测和编译器工具函数

_is_npu = is_npu()  # 检测是否为NPU(华为昇腾)平台
_is_cpu = is_cpu()  # 检测是否为CPU平台
_is_cpu_amx_available = cpu_has_amx_support()  # 检测CPU是否支持AMX指令集

if _is_npu:  # 如果是NPU平台
    import torch_npu  # 导入华为NPU扩展模块

    NPU_ROTARY_MUL_MAX_NUM_HEADS = 1000  # NPU旋转乘法操作支持的最大头数
    NPU_ROTARY_MUL_MAX_HEAD_SIZE = 896  # NPU旋转乘法操作支持的最大头维度


def rotate_neox(x: torch.Tensor) -> torch.Tensor:  # NeoX风格的旋转操作，将向量前半部分和后半部分交换并取负
    x1 = x[..., : x.shape[-1] // 2]  # 取前半部分
    x2 = x[..., x.shape[-1] // 2 :]  # 取后半部分
    return torch.cat((-x2, x1), dim=-1)  # 拼接取负后的后半部分和前半部分


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:  # GPT-J风格的旋转操作，交错交换相邻元素对并取负
    x1 = x[..., ::2]  # 取偶数索引位置的元素
    x2 = x[..., 1::2]  # 取奇数索引位置的元素
    x = torch.stack((-x2, x1), dim=-1)  # 堆叠取负后的奇数位和偶数位
    return x.flatten(-2)  # 展平最后两个维度


def apply_rotary_emb(  # 应用旋转位置编码到输入张量
    x: torch.Tensor,  # 输入张量 [num_tokens, num_heads, head_size]
    cos: torch.Tensor,  # 余弦值 [num_tokens, head_size // 2]
    sin: torch.Tensor,  # 正弦值 [num_tokens, head_size // 2]
    is_neox_style: bool,  # 是否使用NeoX风格
) -> torch.Tensor:  # 返回旋转后的张量
    """
    Args:
        x: [num_tokens, num_heads, head_size]  # 输入张量形状
        cos: [num_tokens, head_size // 2]  # 余弦值形状
        sin: [num_tokens, head_size // 2]  # 正弦值形状
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary  # 是否使用NeoX风格或GPT-J风格的旋转
            positional embeddings.  # 位置编码
    """
    cos = cos.unsqueeze(-2).to(x.dtype)  # 增加头维度并转换数据类型
    sin = sin.unsqueeze(-2).to(x.dtype)  # 增加头维度并转换数据类型
    if is_neox_style:  # NeoX风格：前后各半分开旋转
        x1, x2 = torch.chunk(x, 2, dim=-1)  # 将输入分为前后两半
    else:  # GPT-J风格：交错元素旋转
        x1 = x[..., ::2]  # 取偶数索引位置的元素
        x2 = x[..., 1::2]  # 取奇数索引位置的元素
    o1 = x1 * cos - x2 * sin  # 旋转公式第一部分
    o2 = x2 * cos + x1 * sin  # 旋转公式第二部分
    if is_neox_style:  # NeoX风格的输出拼接
        return torch.cat((o1, o2), dim=-1)  # 拼接两半
    else:  # GPT-J风格的输出拼接
        return torch.stack((o1, o2), dim=-1).flatten(-2)  # 交错堆叠后展平


# Copied from transformers  # 从transformers库复制
def rotate_half(x):  # 旋转输入的半维度
    """Rotates half the hidden dims of the input."""  # 旋转输入隐藏维度的一半
    x1 = x[..., : x.shape[-1] // 2]  # 取前半部分
    x2 = x[..., x.shape[-1] // 2 :]  # 取后半部分
    return torch.cat((-x2, x1), dim=-1)  # 拼接取负后的后半部分和前半部分


@torch.compile(dynamic=True, backend=get_compiler_backend())  # 使用torch.compile编译，启用动态形状，使用优化的编译后端
def apply_rotary_pos_emb_native(  # 原生实现的旋转位置编码应用函数
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    cos: torch.Tensor,  # 余弦值
    sin: torch.Tensor,  # 正弦值
    unsqueeze_dim=1,  # 扩展维度的位置，默认为1
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键
    orig_q_dtype = q.dtype  # 保存查询的原始数据类型
    orig_k_dtype = k.dtype  # 保存键的原始数据类型
    q, k = q.float(), k.float()  # 转换为浮点型进行计算

    # embedding is performed in float  # 嵌入在浮点精度下执行
    cos = cos.unsqueeze(unsqueeze_dim).float()  # 扩展余弦维度并转为浮点
    sin = sin.unsqueeze(unsqueeze_dim).float()  # 扩展正弦维度并转为浮点
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 应用旋转位置编码到查询
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 应用旋转位置编码到键

    q_embed = q_embed.to(orig_q_dtype)  # 转回查询的原始数据类型
    k_embed = k_embed.to(orig_k_dtype)  # 转回键的原始数据类型

    return q_embed, k_embed  # 返回旋转后的查询和键


def apply_rotary_pos_emb_npu(  # NPU平台实现的旋转位置编码应用函数
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    cos: torch.Tensor,  # 余弦值
    sin: torch.Tensor,  # 正弦值
    unsqueeze_dim=1,  # 扩展维度的位置
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的查询和键
    """Ascend implementation equivalent to apply_rotary_pos_emb_native.  # 昇腾实现，等价于apply_rotary_pos_emb_native

    Args:
        q: [num_tokens, num_heads, head_size]  # 查询张量形状
        k: [num_tokens, num_kv_heads, head_size]  # 键张量形状
        cos: [num_tokens, head_size]  # 余弦值形状
        sin: [num_tokens, head_size]  # 正弦值形状
    """
    if (  # 检查是否满足NPU操作的限制条件
        cos.dim() != 2  # 余弦维度必须为2
        or q.dim() != 3  # 查询维度必须为3
        or q.shape[1] >= NPU_ROTARY_MUL_MAX_NUM_HEADS  # 查询头数不能超过最大限制
        or q.shape[2] >= NPU_ROTARY_MUL_MAX_HEAD_SIZE  # 头维度不能超过最大限制
    ):
        # Note: num_heads and head_size of q must be less than 1000 and 896, respectively  # 注意：查询的头数和头维度必须分别小于1000和896
        return apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim)  # 不满足条件时回退到原生实现
    cos = cos.unsqueeze(unsqueeze_dim).unsqueeze(0)  # 扩展余弦维度
    sin = sin.unsqueeze(unsqueeze_dim).unsqueeze(0)  # 扩展正弦维度
    q = q.unsqueeze(0)  # 增加批次维度
    k = k.unsqueeze(0)  # 增加批次维度
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)  # 使用NPU旋转乘法操作
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)  # 使用NPU旋转乘法操作
    q_embed = q_embed.squeeze(0)  # 去除批次维度
    k_embed = k_embed.squeeze(0)  # 去除批次维度
    return q_embed, k_embed  # 返回旋转后的查询和键


if _is_npu:  # 如果是NPU平台
    apply_rotary_pos_emb = apply_rotary_pos_emb_npu  # 使用NPU实现
elif _is_cpu and _is_cpu_amx_available:  # 如果是CPU平台且支持AMX
    apply_rotary_pos_emb = torch.ops.sgl_kernel.apply_rotary_pos_emb_cpu  # 使用CPU AMX优化实现
else:  # 其他平台
    apply_rotary_pos_emb = apply_rotary_pos_emb_native  # 使用原生实现
