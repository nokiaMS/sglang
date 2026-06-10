# 文件说明：合并注意力状态的Triton内核实现
# 本文件实现了将前缀（prefix）和后缀（suffix）的注意力输出和log-sum-exp值合并的功能
# 核心算法基于log-sum-exp的数值稳定合并公式

from typing import Optional, Tuple  # 导入可选类型和元组类型 # 可选类型和元组类型提示

import torch  # 导入PyTorch库 # 深度学习框架
import triton  # 导入Triton库 # GPU内核编写框架
import triton.language as tl  # 导入Triton语言并别名为tl # Triton编程语言


@triton.jit  # Triton JIT编译装饰器 # 将函数编译为GPU内核
def merge_state_kernel(  # 合并状态内核函数 # 将前缀和后缀的注意力状态合并
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_merged # 合并后的输出值张量
    output_lse,  # [NUM_TOKENS, NUM_HEADS] s_merged # 合并后的log-sum-exp张量
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_a # 前缀部分的输出值
    prefix_lse,  # [NUM_TOKENS, NUM_HEADS] s_a # 前缀部分的log-sum-exp值
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_b # 后缀部分的输出值
    suffix_lse,  # [NUM_TOKENS, NUM_HEADS] s_b # 后缀部分的log-sum-exp值
    HEAD_SIZE: tl.constexpr,  # 头维度大小（编译时常量） # 注意力头的维度
    PADDED_HEAD_SIZE: tl.constexpr,  # 填充后的头维度大小（编译时常量） # 对齐到2的幂的头维度
    OUTPUT_LSE: tl.constexpr,  # 是否输出log-sum-exp（编译时常量） # 控制是否计算并存储lse
):
    token_idx = tl.program_id(0)  # 获取当前程序在维度0上的ID（token索引） # 当前token索引
    num_tokens = tl.num_programs(0)  # 获取维度0上的程序总数（token数量） # token总数
    head_idx = tl.program_id(1)  # 获取当前程序在维度1上的ID（头索引） # 当前注意力头索引
    num_heads = tl.num_programs(1)  # 获取维度1上的程序总数（头数量） # 注意力头总数

    p_lse = tl.load(prefix_lse + token_idx * num_heads + head_idx)  # 加载前缀的log-sum-exp值 # 读取前缀lse
    s_lse = tl.load(suffix_lse + token_idx * num_heads + head_idx)  # 加载后缀的log-sum-exp值 # 读取后缀lse
    p_lse = float("-inf") if p_lse == float("inf") else p_lse  # 将inf转换为-inf（表示空状态） # 处理无穷大值
    s_lse = float("-inf") if s_lse == float("inf") else s_lse  # 将inf转换为-inf（表示空状态） # 处理无穷大值

    max_lse = tl.maximum(p_lse, s_lse)  # 计算两个lse的最大值（用于数值稳定） # 取最大lse保证数值稳定
    p_lse = p_lse - max_lse  # 减去最大值（数值稳定化） # 前缀lse减去最大值
    s_lse = s_lse - max_lse  # 减去最大值（数值稳定化） # 后缀lse减去最大值
    out_se = tl.exp(p_lse) + tl.exp(s_lse)  # 计算指数求和（归一化分母） # 计算归一化分母

    if OUTPUT_LSE:  # 如果需要输出log-sum-exp # 判断是否计算合并后的lse
        out_lse = tl.log(out_se) + max_lse  # 计算合并后的log-sum-exp # 还原对数求和的值
        tl.store(output_lse + token_idx * num_heads + head_idx, out_lse)  # 存储合并后的lse # 写入合并后的lse

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)  # 生成头维度的索引范围 # 头维度索引序列
    head_mask = head_arange < HEAD_SIZE  # 创建掩码，遮蔽填充部分 # 遮蔽超出实际头维度的部分
    p_out = tl.load(  # 加载前缀输出值 # 读取前缀注意力输出
        prefix_output  # 前缀输出基地址 # 前缀输出指针
        + token_idx * num_heads * HEAD_SIZE  # token偏移量 # token维度偏移
        + head_idx * HEAD_SIZE  # 头偏移量 # 头维度偏移
        + head_arange,  # 维度偏移量 # 头内维度偏移
        mask=head_mask,  # 应用掩码 # 应用填充掩码
    )
    s_out = tl.load(  # 加载后缀输出值 # 读取后缀注意力输出
        suffix_output  # 后缀输出基地址 # 后缀输出指针
        + token_idx * num_heads * HEAD_SIZE  # token偏移量 # token维度偏移
        + head_idx * HEAD_SIZE  # 头偏移量 # 头维度偏移
        + head_arange,  # 维度偏移量 # 头内维度偏移
        mask=head_mask,  # 应用掩码 # 应用填充掩码
    )

    p_scale = tl.exp(p_lse) / out_se  # 计算前缀的缩放因子（softmax权重） # 前缀归一化权重
    s_scale = tl.exp(s_lse) / out_se  # 计算后缀的缩放因子（softmax权重） # 后缀归一化权重
    out = p_out * p_scale + s_out * s_scale  # 加权合并前缀和后缀输出 # 加权求和得到合并结果
    tl.store(  # 存储合并结果 # 写入合并后的输出
        output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,  # 输出地址偏移 # 计算输出写入地址
        out,  # 合并后的值 # 合并结果值
        mask=head_mask,  # 应用掩码 # 应用填充掩码
    )


def merge_state_triton(  # 合并状态的Triton接口函数 # 使用Triton内核合并前缀和后缀注意力状态
    prefix_output: torch.Tensor,  # 前缀输出值张量 # 前缀注意力输出
    prefix_lse: torch.Tensor,  # 前缀log-sum-exp张量 # 前缀对数求和指数值
    suffix_output: torch.Tensor,  # 后缀输出值张量 # 后缀注意力输出
    suffix_lse: torch.Tensor,  # 后缀log-sum-exp张量 # 后缀对数求和指数值
    output: Optional[torch.Tensor] = None,  # 输出张量（可选） # 合并后的输出值
    output_lse: Optional[torch.Tensor] = None,  # 输出log-sum-exp张量（可选） # 合并后的lse
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 返回输出值和log-sum-exp的元组 # 返回合并结果
    # Avoid creating new tensors if they are already provided # 如果已提供则避免创建新张量
    if output is None:  # 如果未提供输出张量 # 检查输出是否为空
        output = torch.empty_like(prefix_output)  # 创建与前缀输出形状相同的空张量 # 分配输出内存
    if output_lse is None:  # 如果未提供输出lse张量 # 检查lse输出是否为空
        output_lse = torch.empty_like(prefix_lse)  # 创建与前缀lse形状相同的空张量 # 分配lse输出内存

    num_tokens = output.shape[0]  # 获取token数量 # token数
    num_query_heads = output.shape[1]  # 获取查询头数量 # 查询头数
    head_size = output.shape[2]  # 获取头维度大小 # 头维度
    padded_head_size = triton.next_power_of_2(head_size)  # 计算大于等于head_size的最小2的幂 # 对齐到2的幂

    merge_state_kernel[(num_tokens, num_query_heads)](  # 启动合并状态内核 # 按token和头数配置内核网格
        output,  # 输出值张量 # 合并输出
        output_lse,  # 输出lse张量 # 合并lse
        prefix_output,  # 前缀输出值 # 前缀注意力输出
        prefix_lse,  # 前缀lse # 前缀lse值
        suffix_output,  # 后缀输出值 # 后缀注意力输出
        suffix_lse,  # 后缀lse # 后缀lse值
        head_size,  # 头维度大小 # 头维度
        padded_head_size,  # 填充后的头维度大小 # 对齐后的头维度
        output_lse is not None,  # 是否输出lse # 控制lse输出标志
    )
    return output, output_lse  # 返回合并后的输出和lse # 返回合并结果
