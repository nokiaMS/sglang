# CUTLASS MoE参数定义模块，包含MoE操作类型枚举和参数数据类
from dataclasses import dataclass  # 导入数据类装饰器
from enum import Enum, auto  # 导入枚举类型及自动值生成
from typing import Optional  # 导入可选类型提示

import torch  # 导入PyTorch


class CutlassMoEType(Enum):  # CUTLASS MoE操作类型枚举
    """
    Enum for the different types of cutlass moe operations
    that are currently supported in SGLang.
    # SGLang中当前支持的CUTLASS MoE操作类型枚举。
    """

    BlockscaledFP8 = auto()  # 分块缩放FP8类型
    BlockscaledFP4 = auto()  # 分块缩放FP4类型


@dataclass
class CutlassMoEParams:  # CUTLASS MoE参数数据类
    """
    Parameters for the cutlass moe operation.
    # CUTLASS MoE操作的参数。
    """

    #  Type as defined above  # 如上定义的类型
    cutlass_moe_type: CutlassMoEType  # CUTLASS MoE操作类型

    # Strides for activations, weights and output in logical number of elements.
    # The activations & output stride is the number of elements to the next row.
    # The weights stride is the number of elements to the next row per expert.
    # For example, if the weight is [e, n, k], then the b_stride is a tensor of
    # shape [e] with each element being k. Similarly for activations, if the
    # shape is [m, k], then the a_stride has shape [e] with each value k.
    # Similarly for output, if the output is [m, n], then the c_stride is a
    # tensor of shape [e] with each element being k.
    # 激活、权重和输出的步幅（逻辑元素数）。
    # 激活和输出的步幅是到下一行的元素数。
    # 权重的步幅是每个专家到下一行的元素数。
    # 例如，权重为[e, n, k]，则b_stride为形状[e]的张量，每个元素为k。
    # 激活形状[m, k]，则a_stride形状[e]，每个值为k。
    # 输出形状[m, n]，则c_stride形状[e]，每个元素为k。

    # Note: cutlass_fp4_group_mm is designed to accept the strides of
    # activations and weights to be the same, so it is passed in as a single
    # tensor.
    # 注意：cutlass_fp4_group_mm设计为接受激活和权重相同的步幅，因此作为单个张量传入。
    # ab_strides_13: [e] dtype: int64 [Gemm 1: Activation / Weight strides]  # Gemm 1：激活/权重步幅
    # ab_strides_2: [e] dtype: int64 [Gemm 2: Activation / Weight strides]  # Gemm 2：激活/权重步幅
    # c_strides_13: [e] dtype: int64 [Gemm 1: Output Strides]  # Gemm 1：输出步幅
    # c_strides_2: [e] dtype: int64 [Gemm 2: Output Strides]  # Gemm 2：输出步幅
    ab_strides_13: torch.Tensor  # Gemm 1的激活/权重步幅 [e]
    ab_strides_2: torch.Tensor  # Gemm 2的激活/权重步幅 [e]
    c_strides_13: torch.Tensor  # Gemm 1的输出步幅 [e]
    c_strides_2: torch.Tensor  # Gemm 2的输出步幅 [e]

    # m: Total number of tokens  # m: 令牌总数
    # n: intermediate size per partition  # n: 每分区的中间维度
    # k: hidden size per expert  # k: 每专家的隐藏维度
    # e: Number of experts  # e: 专家数
    # device: Device to run computation on and store tensors  # device: 运行计算和存储张量的设备
    m: int  # 令牌总数
    intermediate_size_per_partition: int  # 每分区的中间维度
    hidden_size: int  # 隐藏维度
    num_experts: int  # 专家数量
    device: torch.device  # 计算设备

    # Pointers container for calculating offsets of the input activations for each expert
    # a_ptrs: [e] dtype: int64
    # 用于计算每个专家输入激活偏移量的指针容器 [e] dtype: int64
    a_ptrs: torch.Tensor  # 输入激活指针容器

    # Pointers container for calculating offsets of the input weights for each expert
    # b_ptrs: [e] dtype: int64
    # 用于计算每个专家输入权重偏移量的指针容器 [e] dtype: int64
    b_ptrs: torch.Tensor  # 输入权重指针容器

    # Pointers container for calculating offsets of the output activations for each expert
    # out_ptrs: [e] dtype: int64
    # 用于计算每个专家输出激活偏移量的指针容器 [e] dtype: int64
    out_ptrs: torch.Tensor  # 输出激活指针容器
    # Pointers container for calculating offsets of the input scales for each expert
    # a_scales_ptrs: [e] dtype: int64
    # b_scales_ptrs: [e] dtype: int64
    # 用于计算每个专家输入/权重缩放偏移量的指针容器 [e] dtype: int64
    a_scales_ptrs: torch.Tensor  # 输入缩放指针容器
    b_scales_ptrs: torch.Tensor  # 权重缩放指针容器
    # Pointers for per-expert alpha values  # 逐专家alpha值的指针
    alpha_ptrs: torch.Tensor  # alpha指针容器
    # CUTLASS blockscale layouts for A and B operands  # A和B操作数的CUTLASS分块缩放布局
    layout_sfa: torch.Tensor  # A操作数缩放布局
    layout_sfb: torch.Tensor  # B操作数缩放布局

    # Offsets that mark at which token index each expert begins its computation
    # The number of tokens computed with expert E is expert_offsets[E + 1] - expert_offsets[E]
    # expert_offsets: [e+1] dtype: int32
    # 标记每个专家开始计算的令牌索引的偏移量
    # 专家E计算的令牌数为 expert_offsets[E + 1] - expert_offsets[E]
    expert_offsets: torch.Tensor  # 专家偏移量 [e+1]

    # Problem size: (num_experts, (m,2n,k)) for first GEMM
    # problem_sizes1: [e, 3] dtype: int32
    # Problem size: (num_experts, (m,n,k)) for second GEMM
    # problem_sizes2: [e, 3] dtype: int32
    # 第一个GEMM的问题尺寸：(专家数, (m,2n,k)) [e, 3]
    # 第二个GEMM的问题尺寸：(专家数, (m,n,k)) [e, 3]
    problem_sizes1: torch.Tensor  # 第一个GEMM问题尺寸
    problem_sizes2: torch.Tensor  # 第二个GEMM问题尺寸
    # Similar to expert_offsets, but for blockscales for FP4 blockscaled Group GEMM
    # 类似于expert_offsets，但用于FP4分块缩放分组GEMM的分块缩放偏移
    blockscale_offsets: Optional[torch.Tensor] = None  # 分块缩放偏移（可选）

    def __init__(  # 初始化方法，构造CUTLASS MoE参数
        self,
        cutlass_moe_type: CutlassMoEType,  # CUTLASS MoE操作类型
        device: torch.device,  # 计算设备
        num_experts: int,  # 专家数量
        intermediate_size_per_partition: int,  # 每分区的中间维度
        hidden_size: int,  # 隐藏维度
    ):
        self.cutlass_moe_type = cutlass_moe_type  # 设置MoE类型
        self.device = device  # 设置设备
        self.num_experts = num_experts  # 设置专家数
        self.intermediate_size_per_partition = intermediate_size_per_partition  # 设置中间维度
        self.hidden_size = hidden_size  # 设置隐藏维度
        self.n = self.intermediate_size_per_partition  # 中间维度简写
        self.k = self.hidden_size  # 隐藏维度简写
        self.e = self.num_experts  # 专家数简写
        self.ab_strides_13 = torch.full(  # Gemm 1的激活/权重步幅，全部填充为k
            (self.e,), self.k, dtype=torch.int64, device=self.device
        )
        self.ab_strides_2 = torch.full(  # Gemm 2的激活/权重步幅，全部填充为n
            (self.e,), self.n, dtype=torch.int64, device=self.device
        )
        self.c_strides_13 = torch.full(  # Gemm 1的输出步幅，全部填充为2*n
            (self.e,), 2 * self.n, dtype=torch.int64, device=self.device
        )
        self.c_strides_2 = torch.full(  # Gemm 2的输出步幅，全部填充为k
            (self.e,), self.k, dtype=torch.int64, device=self.device
        )
        self.expert_offsets = torch.empty(  # 专家偏移量缓冲区 [e+1]
            (self.e + 1,), dtype=torch.int32, device=self.device
        )
        self.problem_sizes1 = torch.empty(  # 第一个GEMM问题尺寸缓冲区 [e, 3]
            (self.e, 3), dtype=torch.int32, device=self.device
        )
        self.problem_sizes2 = torch.empty(  # 第二个GEMM问题尺寸缓冲区 [e, 3]
            (self.e, 3), dtype=torch.int32, device=self.device
        )
        if self.cutlass_moe_type == CutlassMoEType.BlockscaledFP4:  # 如果是FP4类型
            self.blockscale_offsets = torch.empty(  # 分配分块缩放偏移缓冲区 [e+1]
                (self.e + 1,), dtype=torch.int32, device=self.device
            )
        else:  # 否则
            self.blockscale_offsets = None  # 不需要分块缩放偏移
        self.a_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)  # 输入激活指针 [e]
        self.b_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)  # 输入权重指针 [e]
        self.out_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)  # 输出激活指针 [e]
        self.a_scales_ptrs = torch.empty(  # 输入缩放指针 [e]
            (self.e,), dtype=torch.int64, device=self.device
        )
        self.b_scales_ptrs = torch.empty(  # 权重缩放指针 [e]
            (self.e,), dtype=torch.int64, device=self.device
        )
        self.alpha_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)  # alpha指针 [e]
        self.layout_sfa = torch.empty(  # A操作数缩放布局 [e, 5]
            (self.e, 5), dtype=torch.int64, device=self.device
        )
        self.layout_sfb = torch.empty(  # B操作数缩放布局 [e, 5]
            (self.e, 5), dtype=torch.int64, device=self.device
        )

    def to_gemm1_args(self) -> dict:  # 转换为第一个GEMM的参数字典
        return {
            "ab_strides": self.ab_strides_13,  # 激活/权重步幅
            "c_strides": self.c_strides_13,  # 输出步幅
            "problem_sizes": self.problem_sizes1,  # 问题尺寸
            "expert_offsets": self.expert_offsets[:-1],  # 专家偏移（不含末尾）
            "blockscale_offsets": self.blockscale_offsets[:-1],  # 分块缩放偏移（不含末尾）
            "a_ptrs": self.a_ptrs,  # 输入激活指针
            "b_ptrs": self.b_ptrs,  # 输入权重指针
            "out_ptrs": self.out_ptrs,  # 输出激活指针
            "a_scales_ptrs": self.a_scales_ptrs,  # 输入缩放指针
            "b_scales_ptrs": self.b_scales_ptrs,  # 权重缩放指针
            "alpha_ptrs": self.alpha_ptrs,  # alpha指针
            "layout_sfa": self.layout_sfa,  # A缩放布局
            "layout_sfb": self.layout_sfb,  # B缩放布局
        }

    def to_gemm2_args(self) -> dict:  # 转换为第二个GEMM的参数字典
        return {
            "ab_strides": self.ab_strides_2,  # 激活/权重步幅
            "c_strides": self.c_strides_2,  # 输出步幅
            "problem_sizes": self.problem_sizes2,  # 问题尺寸
            "expert_offsets": self.expert_offsets[:-1],  # 专家偏移（不含末尾）
            "blockscale_offsets": self.blockscale_offsets[:-1],  # 分块缩放偏移（不含末尾）
            "a_ptrs": self.a_ptrs,  # 输入激活指针
            "b_ptrs": self.b_ptrs,  # 输入权重指针
            "out_ptrs": self.out_ptrs,  # 输出激活指针
            "a_scales_ptrs": self.a_scales_ptrs,  # 输入缩放指针
            "b_scales_ptrs": self.b_scales_ptrs,  # 权重缩放指针
            "alpha_ptrs": self.alpha_ptrs,  # alpha指针
            "layout_sfa": self.layout_sfa,  # A缩放布局
            "layout_sfb": self.layout_sfb,  # B缩放布局
        }
