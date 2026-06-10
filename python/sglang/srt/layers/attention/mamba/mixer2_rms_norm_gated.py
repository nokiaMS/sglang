# Mixer2 RMS归一化门控层实现
# 该模块实现了Mixer2架构中的RMS归一化与门控机制，
# 支持张量并行，并提供原生PyTorch和CUDA两种前向传播实现。

from typing import Union  # 导入类型联合类型

import torch  # 导入PyTorch库

from sglang.srt.distributed.communication_op import (  # 从分布式通信模块导入集合通信操作
    tensor_model_parallel_all_gather,  # 张量并行全收集操作
    tensor_model_parallel_all_reduce,  # 张量并行全归约操作
)
from sglang.srt.distributed.parallel_state import (  # 从并行状态模块导入并行信息
    get_tensor_model_parallel_rank,  # 获取当前张量并行秩
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.layers.attention.fla.layernorm_gated import rms_norm_gated  # 导入FLA的RMS门控归一化函数
from sglang.srt.layers.utils import MultiPlatformOp  # 导入多平台操作基类
from sglang.srt.model_loader.weight_utils import sharded_weight_loader  # 导入分片权重加载器
from sglang.srt.utils.common import set_weight_attrs  # 导入设置权重属性的工具函数


class Mixer2RMSNormGated(MultiPlatformOp):  # Mixer2 RMS门控归一化层，继承多平台操作基类
    def __init__(  # 初始化函数
        self,
        full_hidden_size: int,  # 完整隐藏层大小
        full_n_groups: int,  # 完整分组数
        use_rms_norm: bool = True,  # 是否使用RMS归一化，默认为True
        eps: float = 1e-6,  # 方差epsilon，防止除零，默认1e-6
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        self.tp_rank = get_tensor_model_parallel_rank()  # 获取当前张量并行秩
        self.full_hidden_size = full_hidden_size  # 保存完整隐藏层大小
        self.group_size = full_hidden_size // full_n_groups  # 计算每个分组的大小
        self.per_rank_hidden_size = full_hidden_size // self.tp_size  # 计算每个并行秩的隐藏层大小
        self.n_groups = full_hidden_size // self.group_size  # 计算分组数

        self.variance_epsilon = eps  # 保存方差epsilon值
        self.use_rms_norm = use_rms_norm  # 保存是否使用RMS归一化标志
        if self.use_rms_norm:  # 如果使用RMS归一化
            # Register norm weight only if we're actually applying RMSNorm  # 仅在实际应用RMS归一化时注册归一化权重
            self.weight = torch.nn.Parameter(torch.ones(self.per_rank_hidden_size))  # 创建归一化权重参数，初始化为1
            set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader(0)})  # 设置权重属性，使用第0维分片加载
        else:  # 如果不使用RMS归一化
            # Avoid checkpoint mismatch by skipping unused parameter  # 通过跳过未使用的参数来避免检查点不匹配
            self.register_parameter("weight", None)  # 注册空权重参数
        assert (  # 断言检查
            self.full_hidden_size % self.tp_size == 0  # 隐藏层大小必须能被张量并行世界大小整除
        ), "Tensor parallel world size must divide hidden size."  # 错误提示信息

    def forward_native(  # 原生PyTorch前向传播函数
        self,
        x: torch.Tensor,  # 输入张量
        gate: torch.Tensor,  # 门控张量
    ):
        # Three tensor-parallel cases:  # 三种张量并行情况：
        #   1. n_groups is 1  #   1. 分组数为1
        #      In this case we parallelize along the reduction dim.  #      这种情况下沿归约维度并行
        #      Each rank computes a local sum of squares followed by AllReduce  #      每个秩计算局部平方和，然后进行全归约
        #   2. tp_size divides n_groups  #   2. 张量并行大小能整除分组数
        #      Each rank only reduces within its local group(s).  #      每个秩仅在其本地分组内归约
        #      No collective ops necessary.  #      不需要集合通信操作
        #   3. The general case can be pretty complicated so we AllGather  #   3. 一般情况比较复杂，因此使用全收集
        #      the input and then redundantly compute the RMSNorm.  #      输入，然后冗余计算RMS归一化
        input_dtype = x.dtype  # 保存输入数据类型
        x = x * torch.nn.functional.silu(gate.to(torch.float32))  # 将门控信号通过SiLU激活后与输入相乘
        if not self.use_rms_norm:  # 如果不使用RMS归一化
            return x.to(input_dtype)  # 直接返回转换类型后的结果

        if self.n_groups == 1:  # 情况1：分组数为1
            if self.tp_size > 1:  # 如果张量并行度大于1
                # Compute local sum and then reduce to obtain global sum  # 计算局部和，然后归约得到全局和
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)  # 计算局部平方和
                global_sums = tensor_model_parallel_all_reduce(local_sums)  # 全归约得到全局平方和
                # Calculate the variance  # 计算方差
                count = self.tp_size * x.shape[-1]  # 计算总元素数
                variance = global_sums / count  # 计算方差

            else:  # 张量并行度为1，不需要通信
                variance = x.pow(2).mean(-1, keepdim=True)  # 直接计算方差
            x = x * torch.rsqrt(variance + self.variance_epsilon)  # 应用RMS归一化
        else:  # 分组数大于1的情况
            redundant_tp: bool = self.n_groups % self.tp_size != 0  # 判断是否需要冗余张量并行
            if redundant_tp:  # 如果需要冗余计算
                # To handle the general case, redundantly apply the variance  # 处理一般情况，冗余应用方差计算
                x = tensor_model_parallel_all_gather(x, -1)  # 全收集输入张量

            *prefix_dims, hidden_dim = x.shape  # 解析张量形状，获取前缀维度和隐藏维度
            group_count = hidden_dim // self.group_size  # 计算分组数量
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)  # 将张量重塑为分组形式
            variance = x_grouped.pow(2).mean(-1, keepdim=True)  # 计算每个分组的方差
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)  # 对每个分组应用RMS归一化
            x = x_grouped.view(*prefix_dims, hidden_dim)  # 将张量重塑回原始形状

            if redundant_tp:  # 如果使用了冗余张量并行
                start = self.per_rank_hidden_size * self.tp_rank  # 计算当前秩的起始索引
                end = start + self.per_rank_hidden_size  # 计算当前秩的结束索引
                x = x[..., start:end]  # 截取当前秩对应的部分

        return self.weight * x.to(input_dtype)  # 应用权重并转换回原始数据类型

    def forward_cuda(  # CUDA前向传播函数
        self,
        x: torch.Tensor,  # 输入张量
        gate: torch.Tensor,  # 门控张量
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # 返回张量或张量元组
        input_dtype = x.dtype  # 保存输入数据类型
        if not self.use_rms_norm:  # 如果不使用RMS归一化
            # Keep gate in float32 for numerical stability during silu  # 在SiLU计算中保持门控为float32以确保数值稳定性
            return x * torch.nn.functional.silu(gate.to(torch.float32)).to(input_dtype)  # 应用SiLU门控并返回

        if ((self.n_groups % self.tp_size) != 0) or self.n_groups != 1:  # 如果分组数不能被tp_size整除或分组数不为1
            return self.forward_native(x, gate)  # 回退到原生实现

        return rms_norm_gated(  # 调用优化的RMS门控归一化内核
            x=x,  # 输入张量
            weight=self.weight.data,  # 归一化权重
            bias=None,  # 无偏置
            z=gate,  # 门控信号
            eps=self.variance_epsilon,  # 方差epsilon
            norm_before_gate=False,  # 先门控后归一化
            is_rms_norm=True,  # 使用RMS归一化
        )
