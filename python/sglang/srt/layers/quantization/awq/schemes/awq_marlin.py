# AWQ Marlin线性层量化方案实现
# 本文件实现了基于Marlin内核的AWQ线性层量化方案，利用Marlin高效推理内核进行量化计算
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations  # 启用延迟类型注解评估

from typing import TYPE_CHECKING, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch库

from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (  # 从GPU后端导入AWQ Marlin线性核心
    AWQMarlinLinearKernel,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter  # 导入量化参数类
from sglang.srt.layers.quantization.marlin_utils import verify_marlin_supports_shape  # 导入Marlin形状验证工具

from .awq_scheme import AWQLinearSchemeBase  # 导入AWQ线性方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.quantization.awq.awq import AWQMarlinConfig  # 导入AWQ Marlin配置类

__all__ = ["AWQMarlinLinearScheme"]  # 模块公开接口


class AWQMarlinLinearScheme(AWQLinearSchemeBase):  # AWQ Marlin线性层量化方案类，继承自AWQLinearSchemeBase
    def __init__(self, quant_config: "AWQMarlinConfig"):  # 初始化方法，接收AWQ Marlin量化配置
        self.quant_config = quant_config  # 保存量化配置
        self.kernel = AWQMarlinLinearKernel(quant_config)  # 初始化AWQ Marlin线性计算核心

    def create_weights(  # 创建量化权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 输入总大小
        params_dtype: torch.dtype,  # 参数数据类型
        weight_loader,  # 权重加载器
        **kwargs,  # 其他关键字参数
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)  # 计算总输出分区大小

        group_size = (  # 确定分组大小
            self.quant_config.group_size
            if self.quant_config.group_size != -1
            else input_size
        )

        verify_marlin_supports_shape(  # 验证Marlin内核是否支持当前形状
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size,
        )

        qweight = PackedvLLMParameter(  # 创建打包的量化权重参数
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,  # 输入维度索引
            output_dim=1,  # 输出维度索引
            packed_dim=1,  # 打包维度索引
            packed_factor=self.quant_config.pack_factor,  # 打包因子
            weight_loader=weight_loader,  # 权重加载器
        )

        num_groups = input_size_per_partition // group_size  # 计算量化分组数量

        qzeros = PackedvLLMParameter(  # 创建打包的量化零点参数
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,  # 输入维度索引
            output_dim=1,  # 输出维度索引
            packed_dim=1,  # 打包维度索引
            packed_factor=self.quant_config.pack_factor,  # 打包因子
            weight_loader=weight_loader,  # 权重加载器
        )

        scales = GroupQuantScaleParameter(  # 创建分组量化缩放参数
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,  # 输入维度索引
            output_dim=1,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )

        layer.register_parameter("qweight", qweight)  # 注册量化权重参数到层
        layer.register_parameter("qzeros", qzeros)  # 注册量化零点参数到层
        layer.register_parameter("scales", scales)  # 注册缩放参数到层

        layer.input_size_per_partition = input_size_per_partition  # 保存分区输入大小到层
        layer.output_size_per_partition = output_size_per_partition  # 保存分区输出大小到层
        layer.num_groups = num_groups  # 保存分组数量到层

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理方法
        self.kernel.process_weights_after_loading(layer)  # 调用核心的后处理方法

    def apply_weights(  # 应用量化权重进行前向计算方法
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        return self.kernel.apply(layer, x, bias)  # 调用核心的apply方法执行计算
