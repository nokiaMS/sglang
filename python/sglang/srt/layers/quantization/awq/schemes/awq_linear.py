# AWQ线性层量化方案实现
# 本文件实现了AWQ量化框架下线性层的量化方案，包括标准GPU方案和昇腾NPU方案
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations  # 启用延迟类型注解评估

from typing import TYPE_CHECKING, List, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch库

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter  # 导入量化参数类

from .awq_scheme import AWQLinearSchemeBase  # 导入AWQ线性方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.quantization.awq.awq import AWQConfig  # 导入AWQ配置类

__all__ = ["AWQLinearScheme", "AWQAscendLinearScheme"]  # 模块公开接口


class AWQLinearScheme(AWQLinearSchemeBase):  # AWQ线性层量化方案类，继承自AWQLinearSchemeBase
    def __init__(self, quant_config: "AWQConfig"):  # 初始化方法，接收AWQ量化配置
        self.quant_config = quant_config  # 保存量化配置
        self.kernel = self._init_kernel(quant_config)  # 初始化量化计算核心

    def _init_kernel(self, quant_config: "AWQConfig"):  # 初始化计算核心方法
        from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (  # 从GPU后端导入AWQ线性核心
            AWQLinearKernel,
        )

        return AWQLinearKernel(quant_config)  # 返回AWQ线性核心实例

    def create_weights(  # 创建量化权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        params_dtype: torch.dtype,  # 参数数据类型
        weight_loader,  # 权重加载器
        **kwargs,  # 其他关键字参数
    ):
        if input_size_per_partition % self.quant_config.group_size != 0:  # 检查输入大小是否与分组大小对齐
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)  # 计算总输出分区大小
        if output_size_per_partition % self.quant_config.pack_factor != 0:  # 检查输出大小是否与打包因子对齐
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
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

        qzeros = PackedvLLMParameter(  # 创建打包的量化零点参数
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
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
                input_size_per_partition // self.quant_config.group_size,
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理方法
        self.kernel.process_weights_after_loading(layer)  # 调用核心的后处理方法

    def apply_weights(  # 应用量化权重进行前向计算方法
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        return self.kernel.apply(layer, x, bias)  # 调用核心的apply方法执行计算


class AWQAscendLinearScheme(AWQLinearScheme):  # AWQ昇腾NPU线性层量化方案，继承自AWQLinearScheme
    def _init_kernel(self, quant_config: "AWQConfig"):  # 初始化昇腾NPU计算核心方法
        from sglang.srt.hardware_backend.npu.quantization.awq_kernels import (  # 从NPU后端导入AWQ昇腾线性核心
            AWQAscendLinearKernel,
        )

        return AWQAscendLinearKernel(quant_config)  # 返回昇腾AWQ线性核心实例
