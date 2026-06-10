# SPDX-License-Identifier: Apache-2.0
# 文件说明：GPTQ Marlin量化方案的线性层实现，提供基于Marlin内核的GPTQ量化权重创建、加载后处理和前向推理功能
from __future__ import annotations  # 启用延迟类型注解求值

from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型

import torch  # 导入PyTorch深度学习框架

from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (  # 导入GPTQ Marlin线性核相关类
    GPTQMarlinLinearKernel,  # GPTQ Marlin线性计算核
    MarlinLinearLayerConfig,  # Marlin线性层配置类
)
from sglang.srt.layers.parameter import (  # 导入量化参数类
    ChannelQuantScaleParameter,  # 通道级量化缩放参数
    GroupQuantScaleParameter,  # 分组级量化缩放参数
    PackedColumnParameter,  # 打包列参数
    PackedvLLMParameter,  # vLLM打包参数
    RowvLLMParameter,  # vLLM行参数
)
from sglang.srt.layers.quantization.marlin_utils import (  # 导入Marlin工具函数
    marlin_repeat_scales_on_all_ranks,  # 判断是否在所有rank上重复缩放因子
    verify_marlin_supported,  # 验证Marlin是否支持当前配置
)

from .gptq_scheme import GPTQLinearSchemeBase  # 导入GPTQ线性方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig  # 导入GPTQ Marlin配置类

__all__ = ["GPTQMarlinLinearScheme"]  # 模块公开接口，导出GPTQ Marlin线性方案类


class GPTQMarlinLinearScheme(GPTQLinearSchemeBase):  # GPTQ Marlin线性量化方案类，继承自GPTQ线性方案基类
    def __init__(self, quant_config: "GPTQMarlinConfig"):  # 初始化方法，接收GPTQ Marlin量化配置
        self.quant_config = quant_config  # 保存量化配置
        self.kernel = GPTQMarlinLinearKernel(quant_config)  # 创建GPTQ Marlin线性计算核实例

        verify_marlin_supported(  # 验证Marlin是否支持当前量化类型和分组大小
            quant_type=self.quant_config.quant_type,  # 量化类型
            group_size=self.quant_config.group_size,  # 分组大小
        )

    def create_weights(  # 创建量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 完整输入大小
        output_size: int,  # 完整输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        weight_loader,  # 权重加载器
        **kwargs,  # 其他关键字参数
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出总大小
        is_row_parallel = input_size != input_size_per_partition  # 判断是否为行并行（输入大小不等则说明被切分）

        self.kernel.kernel_config = MarlinLinearLayerConfig(  # 配置Marlin线性层参数
            full_weight_shape=(input_size, output_size),  # 完整权重形状
            partition_weight_shape=(  # 分区权重形状
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_config.quant_type,  # 权重量化类型
            act_type=params_dtype,  # 激活值数据类型
            group_size=self.quant_config.group_size,  # 量化分组大小
            zero_points=False,  # 不使用零点
            has_g_idx=self.quant_config.desc_act,  # 是否有分组索引（按激活值降序排列）
        )

        group_size = (  # 确定实际分组大小
            self.quant_config.group_size  # 如果配置的分组大小不是-1，则使用配置值
            if self.quant_config.group_size != -1
            else input_size  # 否则使用输入大小作为分组大小（即每列一组）
        )

        if marlin_repeat_scales_on_all_ranks(  # 判断是否需要在所有rank上重复缩放因子
            self.quant_config.desc_act, self.quant_config.group_size, is_row_parallel
        ):
            scales_and_zp_input_dim = None  # 不指定输入维度，表示缩放因子在所有rank上重复
            scales_and_zp_size = input_size // group_size  # 缩放因子和零点的大小基于完整输入
        else:
            scales_and_zp_input_dim = 0  # 指定输入维度为0
            scales_and_zp_size = input_size_per_partition // group_size  # 缩放因子和零点的大小基于分区输入

        qweight = PackedvLLMParameter(  # 创建打包量化权重参数
            data=torch.empty(  # 分配空张量
                input_size_per_partition // self.quant_config.pack_factor,  # 打包后的输入维度大小
                output_size_per_partition,  # 输出维度大小
                dtype=torch.int32,  # 32位整数类型
            ),
            input_dim=0,  # 输入维度索引
            output_dim=1,  # 输出维度索引
            packed_dim=0,  # 打包维度索引
            packed_factor=self.quant_config.pack_factor,  # 打包因子
            weight_loader=weight_loader,  # 权重加载器
        )

        g_idx = RowvLLMParameter(  # 创建分组索引参数
            data=torch.empty(  # 分配空张量
                input_size_per_partition,  # 输入维度大小
                dtype=torch.int32,  # 32位整数类型
            ),
            input_dim=0,  # 输入维度索引
            weight_loader=weight_loader,  # 权重加载器
        )

        qzeros_args = {  # 量化零点参数字典
            "data": torch.empty(  # 分配空张量
                scales_and_zp_size,  # 缩放和零点大小
                output_size_per_partition // self.quant_config.pack_factor,  # 打包后的输出维度大小
                dtype=torch.int32,  # 32位整数类型
            ),
            "weight_loader": weight_loader,  # 权重加载器
        }
        weight_scale_args = {  # 权重缩放因子参数字典
            "data": torch.empty(  # 分配空张量
                scales_and_zp_size,  # 缩放和零点大小
                output_size_per_partition,  # 输出维度大小
                dtype=params_dtype,  # 参数数据类型
            ),
            "weight_loader": weight_loader,  # 权重加载器
        }

        if scales_and_zp_input_dim is None:  # 如果缩放因子在所有rank上重复
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)  # 创建通道级缩放参数
            qzeros = PackedColumnParameter(  # 创建打包列零点参数
                output_dim=1,  # 输出维度索引
                packed_dim=1,  # 打包维度索引
                packed_factor=self.quant_config.pack_factor,  # 打包因子
                **qzeros_args,  # 零点参数
            )
        else:  # 否则使用分组级缩放
            scales = GroupQuantScaleParameter(  # 创建分组级缩放参数
                output_dim=1, input_dim=0, **weight_scale_args  # 指定输出和输入维度
            )
            qzeros = PackedvLLMParameter(  # 创建vLLM打包零点参数
                input_dim=0,  # 输入维度索引
                output_dim=1,  # 输出维度索引
                packed_dim=1,  # 打包维度索引
                packed_factor=self.quant_config.pack_factor,  # 打包因子
                **qzeros_args,  # 零点参数
            )

        layer.register_parameter("qweight", qweight)  # 注册量化权重参数到层
        layer.register_parameter("g_idx", g_idx)  # 注册分组索引参数到层
        layer.register_parameter("scales", scales)  # 注册缩放因子参数到层
        layer.register_parameter("qzeros", qzeros)  # 注册量化零点参数到层

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法，对权重进行Marlin格式转换
        self.kernel.process_weights_after_loading(layer)  # 调用内核的权重加载后处理方法

    def apply_weights(  # 应用量化权重进行前向计算
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]  # 层、输入张量、偏置
    ):
        return self.kernel.apply(layer, x, bias)  # 调用内核的apply方法执行量化矩阵乘法
