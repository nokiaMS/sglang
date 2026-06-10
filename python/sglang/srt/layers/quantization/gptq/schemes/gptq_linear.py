# GPTQ线性量化方案模块：定义GPTQ线性层权重量化的创建、处理和推理方案，包括标准GPU方案和Ascend NPU方案
# SPDX-License-Identifier: Apache-2.0  # Apache 2.0许可证声明
from __future__ import annotations  # 启用延迟注解评估，支持类型提示中的前向引用

from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型工具

import torch  # 导入PyTorch张量库

from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import GPTQLinearKernel  # 导入GPU端GPTQ线性核
from sglang.srt.layers.parameter import (  # 导入各种量化参数类
    ChannelQuantScaleParameter,  # 通道级量化缩放参数
    GroupQuantScaleParameter,  # 组级量化缩放参数
    PackedColumnParameter,  # 打包列参数
    PackedvLLMParameter,  # vLLM打包参数
    RowvLLMParameter,  # vLLM行参数
)
from sglang.srt.utils import set_weight_attrs  # 导入权重属性设置工具函数

from .gptq_scheme import GPTQLinearSchemeBase  # 导入GPTQ线性方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免运行时循环依赖
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig  # 导入GPTQ配置类型

__all__ = ["GPTQLinearScheme", "GPTQAscendLinearScheme"]  # 定义模块公开导出的符号列表


class GPTQLinearScheme(GPTQLinearSchemeBase):  # GPTQ线性量化方案类，继承自基类
    def __init__(self, quant_config: "GPTQConfig"):  # 初始化方法，接收GPTQ配置
        self.quant_config = quant_config  # 保存量化配置
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"  # 判断是否使用v2格式
        self.kernel = self._init_kernel(quant_config)  # 初始化量化计算核

    def _init_kernel(self, quant_config: "GPTQConfig"):  # 初始化量化计算核，可被子类覆写
        return GPTQLinearKernel(quant_config)  # 返回GPU端GPTQ线性核实例

    def create_weights(  # 创建量化权重参数并注册到层中
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        params_dtype: torch.dtype,  # 参数数据类型
        weight_loader,  # 权重加载器
        **kwargs,  # 其他关键字参数
    ):  # 创建量化权重参数
        if input_size_per_partition % self.quant_config.group_size != 0:  # 检查输入大小是否与组大小对齐
            raise ValueError(  # 抛出值错误
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."  # 输入大小未对齐，可能由于张量并行度过大
            )
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出总大小
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:  # 检查输出大小是否与打包因子对齐
            raise ValueError(  # 抛出值错误
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."  # 输出大小未对齐，可能由于张量并行度过大
            )

        group_size = (  # 确定实际组大小
            self.quant_config.group_size  # 使用配置中的组大小
            if self.quant_config.group_size != -1  # 如果组大小不为-1（即非每通道量化）
            else input_size  # 否则组大小等于整个输入大小（每通道量化）
        )
        self.kernel.use_shuffle = True  # 默认启用shuffle
        scale_and_zero_size = input_size // group_size  # 计算缩放和零点的维度大小
        scale_and_zero_input_dim = None  # 初始化缩放和零点的输入维度为None
        if (  # 如果输入大小与分区输入大小不同且组大小不为-1
            input_size != input_size_per_partition
            and self.quant_config.group_size != -1
        ):
            if self.quant_config.desc_act:  # 如果启用了描述符激活（desc_act）
                self.kernel.use_shuffle = False  # 禁用shuffle
            else:  # 未启用desc_act
                scale_and_zero_size = input_size_per_partition // group_size  # 按分区大小计算缩放零点维度
                scale_and_zero_input_dim = 0  # 设置输入维度为0

        qweight = PackedvLLMParameter(  # 创建打包的量化权重参数
            data=torch.empty(  # 创建空张量存储量化权重
                input_size_per_partition // self.quant_config.pack_factor,  # 打包后的第一维大小
                output_size_per_partition,  # 第二维为输出大小
                dtype=torch.int32,  # 数据类型为int32
            ),
            input_dim=0,  # 输入维度为第0维
            output_dim=1,  # 输出维度为第1维
            packed_dim=0,  # 打包维度为第0维
            packed_factor=self.quant_config.pack_factor,  # 打包因子
            weight_loader=weight_loader,  # 权重加载器
        )

        g_idx = RowvLLMParameter(  # 创建组索引参数
            data=torch.tensor(  # 创建组索引张量
                [
                    i // self.quant_config.group_size  # 每个元素对应的组索引
                    for i in range(input_size_per_partition)  # 遍历输入分区大小
                ],
                dtype=torch.int32,  # 数据类型为int32
            ),
            input_dim=0,  # 输入维度为第0维
            weight_loader=weight_loader,  # 权重加载器
        )
        qzeros_args = {  # 量化零点参数构造参数
            "data": torch.empty(  # 创建空张量存储量化零点
                scale_and_zero_size,  # 第一维为缩放零点大小
                output_size_per_partition // self.quant_config.pack_factor,  # 第二维为打包后的输出大小
                dtype=torch.int32,  # 数据类型为int32
            ),
            "weight_loader": weight_loader,  # 权重加载器
        }
        weight_scale_args = {  # 权重缩放参数构造参数
            "data": torch.empty(  # 创建空张量存储权重缩放因子
                scale_and_zero_size,  # 第一维为缩放零点大小
                output_size_per_partition,  # 第二维为输出大小
                dtype=params_dtype,  # 数据类型与参数类型一致
            ),
            "weight_loader": weight_loader,  # 权重加载器
        }
        if scale_and_zero_input_dim is None:  # 如果缩放零点没有分区输入维度
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)  # 创建通道级缩放参数
            qzeros = PackedColumnParameter(  # 创建打包列零点参数
                output_dim=1,  # 输出维度为第1维
                packed_dim=1,  # 打包维度为第1维
                packed_factor=self.quant_config.pack_factor,  # 打包因子
                **qzeros_args,  # 传入零点构造参数
            )
        else:  # 如果缩放零点有分区输入维度
            scales = GroupQuantScaleParameter(  # 创建组级缩放参数
                output_dim=1, input_dim=0, **weight_scale_args  # 设置输出和输入维度
            )
            qzeros = PackedvLLMParameter(  # 创建打包的零点参数
                input_dim=0,  # 输入维度为第0维
                output_dim=1,  # 输出维度为第1维
                packed_dim=1,  # 打包维度为第1维
                packed_factor=self.quant_config.pack_factor,  # 打包因子
                **qzeros_args,  # 传入零点构造参数
            )

        layer.register_parameter("qweight", qweight)  # 注册量化权重参数到层
        layer.register_parameter("g_idx", g_idx)  # 注册组索引参数到层
        layer.register_parameter("qzeros", qzeros)  # 注册量化零点参数到层
        layer.register_parameter("scales", scales)  # 注册缩放因子参数到层

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法
        self.kernel.process_weights_after_loading(layer)  # 委托给计算核处理

    def apply_weights(  # 应用量化权重进行推理计算
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]  # 层、输入张量、偏置
    ):  # 返回推理结果
        return self.kernel.apply(layer, x, bias)  # 委托给计算核执行


class GPTQAscendLinearScheme(GPTQLinearScheme):  # GPTQ Ascend NPU线性量化方案，继承自标准方案
    def _init_kernel(self, quant_config: "GPTQConfig"):  # 初始化Ascend NPU专用计算核
        from sglang.srt.hardware_backend.npu.quantization.gptq_kernels import (  # 从NPU后端导入Ascend核
            GPTQLinearAscendKernel,  # Ascend NPU GPTQ线性核
        )

        return GPTQLinearAscendKernel(quant_config)  # 返回Ascend NPU核实例

    def create_weights(self, layer: torch.nn.Module, **kwargs):  # 创建量化权重（Ascend NPU版本）
        super().create_weights(layer=layer, **kwargs)  # 调用父类方法创建权重
        set_weight_attrs(layer.qzeros, {"pack_factor": self.quant_config.pack_factor})  # 为零点参数设置打包因子属性
        set_weight_attrs(layer.qweight, {"pack_factor": self.quant_config.pack_factor})  # 为权重参数设置打包因子属性

        if self.quant_config.desc_act:  # 如果启用了desc_act
            raise ValueError(  # 抛出值错误，Ascend NPU不支持desc_act
                "Currently, desc_act (True) is not supported by GPTQ "
                "quantization on npu."  # 当前Ascend NPU上的GPTQ量化不支持desc_act
            )
