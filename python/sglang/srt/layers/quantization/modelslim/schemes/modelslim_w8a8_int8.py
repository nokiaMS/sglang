# ModelSlim W8A8 INT8 量化方案实现
# 本文件实现了 ModelSlim 框架下的 W8A8 INT8 线性层量化方案，
# 支持静态量化和动态量化两种模式，基于 NPU 硬件后端进行推理加速。

# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# 改编自 vLLM 项目的压缩张量量化模块
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 版权归属于 vLLM 项目的贡献者
from typing import Dict, List, Optional  # 导入类型提示工具 # import type hints

import torch  # 导入 PyTorch 深度学习框架 # import PyTorch framework

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (  # 从 NPU 量化模块导入线性层方法 # import linear methods from NPU quantization module
    NPUW8A8Int8DynamicLinearMethod,  # NPU W8A8 INT8 动态量化线性方法 # NPU W8A8 INT8 dynamic linear method
    NPUW8A8Int8LinearMethod,  # NPU W8A8 INT8 静态量化线性方法 # NPU W8A8 INT8 static linear method
)
from sglang.srt.layers.parameter import (  # 导入量化参数类型 # import quantization parameter types
    ChannelQuantScaleParameter,  # 逐通道量化缩放参数 # per-channel quantization scale parameter
    ModelWeightParameter,  # 模型权重参数 # model weight parameter
    PerTensorScaleParameter,  # 逐张量缩放参数 # per-tensor scale parameter
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme  # 导入 ModelSlim 线性层量化基类 # import ModelSlim linear scheme base class


class ModelSlimW8A8Int8(ModelSlimLinearScheme):  # ModelSlim W8A8 INT8 量化方案类，继承自 ModelSlimLinearScheme # ModelSlim W8A8 INT8 quantization scheme class

    def __init__(  # 初始化方法 # initializer
        self,
        quant_config: Dict[str, any],  # 量化配置字典 # quantization config dict
        prefix: str,  # 层名前缀 # layer name prefix
    ):
        self.quant_config = quant_config  # 保存量化配置 # save quantization config
        self.is_dynamic = (  # 判断是否为动态量化模式 # check if dynamic quantization mode
            self.quant_config.get(prefix + ".weight", "") == "W8A8_DYNAMIC"  # 根据权重配置判断是否为动态量化 # check if W8A8_DYNAMIC based on weight config
        )
        if self.is_dynamic:  # 如果是动态量化 # if dynamic quantization
            self.kernel = NPUW8A8Int8DynamicLinearMethod()  # 使用动态量化线性方法 # use dynamic quantization linear method
        else:  # 否则为静态量化 # otherwise static quantization
            self.kernel = NPUW8A8Int8LinearMethod()  # 使用静态量化线性方法 # use static quantization linear method

    def create_weights(  # 创建量化权重参数 # create quantized weight parameters
        self,
        layer: torch.nn.Module,  # 目标神经网络层 # target neural network layer
        input_size_per_partition: int,  # 每个分区的输入大小 # input size per partition
        output_partition_sizes: List[int],  # 输出分区大小列表 # output partition sizes list
        input_size: int,  # 总输入大小 # total input size
        output_size: int,  # 总输出大小 # total output size
        params_dtype: torch.dtype,  # 参数数据类型 # parameter data type
        **extra_weight_attrs,  # 额外权重属性 # extra weight attributes
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器 # get weight loader
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区总输出大小 # compute total output size per partition

        weight = ModelWeightParameter(  # 创建模型权重参数 # create model weight parameter
            data=torch.empty(  # 创建空张量 # create empty tensor
                (output_size_per_partition, input_size_per_partition), dtype=torch.int8  # int8 类型的权重张量 # int8 dtype weight tensor
            ),
            input_dim=1,  # 输入维度索引 # input dimension index
            output_dim=0,  # 输出维度索引 # output dimension index
            weight_loader=weight_loader,  # 权重加载器 # weight loader
        )
        layer.register_parameter("weight", weight)  # 注册权重参数到层 # register weight parameter to layer

        weight_scale = ChannelQuantScaleParameter(  # 创建逐通道权重缩放参数 # create per-channel weight scale parameter
            data=torch.empty((output_size_per_partition, 1), dtype=params_dtype),  # 缩放因子张量 # scale factor tensor
            output_dim=0,  # 输出维度索引 # output dimension index
            weight_loader=weight_loader,  # 权重加载器 # weight loader
        )
        layer.register_parameter("weight_scale", weight_scale)  # 注册权重缩放参数到层 # register weight scale parameter to layer

        weight_offset = ChannelQuantScaleParameter(  # 创建逐通道权重偏移参数 # create per-channel weight offset parameter
            data=torch.empty((output_size_per_partition, 1), dtype=params_dtype),  # 偏移张量 # offset tensor
            output_dim=0,  # 输出维度索引 # output dimension index
            weight_loader=weight_loader,  # 权重加载器 # weight loader
        )
        layer.register_parameter("weight_offset", weight_offset)  # 注册权重偏移参数到层 # register weight offset parameter to layer

        if not self.is_dynamic:  # 如果是静态量化模式 # if static quantization mode
            input_scale = PerTensorScaleParameter(  # 创建逐张量输入缩放参数 # create per-tensor input scale parameter
                data=torch.empty(1, dtype=params_dtype),  # 单元素缩放张量 # single-element scale tensor
                weight_loader=weight_loader,  # 权重加载器 # weight loader
            )
            input_scale.ignore_warning = True  # 忽略警告 # ignore warning
            layer.register_parameter("input_scale", input_scale)  # 注册输入缩放参数到层 # register input scale parameter to layer

            input_offset = PerTensorScaleParameter(  # 创建逐张量输入偏移参数 # create per-tensor input offset parameter
                data=torch.empty(1, dtype=params_dtype),  # 单元素偏移张量 # single-element offset tensor
                weight_loader=weight_loader,  # 权重加载器 # weight loader
            )
            input_offset.ignore_warning = True  # 忽略警告 # ignore warning
            layer.register_parameter("input_offset", input_offset)  # 注册输入偏移参数到层 # register input offset parameter to layer

            quant_bias = ChannelQuantScaleParameter(  # 创建逐通道量化偏置参数 # create per-channel quantization bias parameter
                data=torch.empty(output_size_per_partition, dtype=torch.int32),  # int32 类型的偏置张量 # int32 dtype bias tensor
                output_dim=0,  # 输出维度索引 # output dimension index
                weight_loader=weight_loader,  # 权重加载器 # weight loader
            )
            layer.register_parameter("quant_bias", quant_bias)  # 注册量化偏置参数到层 # register quantization bias parameter to layer

            if params_dtype == torch.bfloat16:  # 如果参数类型为 bfloat16 # if param dtype is bfloat16
                deq_scale_dtype = torch.float32  # 反量化缩放使用 float32 # dequantization scale uses float32
            elif params_dtype == torch.float16:  # 如果参数类型为 float16 # if param dtype is float16
                deq_scale_dtype = torch.int64  # 反量化缩放使用 int64 # dequantization scale uses int64
            else:  # 其他不支持的类型 # other unsupported types
                raise ValueError(f"Unsupported params_dtype: {params_dtype}")  # 抛出不支持的类型异常 # raise unsupported dtype error
            deq_scale = ChannelQuantScaleParameter(  # 创建逐通道反量化缩放参数 # create per-channel dequantization scale parameter
                data=torch.empty(output_size_per_partition, dtype=deq_scale_dtype),  # 反量化缩放张量 # dequantization scale tensor
                output_dim=0,  # 输出维度索引 # output dimension index
                weight_loader=weight_loader,  # 权重加载器 # weight loader
            )
            layer.register_parameter("deq_scale", deq_scale)  # 注册反量化缩放参数到层 # register dequantization scale parameter to layer

    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后处理 # process weights after loading
        self.kernel.process_weights_after_loading(layer)  # 调用内核的后处理方法 # call kernel's post-processing method

    def apply_weights(  # 应用量化权重进行前向计算 # apply quantized weights for forward computation
        self,
        layer: torch.nn.Module,  # 目标神经网络层 # target neural network layer
        x: torch.Tensor,  # 输入张量 # input tensor
        bias: Optional[torch.Tensor] = None,  # 偏置张量（可选） # bias tensor (optional)
    ) -> torch.Tensor:  # 返回输出张量 # return output tensor
        return self.kernel.apply(layer, x, bias)  # 调用内核的 apply 方法执行计算 # call kernel's apply method for computation
