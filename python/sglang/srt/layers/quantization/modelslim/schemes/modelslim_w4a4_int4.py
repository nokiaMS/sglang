# ModelSlim W4A4 Int4量化方案实现文件，用于线性层的4比特权重4比特激活动态量化
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Dict, List, Optional  # 导入类型注解工具

import torch  # 导入PyTorch库

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (  # 从NPU量化模块导入W4A4动态线性方法
    NPU_W4A4DynamicLinearMethod,
)
from sglang.srt.layers.parameter import PerTensorScaleParameter  # 导入每张量缩放参数类
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme  # 导入ModelSlim线性层量化方案基类
from sglang.srt.utils import set_weight_attrs  # 导入设置权重属性的工具函数


class ModelSlimW4A4Int4(ModelSlimLinearScheme):  # W4A4 Int4量化方案类，继承自ModelSlimLinearScheme

    def __init__(  # 初始化方法
        self,
        quant_config: Dict[str, any],  # 量化配置字典
        prefix: str,  # 参数前缀
    ):
        self.quant_config = quant_config  # 保存量化配置
        self.is_dynamic = self.quant_config[prefix + ".weight"] == "W4A4_DYNAMIC"  # 判断是否为动态量化模式
        self.kernel = NPU_W4A4DynamicLinearMethod()  # 初始化NPU W4A4动态线性方法内核

    @staticmethod  # 静态方法
    def get_weight(  # 获取权重参数字典
        input_size: int, output_size: int, params_dtype: torch.dtype  # input_size:输入维度, output_size:输出维度, params_dtype:参数数据类型
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}  # 创建int8类型的空权重张量
        return params_dict  # 返回参数字典

    @staticmethod  # 静态方法
    def get_perchannel_param(  # 获取逐通道参数字典（缩放和偏移）
        output_size: int,  # 输出维度大小
        params_dtype: torch.dtype,  # 参数数据类型
    ) -> Dict[str, Any]:
        params_dict = {}  # 初始化参数字典
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)  # 创建逐通道权重缩放因子
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)  # 创建逐通道权重偏移量
        return params_dict  # 返回参数字典

    def create_weights(  # 创建权重并注册到层中
        self,
        layer: torch.nn.Module,  # 目标层模块
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        output_size: int,  # 总输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)  # 计算每个分区的总输出大小
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器

        weight_dict = {  # 创建权重字典
            "weight": torch.empty(  # 创建空权重张量
                output_size_per_partition, input_size_per_partition, dtype=torch.int8  # int8类型，形状为(输出分区大小, 输入分区大小)
            )
        }
        for weight_name, weight_param in weight_dict.items():  # 遍历权重字典
            param = torch.nn.Parameter(weight_param, requires_grad=False)  # 创建不可训练的参数
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})  # 设置输入维度为1，输出维度为0
            layer.register_parameter(weight_name, param)  # 将参数注册到层中
            set_weight_attrs(param, extra_weight_attrs)  # 设置额外权重属性

        pertensor_dict = {}  # 初始化每张量参数字典（当前为空）
        for pertensor_name, pertensor_param in pertensor_dict.items():  # 遍历每张量参数字典
            param = PerTensorScaleParameter(  # 创建每张量缩放参数
                data=pertensor_param, weight_loader=weight_loader  # 传入数据和权重加载器
            )
            # disable warning  禁用警告
            param.ignore_warning = True  # 忽略参数警告
            layer.register_parameter(pertensor_name, param)  # 将参数注册到层中

        perchannel_dict = {}  # 初始化逐通道参数字典
        perchannel_dict["weight_scale"] = torch.empty(  # 创建逐通道权重缩放因子
            output_size_per_partition, 1, dtype=params_dtype  # 形状为(输出分区大小, 1)
        )
        perchannel_dict["weight_offset"] = torch.empty(  # 创建逐通道权重偏移量
            output_size_per_partition, 1, dtype=params_dtype  # 形状为(输出分区大小, 1)
        )
        for perchannel_name, perchannel_param in perchannel_dict.items():  # 遍历逐通道参数字典
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)  # 创建不可训练的参数
            set_weight_attrs(param, {"output_dim": 0})  # 设置输出维度为0
            layer.register_parameter(perchannel_name, param)  # 将参数注册到层中
            set_weight_attrs(param, extra_weight_attrs)  # 设置额外权重属性

    def process_weights_after_loading(self, layer):  # 权重加载后的后处理，委托给内核执行
        self.kernel.process_weights_after_loading(layer)  # 调用内核的后处理方法

    def apply_weights(  # 应用权重执行前向传播
        self,
        layer: torch.nn.Module,  # 含权重的层模块
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 可选的偏置参数
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)  # 调用内核的apply方法执行计算并返回结果
