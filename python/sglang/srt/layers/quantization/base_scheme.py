# 量化方案基类模块
# 本文件定义了量化方案的抽象基类，包括 BaseLinearScheme（线性层量化方案基类）
# 和 BaseMoEScheme（MoE 层量化方案基类），规范了权重创建、加载后处理和前向计算的接口。

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架

from sglang.srt.layers.moe import MoeRunnerConfig  # 导入 MoE 运行器配置

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 标准分发输出类型

__all__ = ["BaseLinearScheme", "BaseMoEScheme"]  # 模块公开接口列表


class BaseLinearScheme(ABC):  # 线性层量化方案基类
    """
    Abstract class used to describe the weight creation and forward pass  # 抽象类，用于描述不同量化方案的
    of different quantization schemes.  # 权重创建和前向传播
    """

    @abstractmethod
    def create_weights(self, *args, **kwargs):  # 创建权重方法
        """
        Weight creation for the particular scheme. Inputs to this function  # 特定方案的权重创建。此函数的输入

        """
        raise NotImplementedError  # 需子类实现

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后处理
        """
        Called after weight loading is complete for any cleanup that  # 权重加载完成后调用，用于任何需要
        needs to occur.  # 进行的清理操作
        """
        raise NotImplementedError  # 需子类实现

    @abstractmethod
    def apply_weights(  # 应用权重进行前向计算
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]  # 层、输入张量、偏置
    ):
        """
        Run the forward pass for the particular scheme. This is where  # 运行特定方案的前向传播。这里
        scheme-specific dequant/quant steps/kernels should be applied.  # 应该应用特定方案的反量化/量化步骤/内核

        :param layer: torch.nn.Module with the registered weights and  # 参数 layer: 包含已注册权重和
            other parameters relevant to the particular scheme.  # 与特定方案相关的其他参数的 torch.nn.Module
        :param x: input to the layer  # 参数 x: 层的输入
        :param bias: bias parameter  # 参数 bias: 偏置参数

        """
        raise NotImplementedError  # 需子类实现


class BaseMoEScheme(ABC):  # MoE 层量化方案基类
    """
    Abstract class used to describe the weight creation and forward pass  # 抽象类，用于描述不同量化方案的
    of different quantization schemes.  # 权重创建和前向传播
    """

    @abstractmethod
    def create_weights(self, *args, **kwargs):  # 创建权重方法
        """
        Weight creation for the particular scheme. Inputs to this function  # 特定方案的权重创建。此函数的输入

        """
        raise NotImplementedError  # 需子类实现

    @abstractmethod
    def create_moe_runner(  # 创建 MoE 运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层和 MoE 运行器配置
    ):
        raise NotImplementedError  # 需子类实现

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后处理
        """
        Called after weight loading is complete for any cleanup that  # 权重加载完成后调用，用于任何需要
        needs to occur.  # 进行的清理操作
        """
        raise NotImplementedError  # 需子类实现

    @abstractmethod
    def apply_weights(  # 应用权重进行前向计算
        self,
        layer: torch.nn.Module,  # 目标层
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    ):
        """
        Run the forward pass for the particular scheme. This is where  # 运行特定方案的前向传播。这里
        scheme-specific dequant/quant steps/kernels should be applied.  # 应该应用特定方案的反量化/量化步骤/内核

        :param layer: torch.nn.Module with the registered weights and  # 参数 layer: 包含已注册权重和
            other parameters relevant to the particular scheme.  # 与特定方案相关的其他参数的 torch.nn.Module
        :param x: input to the layer  # 参数 x: 层的输入
        :param bias: bias parameter  # 参数 bias: 偏置参数

        """
        raise NotImplementedError  # 需子类实现
