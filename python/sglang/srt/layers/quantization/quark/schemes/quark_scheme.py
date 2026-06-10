# SPDX-License-Identifier: Apache-2.0
# Quark量化方案的抽象基类定义文件，定义了Quark线性层和MoE层的量化方案接口

from abc import abstractmethod  # 导入抽象方法装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch库

from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme  # 导入线性层和MoE层的基础量化方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型

__all__ = ["QuarkLinearScheme", "QuarkMoEScheme"]  # 模块公开导出的类列表


class QuarkLinearScheme(BaseLinearScheme):  # Quark线性层量化方案抽象类，继承自BaseLinearScheme
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by Quark.
    抽象类，用于描述Quark支持的不同量化方案的权重创建和前向传播。
    """

    @classmethod  # 类方法装饰器
    @abstractmethod  # 抽象方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低设备计算能力要求
        """
        Get minimum device capability.
        获取最低设备计算能力。
        """
        raise NotImplementedError  # 未实现异常

    @abstractmethod  # 抽象方法装饰器
    def create_weights(self, *args, **kwargs):  # 创建权重的抽象方法
        """
        Weight creation for the particular scheme. Inputs to this function
        为特定方案创建权重。此函数的输入参数

        """
        raise NotImplementedError  # 未实现异常

    @abstractmethod  # 抽象方法装饰器
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后的后处理方法
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        在权重加载完成后调用，用于执行所需的清理操作。
        """
        raise NotImplementedError  # 未实现异常

    @abstractmethod  # 抽象方法装饰器
    def apply_weights(  # 应用权重（前向传播）的抽象方法
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.
        运行特定方案的前向传播。此处应应用方案特定的反量化/量化步骤/内核。

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
        :param layer: 包含已注册权重和与特定方案相关的其他参数的torch.nn.Module。
        :param x: input to the layer
        :param x: 层的输入
        :param bias: bias parameter
        :param bias: 偏置参数

        """
        raise NotImplementedError  # 未实现异常


class QuarkMoEScheme(BaseMoEScheme):  # Quark MoE层量化方案抽象类，继承自BaseMoEScheme
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by Quark.
    抽象类，用于描述Quark支持的不同量化方案的权重创建和前向传播。
    """

    @classmethod  # 类方法装饰器
    @abstractmethod  # 抽象方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低设备计算能力要求
        """
        Get minimum device capability.
        获取最低设备计算能力。
        """
        raise NotImplementedError  # 未实现异常

    @abstractmethod  # 抽象方法装饰器
    def create_weights(self, *args, **kwargs):  # 创建权重的抽象方法
        """
        Weight creation for the particular scheme. Inputs to this function
        为特定方案创建权重。此函数的输入参数

        """
        raise NotImplementedError  # 未实现异常

    @abstractmethod  # 抽象方法装饰器
    def create_moe_runner(  # 创建MoE运行器的抽象方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError  # 未实现异常

    @abstractmethod  # 抽象方法装饰器
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后的后处理方法
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        在权重加载完成后调用，用于执行所需的清理操作。
        """
        raise NotImplementedError  # 未实现异常

    @abstractmethod  # 抽象方法装饰器
    def apply_weights(  # 应用权重（前向传播）的抽象方法
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.
        运行特定方案的前向传播。此处应应用方案特定的反量化/量化步骤/内核。

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
        :param layer: 包含已注册权重和与特定方案相关的其他参数的torch.nn.Module。
        :param x: input to the layer
        :param x: 层的输入
        :param bias: bias parameter
        :param bias: 偏置参数

        """
        raise NotImplementedError  # 未实现异常
