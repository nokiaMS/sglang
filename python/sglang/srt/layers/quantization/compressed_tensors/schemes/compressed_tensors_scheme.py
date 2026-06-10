# 压缩张量量化方案的抽象基类模块
# 定义了CompressedTensorsLinearScheme和CompressedTensorsMoEScheme两个抽象基类
# 分别描述线性层和MoE层的权重量化创建与前向传播接口
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod  # 导入抽象方法装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型

import torch  # 导入PyTorch

from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme  # 导入线性方案和MoE方案的基类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型

__all__ = ["CompressedTensorsLinearScheme", "CompressedTensorsMoEScheme"]  # 模块公开导出的类列表


class CompressedTensorsLinearScheme(BaseLinearScheme):  # 压缩张量线性层量化方案抽象基类，继承自BaseLinearScheme
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by CompressedTensors.
    描述CompressedTensors支持的不同量化方案的权重创建和前向传播的抽象类。
    """

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低设备算力要求
        """
        Get minimum device capability.
        获取最低设备算力。
        """
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def create_weights(self, *args, **kwargs):  # 创建权重的抽象方法
        """
        Weight creation for the particular scheme. Inputs to this function
        为特定方案创建权重。此函数的输入参数

        """
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def apply_weights(  # 应用权重的抽象方法（执行前向传播）
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.
        执行特定方案的前向传播。在此处应用方案特定的反量化/量化步骤/内核。

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
            带有已注册权重和特定方案相关参数的torch.nn.Module。
        :param x: input to the layer
            该层的输入
        :param bias: bias parameter
            偏置参数

        """
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载完成后的后处理抽象方法
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        在权重加载完成后调用，用于执行任何需要的清理操作。
        """
        raise NotImplementedError  # 子类必须实现


class CompressedTensorsMoEScheme(BaseMoEScheme):  # 压缩张量MoE层量化方案抽象基类，继承自BaseMoEScheme
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by CompressedTensors.
    描述CompressedTensors支持的不同量化方案的权重创建和前向传播的抽象类。
    """

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低设备算力要求
        """
        Get minimum device capability.
        获取最低设备算力。
        """
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def create_weights(self, *args, **kwargs):  # 创建权重的抽象方法
        """
        Weight creation for the particular scheme. Inputs to this function
        为特定方案创建权重。此函数的输入参数

        """
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def create_moe_runner(  # 创建MoE运行器的抽象方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载完成后的后处理抽象方法
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        在权重加载完成后调用，用于执行任何需要的清理操作。
        """
        raise NotImplementedError  # 子类必须实现

    @abstractmethod
    def apply_weights(  # 应用权重的抽象方法（执行MoE前向传播）
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.
        执行特定方案的前向传播。在此处应用方案特定的反量化/量化步骤/内核。

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
            带有已注册权重和特定方案相关参数的torch.nn.Module。
        :param x: input to the layer
            该层的输入
        :param bias: bias parameter
            偏置参数

        """
        raise NotImplementedError  # 子类必须实现
