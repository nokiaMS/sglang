# ModelSlim量化方案的抽象基类定义文件，定义了线性层和MoE层量化方案的基础接口
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod  # 导入抽象方法装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型

import torch  # 导入PyTorch库

from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme  # 导入线性层和MoE层的基础方案类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型

__all__ = ["ModelSlimLinearScheme", "ModelSlimMoEScheme"]  # 定义模块的公开导出列表


class ModelSlimLinearScheme(BaseLinearScheme):  # ModelSlim线性层量化方案抽象类，继承自BaseLinearScheme
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by ModelSlim.
    用于描述ModelSlim支持的不同量化方案的权重创建和前向传播的抽象类。
    """

    @abstractmethod  # 标记为抽象方法
    def create_weights(self, *args, **kwargs):  # 创建权重的抽象方法
        """
        Weight creation for the particular scheme. Inputs to this function
        为特定方案创建权重。该函数的输入参数

        """
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 标记为抽象方法
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后的后处理抽象方法
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        在权重加载完成后调用，用于执行必要的清理操作。
        """
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 标记为抽象方法
    def apply_weights(  # 应用权重（前向传播）的抽象方法
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]  # layer:含权重的模块, x:输入张量, bias:偏置参数
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.
        运行特定方案的前向传播。此处应应用方案特定的反量化/量化步骤/内核。

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
            layer: 包含已注册权重和特定方案相关参数的torch.nn.Module。
        :param x: input to the layer
            x: 该层的输入
        :param bias: bias parameter
            bias: 偏置参数

        """
        raise NotImplementedError  # 未实现则抛出异常


class ModelSlimMoEScheme(BaseMoEScheme):  # ModelSlim MoE层量化方案抽象类，继承自BaseMoEScheme
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by ModelSlim.
    用于描述ModelSlim支持的不同量化方案的权重创建和前向传播的抽象类（MoE版本）。
    """

    @abstractmethod  # 标记为抽象方法
    def create_weights(self, *args, **kwargs):  # 创建权重的抽象方法
        """
        Weight creation for the particular scheme. Inputs to this function
        为特定方案创建权重。该函数的输入参数

        """
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 标记为抽象方法
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后的后处理抽象方法
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        在权重加载完成后调用，用于执行必要的清理操作。
        """
        raise NotImplementedError  # 未实现则抛出异常

    def create_moe_runner(  # 创建MoE运行器的方法
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"  # layer:含权重的模块, moe_runner_config:MoE运行器配置
    ):
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 标记为抽象方法
    def apply_weights(  # 应用权重（前向传播）的抽象方法
        self,
        layer,  # 含权重的模块
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.
        运行特定方案的前向传播。此处应应用方案特定的反量化/量化步骤/内核。

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
            layer: 包含已注册权重和特定方案相关参数的torch.nn.Module。
        :param x: input to the layer
            x: 该层的输入
        :param bias: bias parameter
            bias: 偏置参数

        """
        raise NotImplementedError  # 未实现则抛出异常
