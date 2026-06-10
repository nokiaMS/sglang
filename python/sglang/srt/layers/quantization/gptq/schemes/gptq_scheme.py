# SPDX-License-Identifier: Apache-2.0
# 文件说明：GPTQ量化方案的抽象基类定义，包含线性层方案基类和MoE方案基类

from abc import abstractmethod  # 导入抽象方法装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型

import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme  # 导入线性方案基类和MoE方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型

__all__ = ["GPTQLinearSchemeBase", "GPTQMoESchemeBase"]  # 模块公开接口，导出两个GPTQ方案基类


class GPTQLinearSchemeBase(BaseLinearScheme):  # GPTQ线性量化方案基类，继承自线性方案基类
    @abstractmethod  # 抽象方法装饰器
    def create_weights(self, *args, **kwargs):  # 创建量化权重的抽象方法，子类必须实现
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 抽象方法装饰器
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后处理的抽象方法
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 抽象方法装饰器
    def apply_weights(  # 应用量化权重进行前向计算的抽象方法
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]  # 层、输入张量、偏置
    ):
        raise NotImplementedError  # 未实现则抛出异常


class GPTQMoESchemeBase(BaseMoEScheme):  # GPTQ MoE量化方案基类，继承自MoE方案基类
    @abstractmethod  # 抽象方法装饰器
    def create_weights(self, *args, **kwargs):  # 创建MoE量化权重的抽象方法，子类必须实现
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 抽象方法装饰器
    def create_moe_runner(  # 创建MoE运行器的抽象方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层和MoE运行器配置
    ):
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 抽象方法装饰器
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 权重加载后处理的抽象方法
        raise NotImplementedError  # 未实现则抛出异常

    @abstractmethod  # 抽象方法装饰器
    def apply_weights(  # 应用量化权重进行MoE前向计算的抽象方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    ):
        raise NotImplementedError  # 未实现则抛出异常
