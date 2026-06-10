# AWQ量化方案抽象基类定义
# 本文件定义了AWQ量化框架的线性层和MoE层方案基类，提供抽象接口供具体方案实现
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod  # 导入抽象方法装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch库

from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme  # 导入基础方案类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型

__all__ = ["AWQLinearSchemeBase", "AWQMoESchemeBase"]  # 模块公开接口


class AWQLinearSchemeBase(BaseLinearScheme):  # AWQ线性层量化方案抽象基类，继承自BaseLinearScheme
    @abstractmethod
    def create_weights(self, *args, **kwargs):  # 创建量化权重参数的抽象方法
        raise NotImplementedError  # 未实现异常

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 加载权重后处理的抽象方法
        raise NotImplementedError  # 未实现异常

    @abstractmethod
    def apply_weights(  # 应用量化权重进行前向计算的抽象方法
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        raise NotImplementedError  # 未实现异常


class AWQMoESchemeBase(BaseMoEScheme):  # AWQ MoE量化方案抽象基类，继承自BaseMoEScheme
    @abstractmethod
    def create_weights(self, *args, **kwargs):  # 创建量化权重参数的抽象方法
        raise NotImplementedError  # 未实现异常

    @abstractmethod
    def create_moe_runner(  # 创建MoE运行器的抽象方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError  # 未实现异常

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):  # 加载权重后处理的抽象方法
        raise NotImplementedError  # 未实现异常

    @abstractmethod
    def apply_weights(  # 应用量化权重进行前向计算的抽象方法
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        raise NotImplementedError  # 未实现异常
