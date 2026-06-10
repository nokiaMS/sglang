# AWQ Intel CPU AMX平台量化方案实现模块
# 提供基于Intel AMX指令集的AWQ线性层和MoE量化内核及方案类

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations  # 启用延迟类型注解评估

from typing import TYPE_CHECKING, Optional  # 类型提示工具

import torch  # PyTorch深度学习框架

from sglang.srt.layers.amx_utils import (  # Intel AMX工具函数
    CPUQuantMethod,  # CPU量化方法枚举
    _amx_process_weight_after_loading,  # AMX权重加载后处理函数
)
from sglang.srt.layers.moe import MoeRunnerConfig  # MoE运行器配置

from .awq_linear import AWQLinearScheme  # 通用AWQ线性层方案
from .awq_moe import AWQMoEScheme  # 通用AWQ MoE方案

if TYPE_CHECKING:  # 仅在类型检查时导入，运行时不导入
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 标准分发输出类型
    from sglang.srt.layers.quantization.awq.awq import AWQConfig  # AWQ配置类

__all__ = ["AWQIntelAMXLinearScheme", "AWQIntelAMXMoEScheme"]  # 模块公开导出的符号列表


class AWQIntelAMXLinearKernel:  # Intel AMX平台AWQ线性层内核类
    def __init__(self, quant_config: "AWQConfig"):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的AMX后处理
        _amx_process_weight_after_loading(  # 调用AMX权重后处理函数
            layer, ["qweight", "qzeros", "scales"], None, "awq"  # 处理qweight、qzeros、scales三个权重，使用awq格式
        )
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)  # 将qweight转为不可训练参数
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)  # 将qzeros转为不可训练参数
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)  # 将scales转为不可训练参数

    def apply(  # 应用AMX量化线性计算
        self,
        layer: torch.nn.Module,  # 目标层模块
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量，可选
    ) -> torch.Tensor:  # 返回输出张量
        return torch.ops.sgl_kernel.int4_scaled_mm_cpu(  # 调用CPU int4缩放矩阵乘法内核
            x,  # 输入张量
            layer.qweight,  # 量化权重
            layer.qzeros,  # 量化零点
            layer.scales,  # 缩放因子
            bias,  # 偏置
        )


class AWQIntelAMXLinearScheme(AWQLinearScheme):  # Intel AMX平台AWQ线性层方案类，继承自AWQLinearScheme
    """Linear scheme for AWQ on Intel CPU with AMX."""  # Intel CPU AMX平台上的AWQ线性层方案

    def _init_kernel(self, quant_config: "AWQConfig"):  # 初始化内核（覆盖父类方法）
        return AWQIntelAMXLinearKernel(quant_config)  # 返回Intel AMX线性层内核实例


class AWQIntelAMXMoEKernel:  # Intel AMX平台AWQ MoE内核类
    def __init__(self, quant_config: "AWQConfig"):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的AMX后处理
        _amx_process_weight_after_loading(  # 处理w13（门控+上投影）权重
            layer, ["w13_qweight", "w13_qzeros", "w13_scales"], None, "awq"  # 处理w13的qweight、qzeros、scales
        )
        _amx_process_weight_after_loading(  # 处理w2（下投影）权重
            layer, ["w2_qweight", "w2_qzeros", "w2_scales"], None, "awq"  # 处理w2的qweight、qzeros、scales
        )

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层模块和MoE运行器配置
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def apply(  # 应用AMX量化MoE计算
        self,
        layer: torch.nn.Module,  # 目标层模块
        dispatch_output: "StandardDispatchOutput",  # 分发输出
    ) -> torch.Tensor:  # 返回输出张量
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入类型

        assert (  # 断言
            self.moe_runner_config.activation == "silu"  # 激活函数必须是SiLU
        ), "Only SiLU activation is supported."  # 仅支持SiLU激活函数

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取top-k输出
        topk_weights, topk_ids, _ = topk_output  # 解包top-k权重、ID
        output = torch.ops.sgl_kernel.fused_experts_cpu(  # 调用CPU融合专家内核
            x,  # 输入隐藏状态
            layer.w13_qweight,  # w13量化权重
            layer.w2_qweight,  # w2量化权重
            topk_weights,  # top-k权重
            topk_ids,  # top-k专家ID
            False,  # inplace See [Note] inplace should be False in fused_experts.  # 原地操作标志，参见[Note]在fused_experts中应为False
            CPUQuantMethod.INT4_W4A8,  # 量化方法：INT4 W4A8
            layer.w13_scales,  # w1_scale  # w13缩放因子
            layer.w2_scales,  # w2_scale  # w2缩放因子
            layer.w13_qzeros,  # w13量化零点
            layer.w2_qzeros,  # w2量化零点
            None,  # block_size  # 块大小，不使用
            None,  # w1 bias  # w1偏置，不使用
            None,  # w3 bias  # w3偏置，不使用
            None,  # alpha  # alpha参数，不使用
            None,  # limit  # limit参数，不使用
            True,  # is_vnni  # 是否为VNNI格式
        )
        return StandardCombineInput(hidden_states=output)  # 包装为标准合并输入并返回


class AWQIntelAMXMoEScheme(AWQMoEScheme):  # Intel AMX平台AWQ MoE方案类，继承自AWQMoEScheme
    """MoE scheme for AWQ on Intel CPU with AMX."""  # Intel CPU AMX平台上的AWQ MoE方案

    def _init_kernel(self, quant_config: "AWQConfig"):  # 初始化内核（覆盖父类方法）
        return AWQIntelAMXMoEKernel(quant_config)  # 返回Intel AMX MoE内核实例

    def create_moe_runner(  # 创建MoE运行器（覆盖父类方法）
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 层模块和MoE运行器配置
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        self.kernel.create_moe_runner(layer, moe_runner_config)  # 委托给内核创建MoE运行器
