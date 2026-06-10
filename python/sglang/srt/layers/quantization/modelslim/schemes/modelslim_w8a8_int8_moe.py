# ModelSlim W8A8 INT8 MoE（混合专家）量化方案实现
# 本文件实现了 ModelSlim 框架下的 W8A8 INT8 MoE 层量化方案，
# 支持动态量化模式，基于 NPU 硬件后端进行 MoE 推理加速。

from __future__ import annotations  # 启用延迟注解评估 # enable deferred annotation evaluation

import logging  # 导入日志模块 # import logging module
from typing import TYPE_CHECKING, Any, Dict  # 导入类型提示工具 # import type hints

import torch  # 导入 PyTorch 深度学习框架 # import PyTorch framework

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (  # 从 NPU 量化模块导入 MoE 方法 # import MoE method from NPU quantization module
    NPUW8A8Int8DynamicMoEMethod,  # NPU W8A8 INT8 动态量化 MoE 方法 # NPU W8A8 INT8 dynamic MoE method
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme  # 导入 ModelSlim MoE 量化基类 # import ModelSlim MoE scheme base class
from sglang.srt.utils import set_weight_attrs  # 导入权重属性设置工具 # import weight attribute setting utility

if TYPE_CHECKING:  # 类型检查时才导入 # import only during type checking
    from sglang.srt.layers.moe import MoeRunnerConfig  # MoE 运行器配置 # MoE runner config
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入令牌分发器类型 # import token dispatcher types
        CombineInput,  # 合并输入类型 # combine input type
        StandardDispatchOutput,  # 标准分发输出类型 # standard dispatch output type
    )

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器 # get logger for current module

__all__ = [  # 模块公开接口 # module public interface
    "ModelSlimW8A8Int8MoE",  # ModelSlim W8A8 INT8 MoE 量化方案类 # ModelSlim W8A8 INT8 MoE scheme class
]


class ModelSlimW8A8Int8MoE(ModelSlimMoEScheme):  # ModelSlim W8A8 INT8 MoE 量化方案类，继承自 ModelSlimMoEScheme # ModelSlim W8A8 INT8 MoE scheme class

    def __init__(  # 初始化方法 # initializer
        self,
        quant_config: Dict[str, Any],  # 量化配置字典 # quantization config dict
        prefix: str = None,  # 层名前缀 # layer name prefix
    ):
        self.quant_config = quant_config  # 保存量化配置 # save quantization config
        self.kernel = NPUW8A8Int8DynamicMoEMethod()  # 创建 NPU W8A8 INT8 动态 MoE 方法实例 # create NPU W8A8 INT8 dynamic MoE method instance

    def create_weights(  # 创建 MoE 量化权重参数 # create MoE quantized weight parameters
        self,
        layer: torch.nn.Module,  # 目标神经网络层 # target neural network layer
        num_experts: int,  # 专家数量 # number of experts
        hidden_size: int,  # 隐藏层大小 # hidden size
        intermediate_size_per_partition: int,  # 每个分区的中间层大小 # intermediate size per partition
        params_dtype: torch.dtype,  # 参数数据类型 # parameter data type
        **extra_weight_attrs,  # 额外权重属性 # extra weight attributes
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入 MoE 权重缩放支持枚举 # import MoE weight scale supported enum

        self.num_experts = num_experts  # 保存专家数量 # save number of experts
        extra_weight_attrs.update(  # 更新额外权重属性 # update extra weight attributes
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}  # 设置量化方法为逐通道 # set quant method to per-channel
        )

        # weight
        # 权重
        w13_weight = torch.nn.Parameter(  # 创建 w13（gate+up）权重参数 # create w13 (gate+up) weight parameter
            torch.empty(  # 创建空张量 # create empty tensor
                num_experts,  # 专家数量维度 # experts dimension
                2 * intermediate_size_per_partition,  # gate 和 up 的合并维度 # combined gate and up dimension
                hidden_size,  # 隐藏层大小维度 # hidden size dimension
                dtype=torch.int8,  # int8 数据类型 # int8 data type
            ),
            requires_grad=False,  # 不需要梯度 # no gradient required
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册 w13 权重参数到层 # register w13 weight parameter to layer
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置 w13 权重属性 # set w13 weight attributes
        w2_weight = torch.nn.Parameter(  # 创建 w2（down）权重参数 # create w2 (down) weight parameter
            torch.empty(  # 创建空张量 # create empty tensor
                num_experts,  # 专家数量维度 # experts dimension
                hidden_size,  # 隐藏层大小维度 # hidden size dimension
                intermediate_size_per_partition,  # 中间层大小维度 # intermediate size dimension
                dtype=torch.int8,  # int8 数据类型 # int8 data type
            ),
            requires_grad=False,  # 不需要梯度 # no gradient required
        )
        layer.register_parameter("w2_weight", w2_weight)  # 注册 w2 权重参数到层 # register w2 weight parameter to layer
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置 w2 权重属性 # set w2 weight attributes
        # scale
        # 缩放因子
        w13_weight_scale = torch.nn.Parameter(  # 创建 w13 权重缩放参数 # create w13 weight scale parameter
            torch.empty(  # 创建空张量 # create empty tensor
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32  # float32 类型的缩放张量 # float32 dtype scale tensor
            ),
            requires_grad=False,  # 不需要梯度 # no gradient required
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 注册 w13 缩放参数到层 # register w13 scale parameter to layer
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置 w13 缩放属性 # set w13 scale attributes
        w2_weight_scale = torch.nn.Parameter(  # 创建 w2 权重缩放参数 # create w2 weight scale parameter
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),  # float32 类型的缩放张量 # float32 dtype scale tensor
            requires_grad=False,  # 不需要梯度 # no gradient required
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 注册 w2 缩放参数到层 # register w2 scale parameter to layer
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置 w2 缩放属性 # set w2 scale attributes
        # offset
        # 偏移量
        w13_weight_offset = torch.nn.Parameter(  # 创建 w13 权重偏移参数 # create w13 weight offset parameter
            torch.empty(  # 创建空张量 # create empty tensor
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32  # float32 类型的偏移张量 # float32 dtype offset tensor
            ),
            requires_grad=False,  # 不需要梯度 # no gradient required
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)  # 注册 w13 偏移参数到层 # register w13 offset parameter to layer
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)  # 设置 w13 偏移属性 # set w13 offset attributes
        w2_weight_offset = torch.nn.Parameter(  # 创建 w2 权重偏移参数 # create w2 weight offset parameter
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),  # float32 类型的偏移张量 # float32 dtype offset tensor
            requires_grad=False,  # 不需要梯度 # no gradient required
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)  # 注册 w2 偏移参数到层 # register w2 offset parameter to layer
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)  # 设置 w2 偏移属性 # set w2 offset attributes

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后处理 # process weights after loading
        self.kernel.process_weights_after_loading(layer)  # 调用内核的后处理方法 # call kernel's post-processing method

    def create_moe_runner(  # 创建 MoE 运行器 # create MoE runner
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"  # 目标层和 MoE 运行器配置 # target layer and MoE runner config
    ):
        self.moe_runner_config = moe_runner_config  # 保存 MoE 运行器配置 # save MoE runner config

    def apply_weights(  # 应用量化权重进行 MoE 前向计算 # apply quantized weights for MoE forward computation
        self,
        layer,  # 目标神经网络层 # target neural network layer
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出 # standard dispatch output
    ) -> "CombineInput":  # 返回合并输入 # return combine input
        return self.kernel.apply(layer, dispatch_output)  # 调用内核的 apply 方法执行计算 # call kernel's apply method for computation

    def apply_without_routing_weights(  # 无路由权重应用方法 # apply weights without routing
        self,
        layer,  # 目标神经网络层 # target neural network layer
        hidden_states,  # 隐藏状态张量 # hidden states tensor
        hidden_states_scale,  # 隐藏状态缩放因子 # hidden states scale factor
        group_list_type,  # 分组列表类型 # group list type
        group_list,  # 分组列表 # group list
        output_dtype,  # 输出数据类型 # output data type
    ):
        return self.kernel.apply_without_routing_weights(  # 调用内核的无路由权重应用方法 # call kernel's apply without routing weights method
            layer,  # 目标层 # target layer
            hidden_states,  # 隐藏状态 # hidden states
            hidden_states_scale,  # 隐藏状态缩放 # hidden states scale
            group_list_type,  # 分组列表类型 # group list type
            group_list,  # 分组列表 # group list
            output_dtype,  # 输出数据类型 # output data type
        )
