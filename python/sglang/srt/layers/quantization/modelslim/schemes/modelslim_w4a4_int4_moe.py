# ModelSlim W4A4 Int4 MoE量化方案实现文件，用于MoE（混合专家）层的4比特权重4比特激活动态量化
from __future__ import annotations  # 启用延迟注解评估

import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any, Dict  # 导入类型检查和类型注解工具

import torch  # 导入PyTorch库

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (  # 从NPU量化模块导入W4A4 Int4动态MoE方法
    NPUW4A4Int4DynamicMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme  # 导入ModelSlim MoE层量化方案基类
from sglang.srt.utils import set_weight_attrs  # 导入设置权重属性的工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入令牌分发器相关类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = [  # 定义模块的公开导出列表
    "ModelSlimW4A4Int4MoE",  # W4A4 Int4 MoE量化方案类
]


class ModelSlimW4A4Int4MoE(ModelSlimMoEScheme):  # W4A4 Int4 MoE量化方案类，继承自ModelSlimMoEScheme

    def __init__(  # 初始化方法
        self,
        quant_config: Dict[str, Any],  # 量化配置字典
        prefix: str = None,  # 参数前缀，默认为None
    ):
        self.quant_config = quant_config  # 保存量化配置
        self.kernel = NPUW4A4Int4DynamicMoEMethod()  # 初始化NPU W4A4 Int4动态MoE方法内核

    def create_weights(  # 创建权重并注册到层中
        self,
        layer: torch.nn.Module,  # 目标层模块
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入融合MoE权重缩放支持枚举

        self.num_experts = num_experts  # 保存专家数量
        extra_weight_attrs.update(  # 更新额外权重属性
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}  # 设置量化方法为逐通道缩放
        )

        # weight  权重
        w13_weight = torch.nn.Parameter(  # 创建w13权重参数（门控和上投影的拼接权重）
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 2倍中间层大小（gate+up拼接）
                hidden_size,  # 隐藏层大小
                dtype=torch.int8,  # int8数据类型
            ),
            requires_grad=False,  # 不可训练
        )
        layer.register_parameter("w13_weight", w13_weight)  # 将w13权重注册到层中
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置额外权重属性
        w2_weight = torch.nn.Parameter(  # 创建w2权重参数（下投影权重）
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小
                intermediate_size_per_partition,  # 中间层分区大小
                dtype=torch.int8,  # int8数据类型
            ),
            requires_grad=False,  # 不可训练
        )
        layer.register_parameter("w2_weight", w2_weight)  # 将w2权重注册到层中
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置额外权重属性
        # scale  缩放因子
        w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.empty(  # 创建空张量
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32  # float32类型，形状为(专家数, 2*中间层大小, 1)
            ),
            requires_grad=False,  # 不可训练
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 将w13缩放因子注册到层中
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置额外权重属性
        w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),  # float32类型，形状为(专家数, 隐藏层大小, 1)
            requires_grad=False,  # 不可训练
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 将w2缩放因子注册到层中
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置额外权重属性
        # offset  偏移量
        w13_weight_offset = torch.nn.Parameter(  # 创建w13权重偏移量参数
            torch.empty(  # 创建空张量
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32  # float32类型，形状为(专家数, 2*中间层大小, 1)
            ),
            requires_grad=False,  # 不可训练
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)  # 将w13偏移量注册到层中
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)  # 设置额外权重属性
        w2_weight_offset = torch.nn.Parameter(  # 创建w2权重偏移量参数
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),  # float32类型，形状为(专家数, 隐藏层大小, 1)
            requires_grad=False,  # 不可训练
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)  # 将w2偏移量注册到层中
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)  # 设置额外权重属性

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理，委托给内核执行
        self.kernel.process_weights_after_loading(layer)  # 调用内核的后处理方法

    def create_moe_runner(  # 创建MoE运行器的方法
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"  # layer:含权重的模块, moe_runner_config:MoE运行器配置
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def apply_weights(  # 应用权重执行前向传播
        self,
        layer,  # 含权重的层模块
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    ) -> "CombineInput":
        return self.kernel.apply(layer, dispatch_output)  # 调用内核的apply方法执行计算并返回合并输入

    def apply_without_routing_weights(  # 不使用路由权重的应用方法（用于DeepEP分发器场景）
        self,
        layer,  # 含权重的层模块
        hidden_states,  # 隐藏状态张量
        hidden_states_scale,  # 隐藏状态缩放因子
        group_list_type,  # 组列表类型
        group_list,  # 组列表
        output_dtype,  # 输出数据类型
    ):
        logger.warning_once(  # 记录一次性警告日志
            "Warning: Performance may be reduced, because DeepEP Dispatcher does not support 4-bit quantization, "
            "switching to the bf16 dispatcher, quantization will be performed separately..."
            # 警告：性能可能降低，因为DeepEP分发器不支持4比特量化，切换到bf16分发器，量化将单独执行...
        )
        return self.kernel.apply_without_routing_weights(  # 调用内核的不带路由权重的方法
            layer,  # 层模块
            hidden_states,  # 隐藏状态
            hidden_states_scale,  # 隐藏状态缩放因子
            group_list_type,  # 组列表类型
            group_list,  # 组列表
            output_dtype,  # 输出数据类型
        )
