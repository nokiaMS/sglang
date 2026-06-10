# 压缩张量W8A8 INT8 MoE量化方案
# 本文件实现了NPU平台上W8A8 INT8动态量化的MoE（混合专家）层，
# 支持逐通道权重量化和逐token动态输入量化。

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch深度学习框架
from compressed_tensors.quantization import QuantizationStrategy  # 导入量化策略枚举

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (  # 导入NPU W8A8 INT8动态MoE方法
    NPUW8A8Int8DynamicMoEMethod,  # NPU W8A8 INT8动态MoE方法类
)
from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # 导入压缩张量MoE方案基类
    CompressedTensorsMoEScheme,  # 压缩张量MoE量化方案基类
)
from sglang.srt.utils import set_weight_attrs  # 导入设置权重属性的工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器相关类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )

__all__ = ["NPUCompressedTensorsW8A8Int8DynamicMoE"]  # 模块公开导出列表

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class NPUCompressedTensorsW8A8Int8DynamicMoE(CompressedTensorsMoEScheme):
    """NPU平台上的W8A8 INT8动态量化MoE方案，要求逐通道权重量化和逐token动态输入量化。"""

    def __init__(self, weight_quant, input_quant):  # 初始化方法，接收权重量化配置和输入量化配置
        self.weight_quant = weight_quant  # 保存权重量化配置
        self.input_quant = input_quant  # 保存输入量化配置
        self.kernel = NPUW8A8Int8DynamicMoEMethod()  # 创建NPU W8A8 INT8动态MoE内核实例

        self.static_input_scales = not self.input_quant.dynamic  # 判断输入缩放因子是否为静态（非动态）
        per_channel = (  # 检查是否为逐通道+逐token量化
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL  # 权重量化策略是否为逐通道
            and self.input_quant.strategy == QuantizationStrategy.TOKEN  # 输入量化策略是否为逐token
        )
        if not per_channel:  # 如果不是逐通道+逐token量化
            raise ValueError(  # 抛出值错误
                "For INT8 Fused MoE layers, we require channelwise, "  # INT8融合MoE层需要逐通道
                "dynamic per token quantization. Found "  # 动态逐token量化。发现
                f"{self.weight_quant}, {self.input_quant}"  # 当前的权重和输入量化配置
            )

        self.static_input_scales = not self.input_quant.dynamic  # 再次判断输入缩放因子是否为静态
        if self.static_input_scales:  # 如果输入缩放因子为静态
            raise ValueError(  # 抛出值错误
                "For INT8 Fused MoE layers, we require channelwise, "  # INT8融合MoE层需要逐通道
                "dynamic per token quantization. Found static input scales."  # 动态逐token量化。发现静态输入缩放因子。
            )

    def create_weights(  # 创建权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外的权重属性关键字参数
    ):
        """为MoE层创建并注册W8A8 INT8量化所需的权重参数。"""

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入融合MoE权重缩放支持枚举

        params_dtype = torch.int8  # 强制使用int8数据类型

        # WEIGHTS  # 权重
        w13_weight = torch.nn.Parameter(  # 创建w13（门控+上投影）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                2 * intermediate_size_per_partition,  # 2倍中间层大小（门控+上投影）
                hidden_size,  # 隐藏层大小维度
                dtype=params_dtype,  # 使用int8数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 将w13权重注册到层中
        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置w13权重的额外属性

        w2_weight = torch.nn.Parameter(  # 创建w2（下投影）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小维度
                intermediate_size_per_partition,  # 中间层大小维度
                dtype=params_dtype,  # 使用int8数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 将w2权重注册到层中
        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置w2权重的额外属性

        # WEIGHT_SCALES  # 权重缩放因子
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL  # 断言权重量化策略为逐通道
        w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.ones(  # 创建全1张量
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32  # 形状为[专家数, 2*中间层大小, 1]，float32类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 将w13权重缩放因子注册到层中
        w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),  # 形状为[专家数, 隐藏层大小, 1]，float32类型
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 将w2权重缩放因子注册到层中
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.  # 为FusedMoE的weight_loader添加逐通道量化标记
        extra_weight_attrs.update(  # 更新额外权重属性
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}  # 设置量化方法为逐通道
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置w13权重缩放因子的额外属性
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置w2权重缩放因子的额外属性

        # INPUT_SCALES  # 输入缩放因子
        assert not self.static_input_scales  # 断言输入缩放因子不是静态的
        layer.w13_input_scale = None  # w13输入缩放因子设为None（动态计算）
        layer.w2_input_scale = None  # w2输入缩放因子设为None（动态计算）

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后的后处理方法
        """权重加载后的后处理，委托给内核实现。"""
        self.kernel.process_weights_after_loading(layer)  # 调用内核的权重后处理方法

    def create_moe_runner(  # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):
        """创建MoE运行器，保存运行器配置。"""
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置

    def apply_weights(  # 应用权重进行前向计算方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:  # 返回合并输入
        """应用量化权重进行MoE前向计算，委托给内核实现。"""

        return self.kernel.apply(layer, dispatch_output)  # 调用内核的apply方法执行前向计算
