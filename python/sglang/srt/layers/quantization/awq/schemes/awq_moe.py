# AWQ MoE（混合专家）量化方案实现
# 本文件实现了AWQ量化框架下MoE层的量化方案，包括标准GPU方案和昇腾NPU方案
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations  # 启用延迟类型注解评估

from typing import TYPE_CHECKING  # 导入类型检查工具

import torch  # 导入PyTorch库

from sglang.srt.layers.linear import set_weight_attrs  # 导入权重属性设置工具
from sglang.srt.layers.moe import (  # 导入MoE相关组件
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)

from .awq_scheme import AWQMoESchemeBase  # 导入AWQ MoE方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型
    from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMarlinConfig  # 导入AWQ配置类

__all__ = ["AWQMoEScheme", "AWQAscendMoEScheme"]  # 模块公开接口


class AWQMoEScheme(AWQMoESchemeBase):  # AWQ MoE量化方案类，继承自AWQMoESchemeBase
    def __init__(self, quant_config: "AWQMarlinConfig"):  # 初始化方法，接收AWQ Marlin量化配置
        self.quant_config = quant_config  # 保存量化配置
        if self.quant_config.weight_bits != 4:  # 检查是否为4位量化
            raise ValueError("AWQMoEScheme only supports 4bit now.")  # 仅支持4位量化
        self.kernel = self._init_kernel(quant_config)  # 初始化MoE计算核心

    def _init_kernel(self, quant_config: "AWQMarlinConfig"):  # 初始化计算核心方法
        from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (  # 从GPU后端导入AWQ MoE核心
            AWQMoEKernel,
        )

        return AWQMoEKernel(quant_config)  # 返回AWQ MoE核心实例

    def create_weights(  # 创建量化权重参数方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持枚举

        extra_weight_attrs.update(  # 更新额外权重属性
            {
                "is_transposed": True,  # 标记权重已转置
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,  # 设置量化方法为分组量化
            }
        )

        w13_qweight = torch.nn.Parameter(  # 创建w13门控/上投影量化权重参数
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qweight", w13_qweight)  # 注册w13量化权重参数到层
        set_weight_attrs(w13_qweight, extra_weight_attrs)  # 设置w13量化权重属性

        w2_qweight = torch.nn.Parameter(  # 创建w2下投影量化权重参数
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qweight", w2_qweight)  # 注册w2量化权重参数到层
        set_weight_attrs(w2_qweight, extra_weight_attrs)  # 设置w2量化权重属性

        num_groups_w13 = hidden_size // self.quant_config.group_size  # 计算w13的量化分组数量
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size  # 计算w2的量化分组数量

        w13_scales = torch.nn.Parameter(  # 创建w13缩放参数
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_scales", w13_scales)  # 注册w13缩放参数到层
        set_weight_attrs(w13_scales, extra_weight_attrs)  # 设置w13缩放参数属性

        w2_scales = torch.nn.Parameter(  # 创建w2缩放参数
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_scales", w2_scales)  # 注册w2缩放参数到层
        set_weight_attrs(w2_scales, extra_weight_attrs)  # 设置w2缩放参数属性

        w13_qzeros = torch.nn.Parameter(  # 创建w13量化零点参数
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)  # 注册w13量化零点参数到层
        set_weight_attrs(w13_qzeros, extra_weight_attrs)  # 设置w13量化零点属性

        w2_qzeros = torch.nn.Parameter(  # 创建w2量化零点参数
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)  # 注册w2量化零点参数到层
        set_weight_attrs(w2_qzeros, extra_weight_attrs)  # 设置w2量化零点属性

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理方法
        self.kernel.process_weights_after_loading(layer)  # 调用核心的后处理方法

    def create_moe_runner(  # 创建MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        assert get_moe_runner_backend().is_auto()  # 断言MoE后端为自动选择模式
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        self.kernel.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)  # 创建基于Marlin后端的MoE运行器

    def apply_weights(  # 应用量化权重进行前向计算方法
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: "StandardDispatchOutput",  # 分发输出
    ):
        return self.kernel.apply(layer, dispatch_output)  # 调用核心的apply方法执行计算


class AWQAscendMoEScheme(AWQMoEScheme):  # AWQ昇腾NPU MoE量化方案，继承自AWQMoEScheme
    def _init_kernel(self, quant_config: "AWQConfig"):  # 初始化昇腾NPU计算核心方法
        from sglang.srt.hardware_backend.npu.quantization.awq_kernels import (  # 从NPU后端导入AWQ昇腾MoE核心
            AWQAscendMoEKernel,
        )

        return AWQAscendMoEKernel(quant_config)  # 返回昇腾AWQ MoE核心实例

    def create_moe_runner(  # 创建昇腾NPU MoE运行器方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
