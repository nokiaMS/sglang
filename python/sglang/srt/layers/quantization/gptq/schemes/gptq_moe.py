# SPDX-License-Identifier: Apache-2.0
# 文件说明：GPTQ MoE（混合专家）量化方案实现，包含Ascend NPU和Marlin GPU两种MoE量化方案
from __future__ import annotations  # 启用延迟类型注解求值

from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.linear import set_weight_attrs  # 导入设置权重属性的工具函数
from sglang.srt.layers.moe import MoeRunnerConfig  # 导入MoE运行器配置类

from .gptq_scheme import GPTQMoESchemeBase  # 导入GPTQ MoE方案基类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 导入标准分发输出类型
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig, GPTQMarlinConfig  # 导入GPTQ配置类

__all__ = ["GPTQMoEAscendScheme", "GPTQMarlinMoEScheme"]  # 模块公开接口，导出两种MoE量化方案


class GPTQMoEAscendScheme(GPTQMoESchemeBase):  # GPTQ MoE Ascend NPU量化方案类，继承自GPTQ MoE方案基类
    def __init__(self, quant_config: "GPTQConfig"):  # 初始化方法，接收GPTQ量化配置
        self.quant_config = quant_config  # 保存量化配置
        from sglang.srt.hardware_backend.npu.quantization.gptq_kernels import (  # 延迟导入Ascend NPU的GPTQ MoE核
            GPTQMoEAscendKernel,  # GPTQ MoE Ascend计算核
        )

        self.kernel = GPTQMoEAscendKernel(quant_config)  # 创建GPTQ MoE Ascend计算核实例

    def create_weights(  # 创建MoE量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持枚举

        pack_factor = self.quant_config.pack_factor  # 获取打包因子

        num_groups_w13 = hidden_size // self.quant_config.group_size  # 计算w13权重的分组数量
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size  # 计算w2权重的分组数量

        extra_weight_attrs.update(  # 更新额外权重属性
            {
                "is_transposed": True,  # 权重已转置
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,  # 量化方法为分组量化
            }
        )

        w13_qweight = torch.nn.Parameter(  # 创建w13（gate_up）量化权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                hidden_size // pack_factor,  # 打包后的隐藏层大小
                2 * intermediate_size_per_partition,  # gate和up拼接的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qweight", w13_qweight)  # 注册w13量化权重参数
        set_weight_attrs(w13_qweight, extra_weight_attrs)  # 设置w13量化权重的额外属性

        w2_qweight = torch.nn.Parameter(  # 创建w2（down）量化权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition // pack_factor,  # 打包后的中间层大小
                hidden_size,  # 隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qweight", w2_qweight)  # 注册w2量化权重参数
        set_weight_attrs(w2_qweight, extra_weight_attrs)  # 设置w2量化权重的额外属性

        w13_scales = torch.nn.Parameter(  # 创建w13缩放因子参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                num_groups_w13,  # w13分组数量
                2 * intermediate_size_per_partition,  # gate和up拼接的中间层大小
                dtype=params_dtype,  # 参数数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_scales", w13_scales)  # 注册w13缩放因子参数
        set_weight_attrs(w13_scales, extra_weight_attrs)  # 设置w13缩放因子的额外属性

        w2_scales = torch.nn.Parameter(  # 创建w2缩放因子参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                num_groups_w2,  # w2分组数量
                hidden_size,  # 隐藏层大小
                dtype=params_dtype,  # 参数数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_scales", w2_scales)  # 注册w2缩放因子参数
        set_weight_attrs(w2_scales, extra_weight_attrs)  # 设置w2缩放因子的额外属性

        w13_qzeros = torch.nn.Parameter(  # 创建w13量化零点参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                num_groups_w13,  # w13分组数量
                2 * intermediate_size_per_partition // pack_factor,  # 打包后的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)  # 注册w13量化零点参数
        set_weight_attrs(w13_qzeros, extra_weight_attrs)  # 设置w13量化零点的额外属性

        w2_qzeros = torch.nn.Parameter(  # 创建w2量化零点参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                num_groups_w2,  # w2分组数量
                hidden_size // pack_factor,  # 打包后的隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)  # 注册w2量化零点参数
        set_weight_attrs(w2_qzeros, extra_weight_attrs)  # 设置w2量化零点的额外属性

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):
        self.kernel.create_moe_runner(layer, moe_runner_config)  # 调用内核创建MoE运行器

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法
        self.kernel.process_weights_after_loading(layer)  # 调用内核的权重加载后处理方法

    def apply_weights(  # 应用量化权重进行MoE前向计算
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    ):
        return self.kernel.apply(layer, dispatch_output)  # 调用内核的apply方法执行MoE量化计算


class GPTQMarlinMoEScheme(GPTQMoESchemeBase):  # GPTQ Marlin MoE GPU量化方案类，继承自GPTQ MoE方案基类
    def __init__(self, quant_config: "GPTQMarlinConfig"):  # 初始化方法，接收GPTQ Marlin量化配置
        self.quant_config = quant_config  # 保存量化配置
        from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (  # 延迟导入GPU的GPTQ Marlin MoE核
            GPTQMarlinMoEKernel,  # GPTQ Marlin MoE计算核
        )

        self.kernel = GPTQMarlinMoEKernel(quant_config)  # 创建GPTQ Marlin MoE计算核实例

    def create_weights(  # 创建MoE量化权重参数
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持枚举

        self.kernel.is_k_full = (  # 判断是否使用完整k维度（非激活值降序排列或单卡时为True）
            not self.quant_config.desc_act
        ) or layer.moe_tp_size == 1

        if self.quant_config.group_size != -1:  # 如果分组大小不为-1（即使用分组量化）
            scales_size13 = hidden_size // self.quant_config.group_size  # 计算w13缩放因子大小
            if self.quant_config.desc_act:  # 如果使用激活值降序排列
                w2_scales_size = intermediate_size_per_partition  # w2缩放大小使用分区中间层大小
            else:
                w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size  # w2缩放大小使用完整中间层大小
            scales_size2 = w2_scales_size // self.quant_config.group_size  # 计算w2缩放因子大小
            strategy = FusedMoeWeightScaleSupported.GROUP.value  # 量化策略为分组量化
        else:  # 分组大小为-1，使用通道级量化
            scales_size13 = 1  # 通道级量化缩放大小为1
            scales_size2 = 1  # 通道级量化缩放大小为1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value  # 量化策略为通道量化

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})  # 更新额外权重属性

        w13_qweight = torch.nn.Parameter(  # 创建w13（gate_up）量化权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                hidden_size // self.quant_config.pack_factor,  # 打包后的隐藏层大小
                2 * intermediate_size_per_partition,  # gate和up拼接的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qweight", w13_qweight)  # 注册w13量化权重参数
        set_weight_attrs(w13_qweight, extra_weight_attrs)  # 设置w13量化权重的额外属性

        w2_qweight = torch.nn.Parameter(  # 创建w2（down）量化权重参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition // self.quant_config.pack_factor,  # 打包后的中间层大小
                hidden_size,  # 隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qweight", w2_qweight)  # 注册w2量化权重参数
        set_weight_attrs(w2_qweight, extra_weight_attrs)  # 设置w2量化权重的额外属性

        w13_scales = torch.nn.Parameter(  # 创建w13缩放因子参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                scales_size13,  # w13缩放大小
                2 * intermediate_size_per_partition,  # gate和up拼接的中间层大小
                dtype=torch.half,  # 半精度浮点类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_scales", w13_scales)  # 注册w13缩放因子参数
        set_weight_attrs(w13_scales, extra_weight_attrs)  # 设置w13缩放因子的额外属性

        w2_scales = torch.nn.Parameter(  # 创建w2缩放因子参数
            torch.empty(num_experts, scales_size2, hidden_size, dtype=torch.half),  # 分配空张量
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_scales", w2_scales)  # 注册w2缩放因子参数
        set_weight_attrs(w2_scales, extra_weight_attrs)  # 设置w2缩放因子的额外属性
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})  # 如果使用激活值降序排列，则需要加载完整w2

        w13_qzeros = torch.nn.Parameter(  # 创建w13量化零点参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                scales_size13,  # w13缩放大小
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,  # 打包后的中间层大小
                dtype=params_dtype,  # 参数数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)  # 注册w13量化零点参数
        set_weight_attrs(w13_qzeros, extra_weight_attrs)  # 设置w13量化零点的额外属性

        w2_qzeros = torch.nn.Parameter(  # 创建w2量化零点参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                scales_size2,  # w2缩放大小
                hidden_size // self.quant_config.pack_factor,  # 打包后的隐藏层大小
                dtype=params_dtype,  # 参数数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)  # 注册w2量化零点参数
        set_weight_attrs(w2_qzeros, extra_weight_attrs)  # 设置w2量化零点的额外属性
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})  # 如果使用激活值降序排列，则需要加载完整w2

        w13_g_idx = torch.nn.Parameter(  # 创建w13分组索引参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)  # 注册w13分组索引参数
        set_weight_attrs(w13_g_idx, extra_weight_attrs)  # 设置w13分组索引的额外属性

        w2_g_idx = torch.nn.Parameter(  # 创建w2分组索引参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition,  # 每个分区的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)  # 注册w2分组索引参数
        set_weight_attrs(w2_g_idx, extra_weight_attrs)  # 设置w2分组索引的额外属性

        w13_g_idx_sort_indices = torch.nn.Parameter(  # 创建w13分组索引排序索引参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)  # 注册w13分组索引排序索引参数
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)  # 设置w13分组索引排序索引的额外属性

        w2_g_idx_sort_indices = torch.nn.Parameter(  # 创建w2分组索引排序索引参数
            torch.empty(  # 分配空张量
                num_experts,  # 专家数量维度
                intermediate_size_per_partition,  # 每个分区的中间层大小
                dtype=torch.int32,  # 32位整数类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)  # 注册w2分组索引排序索引参数
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)  # 设置w2分组索引排序索引的额外属性

    def create_moe_runner(  # 创建MoE运行器
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig  # 目标层和MoE运行器配置
    ):
        self.kernel.create_moe_runner(layer, moe_runner_config)  # 调用内核创建MoE运行器

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的处理方法
        self.kernel.process_weights_after_loading(layer)  # 调用内核的权重加载后处理方法

    def apply_weights(  # 应用量化权重进行MoE前向计算
        self,
        layer: torch.nn.Module,  # 目标神经网络层
        dispatch_output: "StandardDispatchOutput",  # 标准分发输出
    ):
        return self.kernel.apply(layer, dispatch_output)  # 调用内核的apply方法执行MoE量化计算
