# SPDX-License-Identifier: Apache-2.0
# Quark W4A4 MXFP4 MoE量化方案实现文件，实现了MoE（混合专家）层的MXFP4格式量化

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any  # 导入类型检查工具

import torch  # 导入PyTorch库

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig  # 导入MoE运行器相关类
from sglang.srt.layers.moe.utils import get_moe_weight_sizes  # 导入MoE权重尺寸计算工具
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme  # 导入Quark MoE量化方案基类
from sglang.srt.utils import (  # 导入工具函数
    get_bool_env_var,  # 获取布尔环境变量
    is_gfx95_supported,  # 检测是否支持gfx95架构
    is_hip,  # 检测是否为HIP平台
    set_weight_attrs,  # 设置权重属性
)

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # 导入token分发器类型
        CombineInput,  # 合并输入类型
        StandardDispatchOutput,  # 标准分发输出类型
    )

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_is_shuffle_moe_mxfp4 = is_gfx95_supported()  # 检测是否支持gfx95，用于决定是否打乱权重

__all__ = ["QuarkW4A4MXFp4MoE"]  # 模块公开导出的类列表

_is_hip = is_hip()  # 检测当前是否为HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用aiter库（需HIP平台且环境变量启用）
if _use_aiter:  # 如果使用aiter
    from aiter.ops.shuffle import shuffle_weight  # 导入权重打乱操作
    from aiter.utility.fp4_utils import e8m0_shuffle  # 导入E8M0格式打乱工具

OCP_MX_BLOCK_SIZE = 32  # OCP MX标准块大小，用于分组量化


class QuarkW4A4MXFp4MoE(QuarkMoEScheme):  # Quark W4A4 MXFP4 MoE量化方案类，继承自QuarkMoEScheme

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):  # 初始化方法
        self.weight_quant = weight_config  # 保存权重量化配置
        self.input_quant = input_config  # 保存输入量化配置

        weight_qscheme = self.weight_quant.get("qscheme")  # 获取权重量化方案
        input_qscheme = self.input_quant.get("qscheme")  # 获取输入量化方案
        if not (weight_qscheme == "per_group" and input_qscheme == "per_group"):  # 检查是否都是按组量化
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}"
            )  # noqa E501  # 对于MX(FP4)融合MoE层，仅支持权重和激活的按组缩放，不支持其他方案

        self.static_input_scales = not self.input_quant.get("is_dynamic")  # 是否使用静态输入缩放因子
        self.with_bias = False  # MoE层不使用偏置

    @classmethod  # 类方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低设备计算能力要求
        return 70  # 最低计算能力为7.0（Volta架构及以上）

    def create_weights(  # 创建权重参数的方法
        self,
        layer: torch.nn.Module,  # 目标层模块
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 每个分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外的权重属性
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported  # 导入MoE权重缩放支持类型

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(  # 计算MoE权重尺寸
            intermediate_size_per_partition,  # 中间层大小
            is_aiter_moe=_use_aiter,  # 是否使用aiter MoE
            is_concat=True,  # 是否拼接
            is_packed=True,  # 是否打包
        )

        # Add the quantization method used (per tensor/grouped/channel)  # 添加使用的量化方法（按张量/按组/按通道）
        # to ensure the weight scales are loaded in properly  # 以确保权重缩放因子正确加载
        extra_weight_attrs.update(
            {
                "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,  # 量化方法为按块
                "weight_padded": weight_padded,  # 权重是否填充
            },
        )

        params_dtype = torch.uint8  # 参数数据类型设为uint8（用于存储MXFP4权重）

        # WEIGHTS  # 权重
        w13_weight = torch.nn.Parameter(  # 创建w13（门控+上投影）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                w13_up_dim,  # 上投影维度
                hidden_size // 2,  # 隐藏维度除以2（4bit打包）
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight)  # 注册w13权重参数到层

        set_weight_attrs(w13_weight, extra_weight_attrs)  # 设置w13权重的额外属性

        w2_weight = torch.nn.Parameter(  # 创建w2（下投影）权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏维度
                w2_down_dim,  # 下投影维度
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w2_weight", w2_weight)  # 注册w2权重参数到层

        set_weight_attrs(w2_weight, extra_weight_attrs)  # 设置w2权重的额外属性

        # WEIGHT_SCALES  # 权重缩放因子
        w13_weight_scale = torch.nn.Parameter(  # 创建w13权重缩放因子参数
            torch.ones(  # 创建全1张量
                num_experts,  # 专家数量维度
                w13_up_dim,  # 上投影维度
                hidden_size // OCP_MX_BLOCK_SIZE,  # 每组的缩放因子数量
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )

        # 1. w2 scale is floor division of inter_dim by blockscale.  # 1. w2缩放因子是中间维度对块缩放因子的整除
        # 2. w2 scale needs to scale up just as w2.  # 2. w2缩放因子需要像w2一样放大
        # We combine 1. and 2. to keep the integer precision.  # 我们结合1和2来保持整数精度
        w2_weight_scale = torch.nn.Parameter(  # 创建w2权重缩放因子参数
            torch.ones(  # 创建全1张量
                num_experts,  # 专家数量维度
                hidden_size,  # 隐藏维度
                (w2_down_dim * 2) // OCP_MX_BLOCK_SIZE,  # w2缩放因子维度（乘2因为打包因子）
                dtype=params_dtype,  # 数据类型
            ),
            requires_grad=False,  # 不需要梯度
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)  # 设置w2缩放因子的额外属性
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)  # 设置w13缩放因子的额外属性

        layer.register_parameter("w13_weight_scale", w13_weight_scale)  # 注册w13缩放因子到层
        layer.register_parameter("w2_weight_scale", w2_weight_scale)  # 注册w2缩放因子到层

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法
        float_dtype = torch.get_default_dtype()  # 获取默认浮点数据类型

        # Pre-shuffle weight scales  # 预打乱权重缩放因子
        s0, s1, _ = layer.w13_weight_scale.shape  # 获取w13缩放因子形状
        w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)  # 重塑为二维视图
        w13_weight_scale = e8m0_shuffle(w13_weight_scale)  # 执行E8M0格式打乱
        # layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)  # 注释掉的参数赋值方式
        layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)  # 恢复原始形状并赋值

        s0, s1, _ = layer.w2_weight_scale.shape  # 获取w2缩放因子形状
        w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)  # 重塑为二维视图
        w2_weight_scale = e8m0_shuffle(w2_weight_scale)  # 执行E8M0格式打乱
        # layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)  # 注释掉的参数赋值方式
        layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)  # 恢复原始形状并赋值

        # Pre-shuffle weight  # 预打乱权重
        if _is_shuffle_moe_mxfp4:  # 如果支持gfx95打乱
            layer.w13_weight.data = shuffle_weight(  # 打乱w13权重
                layer.w13_weight.contiguous(), (16, 16)  # 使用16x16的打乱模式
            )
            layer.w2_weight.data = shuffle_weight(  # 打乱w2权重
                layer.w2_weight.contiguous(), (16, 16)  # 使用16x16的打乱模式
            )
            layer.w13_weight.is_shuffled = True  # 标记w13权重已打乱
            layer.w2_weight.is_shuffled = True  # 标记w2权重已打乱

        if hasattr(layer, "dispatcher"):  # 如果层有分发器属性
            # Weights are stored as torch.uint8 but semantically MXFP4  # 权重以torch.uint8存储但语义上是MXFP4格式
            layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})  # 设置分发器的量化配置

    def create_moe_runner(  # 创建MoE运行器的方法
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        from sglang.srt.layers.moe.utils import (  # 导入MoE工具函数
            get_moe_a2a_backend,  # 获取MoE all-to-all后端
            get_moe_runner_backend,  # 获取MoE运行器后端
        )

        self.moe_runner_config = moe_runner_config  # 保存MoE运行器配置
        moe_runner_backend = get_moe_runner_backend()  # 获取MoE运行器后端类型
        if moe_runner_backend.is_auto() and get_moe_a2a_backend().supports_aiter():  # 如果后端为自动且a2a后端支持aiter
            moe_runner_backend = MoeRunnerBackend.AITER  # 设置为aiter后端

        if moe_runner_backend.is_aiter():  # 如果使用aiter后端
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)  # 创建aiter MoE运行器
        else:
            # TODO(cwan): refactor other backends  # TODO(cwan): 重构其他后端
            pass  # 暂不处理其他后端

    def apply_weights(  # 应用权重执行前向传播的方法
        self,
        layer: torch.nn.Module,  # 目标层模块
        dispatch_output: StandardDispatchOutput,  # 分发输出
    ) -> CombineInput:  # 返回合并输入
        from sglang.srt.layers.moe.moe_runner.aiter import (  # 导入aiter MoE量化相关类
            AiterMoeQuantInfo,  # aiter MoE量化信息类
            AiterQuantType,  # aiter量化类型枚举
        )

        if hasattr(torch, "float4_e2m1fn_x2"):  # 如果PyTorch支持float4_e2m1fn_x2类型
            w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)  # 将w13权重视图转换为float4类型
            w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)  # 将w2权重视图转换为float4类型
        else:
            w13_weight = layer.w13_weight  # 直接使用原始w13权重
            w2_weight = layer.w2_weight  # 直接使用原始w2权重

        if hasattr(layer.w13_weight, "is_shuffled"):  # 如果w13权重有is_shuffled属性
            w13_weight.is_shuffled = True  # 标记w13权重已打乱
            w2_weight.is_shuffled = True  # 标记w2权重已打乱

        quant_info = AiterMoeQuantInfo(  # 创建aiter MoE量化信息
            w13_weight=w13_weight,  # w13权重
            w2_weight=w2_weight,  # w2权重
            quant_type=AiterQuantType.PER_1X32,  # 量化类型为PER_1X32
            w13_scale=layer.w13_weight_scale,  # w13缩放因子
            w2_scale=layer.w2_weight_scale,  # w2缩放因子
            expert_mask=layer.dispatcher.expert_mask_gpu,  # 专家掩码
        )
        return self.runner.run(dispatch_output, quant_info)  # 运行MoE计算并返回结果
