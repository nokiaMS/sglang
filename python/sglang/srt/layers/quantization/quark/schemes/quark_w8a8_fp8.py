# SPDX-License-Identifier: Apache-2.0
# Quark W8A8 FP8量化方案实现文件，实现了权重8bit激活8bit的FP8格式量化线性层

from typing import Any, Callable, Optional, cast  # 导入类型提示工具

import torch  # 导入PyTorch库
from torch.nn import Parameter  # 导入神经网络参数类

from sglang.srt.layers.parameter import (  # 导入参数类
    ChannelQuantScaleParameter,  # 通道量化缩放参数
    ModelWeightParameter,  # 模型权重参数
    PerTensorScaleParameter,  # 按张量缩放参数
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 导入FP8 FNUZ格式检测函数
from sglang.srt.layers.quantization.fp8_utils import (  # 导入FP8工具函数
    apply_fp8_linear,  # 应用FP8线性层计算
    cutlass_fp8_supported,  # 检测CUTLASS FP8是否支持
    normalize_e4m3fn_to_e4m3fnuz,  # 将E4M3FN归一化为E4M3FNUZ
)
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme  # 导入Quark线性层量化方案基类
from sglang.srt.layers.quantization.utils import requantize_with_max_scale  # 导入使用最大缩放因子重新量化的工具
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs  # 导入工具函数

__all__ = ["QuarkW8A8Fp8"]  # 模块公开导出的类列表

_is_fp8_fnuz = is_fp8_fnuz()  # 检测是否使用FP8 FNUZ格式
_is_hip = is_hip()  # 检测当前是否为HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用aiter库（需HIP平台且环境变量启用）
if _use_aiter:  # 如果使用aiter
    from aiter.ops.shuffle import shuffle_weight  # 导入权重打乱操作


class QuarkW8A8Fp8(QuarkLinearScheme):  # Quark W8A8 FP8量化方案类，继承自QuarkLinearScheme

    def __init__(  # 初始化方法
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.cutlass_fp8_supported = cutlass_fp8_supported()  # 检测CUTLASS FP8是否受支持
        self.weight_qscheme = cast(str, weight_config.get("qscheme"))  # 获取权重量化方案
        self.is_static_input_scheme: bool = False  # 是否使用静态输入量化方案，默认为False
        self.input_qscheme: Optional[str] = None  # 输入量化方案，默认为None
        if input_config is not None:  # 如果输入配置存在
            self.is_static_input_scheme = not cast(bool, input_config.get("is_dynamic"))  # 判断是否为静态输入方案
            self.input_qscheme = cast(str, input_config.get("qscheme"))  # 获取输入量化方案

        self.per_token = (  # 是否使用逐token量化
            not self.is_static_input_scheme and self.input_qscheme == "per_channel"  # 非静态方案且按通道量化时启用
        )
        self.out_dtype = torch.get_default_dtype()  # 获取默认输出数据类型

    @classmethod  # 类方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低设备计算能力要求
        # lovelace and up  # Lovelace架构（RTX 40系列）及以上
        return 89  # 最低计算能力为8.9

    def process_weights_after_loading(self, layer) -> None:  # 权重加载后的后处理方法
        # If per tensor, when we have a fused module (e.g. QKV) with per  # 如果按张量量化，当我们有一个融合模块（如QKV），每个
        # tensor scales (thus N scales being passed to the kernel),  # 张量有独立的缩放因子（因此N个缩放因子传递给内核），
        # requantize so we can always run per tensor  # 重新量化以始终按张量运行
        if self.weight_qscheme == "per_tensor":  # 如果权重量化方案为按张量
            if _is_fp8_fnuz:  # 如果是FP8 FNUZ格式
                input_scale = getattr(layer, "input_scale", None)  # 获取输入缩放因子
                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(  # 归一化FP8格式
                    weight=layer.weight,  # 权重
                    weight_scale=layer.weight_scale,  # 权重缩放因子
                    input_scale=input_scale,  # 输入缩放因子
                )
                if input_scale is not None:  # 如果输入缩放因子存在
                    layer.input_scale = Parameter(input_scale, requires_grad=False)  # 更新输入缩放因子参数
            else:
                max_w_scale = layer.weight_scale  # 直接使用权重缩放因子
                weight = layer.weight  # 直接使用权重

            max_w_scale, weight = requantize_with_max_scale(  # 使用最大缩放因子重新量化
                weight=weight,  # 权重
                weight_scale=max_w_scale,  # 权重缩放因子
                logical_widths=layer.logical_widths,  # 逻辑宽度
            )

            layer.weight = Parameter(weight.t(), requires_grad=False)  # 转置权重并设为参数
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)  # 设置缩放因子为参数

        # If channelwise, scales are already lined up, so just transpose.  # 如果按通道量化，缩放因子已对齐，只需转置
        elif self.weight_qscheme == "per_channel":  # 如果权重量化方案为按通道
            weight = layer.weight  # 获取权重

            if _is_fp8_fnuz:  # 如果是FP8 FNUZ格式
                input_scale = getattr(layer, "input_scale", None)  # 获取输入缩放因子
                weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(  # 归一化FP8格式
                    weight=weight,  # 权重
                    weight_scale=layer.weight_scale,  # 权重缩放因子
                    input_scale=input_scale,  # 输入缩放因子
                )
                if input_scale is not None:  # 如果输入缩放因子存在
                    layer.input_scale = Parameter(input_scale, requires_grad=False)  # 更新输入缩放因子参数
            else:
                weight_scale = layer.weight_scale.data  # 获取权重缩放因子数据
            if self.per_token:  # 如果使用逐token量化
                weight_scale = weight_scale.view(-1, 1)  # 将缩放因子重塑为列向量
            if _use_aiter:  # 如果使用aiter
                layer.weight = Parameter(
                    shuffle_weight(weight, (16, 16)).t(), requires_grad=False  # 打乱权重后转置并设为参数
                )
            else:
                layer.weight = Parameter(weight.t(), requires_grad=False)  # 转置权重并设为参数
            # required by torch.compile to be torch.nn.Parameter  # torch.compile要求必须是torch.nn.Parameter类型
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)  # 设置缩放因子为参数

        else:
            raise ValueError(f"Unknown quantization scheme {self.weight_qscheme}")  # 未知的量化方案，抛出异常

        # INPUT SCALE  # 输入缩放因子
        if self.is_static_input_scheme:  # 如果使用静态输入方案
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)  # 取最大值作为统一缩放因子
        else:
            layer.input_scale = None  # 动态方案不设输入缩放因子

    def create_weights(  # 创建权重参数的方法
        self,
        layer: torch.nn.Module,  # 目标层模块
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size_per_partition: int,  # 每个分区的输入大小
        params_dtype: torch.dtype,  # 参数数据类型
        weight_loader: Callable,  # 权重加载器回调
        **kwargs,  # 其他关键字参数
    ):
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出总大小
        layer.logical_widths = output_partition_sizes  # 设置层的逻辑宽度

        # WEIGHT  # 权重
        weight = ModelWeightParameter(  # 创建模型权重参数
            data=torch.empty(  # 创建空张量
                output_size_per_partition,  # 输出维度大小
                input_size_per_partition,  # 输入维度大小
                dtype=torch.float8_e4m3fn,  # FP8 E4M3数据类型
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight", weight)  # 注册权重参数到层

        # WEIGHT SCALE  # 权重缩放因子
        if self.weight_qscheme == "per_channel":  # 如果按通道量化
            weight_scale = ChannelQuantScaleParameter(  # 创建通道量化缩放参数
                data=torch.empty((sum(output_partition_sizes)), dtype=torch.float32),  # 创建空张量，每个输出通道一个缩放值
                output_dim=0,  # 输出维度索引
                weight_loader=weight_loader,  # 权重加载器
            )
        else:
            assert self.weight_qscheme == "per_tensor"  # 断言必须是按张量量化
            weight_scale = PerTensorScaleParameter(  # 创建按张量缩放参数
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),  # 创建空张量，每个分区一个缩放值
                weight_loader=weight_loader,  # 权重加载器
            )
            set_weight_attrs(weight_scale, {"needs_scalar_to_array": True})  # 设置需要标量转数组的属性

        # min requirement for fp8 kernels  # FP8内核的最低要求
        weight_scale[:] = torch.finfo(torch.float32).min  # 用float32最小值初始化缩放因子
        layer.register_parameter("weight_scale", weight_scale)  # 注册权重缩放因子到层

        # INPUT SCALE  # 输入缩放因子
        if self.is_static_input_scheme:  # 如果使用静态输入方案
            input_scale = PerTensorScaleParameter(  # 创建按张量缩放参数
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),  # 创建空张量，每个分区一个缩放值
                weight_loader=weight_loader,  # 权重加载器
            )
            input_scale[:] = torch.finfo(torch.float32).min  # 用float32最小值初始化输入缩放因子
            set_weight_attrs(input_scale, {"needs_scalar_to_array": True})  # 设置需要标量转数组的属性
            layer.register_parameter("input_scale", input_scale)  # 注册输入缩放因子到层

    def apply_weights(  # 应用权重执行前向传播的方法
        self,
        layer: torch.nn.Module,  # 目标层模块
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量（可选）
    ) -> torch.Tensor:  # 返回输出张量

        return apply_fp8_linear(  # 调用FP8线性层计算函数
            x,  # 输入张量
            layer.weight,  # 权重
            layer.weight_scale,  # 权重缩放因子
            input_scale=layer.input_scale,  # 输入缩放因子
            bias=bias,  # 偏置
            cutlass_fp8_supported=self.cutlass_fp8_supported,  # CUTLASS FP8是否受支持
            use_per_token_if_dynamic=self.per_token,  # 动态模式下是否使用逐token量化
        )
