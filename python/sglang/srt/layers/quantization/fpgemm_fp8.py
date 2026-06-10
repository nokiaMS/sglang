# FBGEMM FP8量化配置与线性方法模块 - 实现FBGEMM格式的FP8量化和线性层推理
# 支持CUTLASS FP8和Marlin FP8两种后端，自动检测硬件能力选择最优实现

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from typing import Any, List, Optional  # 导入类型注解

import torch  # 导入PyTorch
from torch.nn import Module  # 导入神经网络模块基类
from torch.nn.parameter import Parameter  # 导入参数类

from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter  # 导入量化参数类
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 导入FP8 FNUZ格式检测函数
from sglang.srt.layers.quantization.fp8_utils import (  # 导入FP8工具函数
    apply_fp8_linear,  # 应用FP8线性运算
    can_auto_enable_marlin_fp8,  # 检测是否可自动启用Marlin FP8
    cutlass_fp8_supported,  # 检测CUTLASS FP8是否受支持
    normalize_e4m3fn_to_e4m3fnuz,  # 将E4M3FN格式归一化为E4M3FNUZ
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (  # 导入Marlin FP8工具函数
    apply_fp8_marlin_linear,  # 应用Marlin FP8线性运算
    prepare_fp8_layer_for_marlin,  # 为Marlin准备FP8层
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化的线性方法
from sglang.srt.layers.quantization.utils import is_layer_skipped  # 导入层跳过检测函数
from sglang.srt.utils import get_bool_env_var, is_cuda  # 导入环境变量和CUDA检测工具

_is_cuda = is_cuda()  # 检测是否为CUDA环境
_is_fp8_fnuz = is_fp8_fnuz()  # 检测FP8是否为FNUZ格式

logger = logging.getLogger(__name__)  # 创建日志记录器


class FBGEMMFp8Config(QuantizationConfig):  # FBGEMM FP8量化配置类
    """Config class for FBGEMM Fp8."""  # FBGEMM FP8的配置类

    def __init__(self, ignore_list: list[str], input_scale_ub: float):  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.ignore_list = ignore_list if ignore_list else []  # 不进行量化的层列表
        self.input_scale_ub = input_scale_ub  # 输入缩放因子上界

        # For GPUs that lack FP8 hardware suspport, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        # self.use_marlin = not marlin_fp8_supported()  # 对于不支持FP8硬件的GPU，可以使用Marlin内核进行快速仅权重FP8量化
        self.use_marlin = False  # 默认不使用Marlin
        if _is_cuda:  # 如果是CUDA环境
            force_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")  # 检查是否强制使用Marlin
            auto_enable = can_auto_enable_marlin_fp8()  # 检查是否可自动启用Marlin FP8
            self.use_marlin = force_marlin or auto_enable  # 强制或自动启用时使用Marlin

    @classmethod  # 类方法装饰器
    def get_name(cls) -> str:  # 获取量化方法名称
        return "fbgemm_fp8"  # 返回FBGEMM FP8

    @classmethod  # 类方法装饰器
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.bfloat16, torch.float16]  # 支持bfloat16和float16

    @classmethod  # 类方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低GPU计算能力要求
        return 80  # 最低要求SM80

    @classmethod  # 类方法装饰器
    def get_config_filenames(cls) -> list[str]:  # 获取配置文件名列表
        return []  # 无额外配置文件

    @classmethod  # 类方法装饰器
    def from_config(cls, config: dict[str, Any]) -> FBGEMMFp8Config:  # 从配置字典创建配置对象
        ignore_list = cls.get_from_keys(config, ["modules_to_not_convert"])  # 获取不转换的模块列表
        input_scale_ub = cls.get_from_keys(config, ["activation_scale_ub"])  # 获取激活缩放上界
        return cls(ignore_list=ignore_list, input_scale_ub=input_scale_ub)  # 创建并返回配置对象

    def get_quant_method(  # 获取量化方法
        self, layer: torch.nn.Module, prefix: str  # 层对象和前缀
    ) -> Optional[QuantizeMethodBase]:  # 返回量化方法或None
        if isinstance(layer, LinearBase):  # 如果是线性层
            if is_layer_skipped(  # 检查该层是否应跳过量化
                prefix=prefix,  # 层前缀
                ignored_layers=self.ignore_list,  # 忽略列表
                fused_mapping=self.packed_modules_mapping,  # 融合模块映射
            ):
                return UnquantizedLinearMethod()  # 返回未量化的线性方法
            return FBGEMMFp8LinearMethod(self)  # 返回FBGEMM FP8线性方法
        return None  # 非线性层返回None

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称
        return []  # 无需缩放的激活


class FBGEMMFp8LinearMethod(LinearMethodBase):  # FBGEMM FP8线性方法类

    def __init__(self, quant_config: FBGEMMFp8Config):  # 初始化函数
        """初始化FBGEMM FP8线性方法，设置量化配置和CUTLASS支持状态。"""  # 中文函数说明
        self.quant_config = quant_config  # 保存量化配置
        # self.fp8_linear = Fp8LinearOp(
        #     act_quant_static=False, act_quant_group_shape=GroupShape.PER_TOKEN)
        self.out_dtype = torch.get_default_dtype()  # 获取默认输出数据类型
        self.cutlass_fp8_supported = cutlass_fp8_supported()  # 检测CUTLASS FP8是否受支持

    def create_weights(  # 创建权重参数
        self,
        layer: torch.nn.Module,  # 神经网络层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size: int,  # 总输入大小
        output_size: int,  # 总输出大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        """创建FP8线性层的权重、缩放因子和输入缩放上界参数。"""  # 中文函数说明
        # maybe_create_device_identity()  # 可能创建设备身份张量
        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器
        del input_size, output_size  # 删除不需要的变量
        output_size_per_partition = sum(output_partition_sizes)  # 计算每个分区的输出总大小

        layer.logical_widths = output_partition_sizes  # 保存逻辑宽度

        layer.input_size_per_partition = input_size_per_partition  # 保存每个分区的输入大小
        layer.output_size_per_partition = output_size_per_partition  # 保存每个分区的输出大小
        layer.orig_dtype = params_dtype  # 保存原始数据类型

        # WEIGHT  # 权重参数
        weight = ModelWeightParameter(  # 创建模型权重参数
            data=torch.empty(  # 创建空张量
                output_size_per_partition,  # 输出行数
                input_size_per_partition,  # 输入列数
                dtype=torch.float8_e4m3fn,  # FP8 E4M3数据类型
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight", weight)  # 注册权重参数

        # WEIGHT SCALE  # 权重缩放因子
        weight_scale = ChannelQuantScaleParameter(  # 创建通道量化缩放参数
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),  # 每个输出通道一个缩放值
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        weight_scale[:] = torch.finfo(torch.float32).min  # 初始化为float32最小值
        layer.register_parameter("weight_scale", weight_scale)  # 注册权重缩放因子

        # INPUT SCALE UPPER BOUND  # 输入缩放上界
        input_scale_ub = torch.nn.Parameter(  # 创建输入缩放上界参数
            torch.tensor((self.quant_config.input_scale_ub), dtype=torch.float32),  # 从配置获取上界值
            requires_grad=False,  # 不需要梯度
        )
        layer.input_scale_ub = input_scale_ub  # 保存到层中

    def process_weights_after_loading(self, layer: Module) -> None:  # 加载后处理权重
        """权重加载后的处理：设置requires_grad、FNUZ格式转换和权重转置。"""  # 中文函数说明
        # required by torch.compile  # torch.compile需要
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)  # 确保缩放因子不需要梯度
        layer.weight = Parameter(layer.weight.data, requires_grad=False)  # 确保权重不需要梯度

        weight = layer.weight  # 获取权重

        if _is_fp8_fnuz:  # 如果FP8格式为FNUZ
            weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(  # 将E4M3FN转换为E4M3FNUZ
                weight=weight, weight_scale=layer.weight_scale, input_scale=None
            )
            if input_scale is not None:  # 如果存在输入缩放因子
                layer.input_scale = Parameter(input_scale, requires_grad=False)  # 保存转换后的输入缩放
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)  # 保存转换后的权重缩放

        layer.weight = Parameter(weight.t(), requires_grad=False)  # 转置权重并设为不需要梯度
        if self.quant_config.use_marlin:  # 如果使用Marlin
            prepare_fp8_layer_for_marlin(layer)  # 为Marlin准备FP8层
            # Activations not quantized for marlin.  # Marlin模式下不量化激活
            del layer.input_scale_ub  # 删除输入缩放上界

    def apply(  # 应用线性运算
        self,
        layer: torch.nn.Module,  # 神经网络层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置项
    ) -> torch.Tensor:  # 返回输出张量
        """执行FBGEMM FP8线性推理，支持Marlin和CUTLASS两种后端。"""  # 中文函数说明

        if self.quant_config.use_marlin:  # 如果使用Marlin后端
            return apply_fp8_marlin_linear(  # 使用Marlin FP8线性运算
                input=x,  # 输入
                weight=layer.weight,  # 权重
                weight_scale=layer.weight_scale,  # 权重缩放
                workspace=layer.workspace,  # Marlin工作空间
                size_n=layer.output_size_per_partition,  # 输出维度N
                size_k=layer.input_size_per_partition,  # 输入维度K
                bias=bias,  # 偏置
            )

        return apply_fp8_linear(  # 使用标准FP8线性运算
            input=x,  # 输入
            weight=layer.weight,  # 权重
            weight_scale=layer.weight_scale,  # 权重缩放
            input_scale=None,  # 无预计算的输入缩放
            input_scale_ub=layer.input_scale_ub,  # 输入缩放上界
            bias=bias,  # 偏置
            cutlass_fp8_supported=self.cutlass_fp8_supported,  # CUTLASS是否受支持
            use_per_token_if_dynamic=False,  # 动态量化时使用逐张量量化
        )
