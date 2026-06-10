# 压缩张量W8A8 FP8量化方案实现
# 本文件实现了CompressedTensors框架下W8A8（权重8比特FP8、激活8比特FP8）量化方案
# 支持逐张量、逐通道和分块三种量化策略，并支持FNUZ格式归一化

# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# 适配自 vLLM 项目的压缩张量量化实现
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX文件版权文本：版权归属vLLM项目的贡献者

from typing import Callable, Optional # 导入类型提示工具

import torch # 导入PyTorch库
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy # 导入量化参数类和策略枚举
from torch.nn import Parameter # 导入神经网络参数类

from sglang.srt.layers.parameter import ( # 导入参数类
    BlockQuantScaleParameter, # 分块量化缩放参数
    ChannelQuantScaleParameter, # 逐通道量化缩放参数
    ModelWeightParameter, # 模型权重参数
    PerTensorScaleParameter, # 逐张量缩放参数
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import ( # 导入压缩张量线性方案基类
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz # 导入FP8 FNUZ格式检测函数
from sglang.srt.layers.quantization.fp8_utils import ( # 导入FP8工具函数
    apply_fp8_linear, # 应用FP8线性计算
    apply_fp8_ptpc_linear, # 应用FP8逐token逐通道线性计算
    dispatch_w8a8_block_fp8_linear, # 分发W8A8分块FP8线性计算
    normalize_e4m3fn_to_e4m3fnuz, # 将E4M3FN格式归一化为E4M3FNUZ格式
    validate_fp8_block_shape, # 验证FP8分块形状
)
from sglang.srt.layers.quantization.utils import requantize_with_max_scale # 导入使用最大缩放重新量化工具
from sglang.srt.utils import get_bool_env_var, is_hip # 导入环境变量获取和HIP平台检测工具

__all__ = ["CompressedTensorsW8A8Fp8"] # 模块公开接口列表

_is_hip = is_hip() # 检测是否为AMD HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip # 是否使用AITER（仅HIP平台支持）
if _use_aiter: # 如果使用AITER
    from aiter.ops.shuffle import shuffle_weight # 导入AITER权重重排函数


strategy_to_parameter_type = { # 量化策略到参数类型的映射字典
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter, # 分块策略 -> 分块量化缩放参数
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter, # 逐通道策略 -> 逐通道量化缩放参数
    QuantizationStrategy.TENSOR: PerTensorScaleParameter, # 逐张量策略 -> 逐张量缩放参数
}


class CompressedTensorsW8A8Fp8(CompressedTensorsLinearScheme): # 压缩张量W8A8 FP8量化方案类
    """W8A8 FP8量化方案：权重和激活均使用FP8（8比特浮点）格式"""

    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool): # 初始化方法，接收权重量化参数和是否静态输入方案标志
        self.weight_quant = weight_quant # 保存权重量化参数
        self.strategy = self.weight_quant.strategy # 提取权重缩放策略
        self.is_static_input_scheme = is_static_input_scheme # 保存是否为静态输入量化方案
        self.weight_block_size = self.weight_quant.block_structure # 提取权重分块结构
        if self.weight_block_size is not None: # 如果存在分块结构
            self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear() # 分发并初始化W8A8分块FP8线性计算函数

    @classmethod
    def get_min_capability(cls) -> int: # 获取最低GPU计算能力要求
        """获取运行此量化方案所需的最低GPU计算能力版本"""
        # lovelace and up # Lovelace架构及以上
        return 89 # 返回8.9（Lovelace架构）

    def create_weights( # 创建权重参数方法
        self,
        layer: torch.nn.Module, # 目标神经网络层
        input_size_per_partition: int, # 每个分区的输入大小
        output_partition_sizes: list[int], # 输出分区大小列表
        input_size: int, # 总输入大小
        output_size: int, # 总输出大小
        params_dtype: torch.dtype, # 参数数据类型
        weight_loader: Callable, # 权重加载器
        **kwargs, # 额外关键字参数
    ):
        """创建并注册W8A8 FP8量化所需的权重和缩放参数"""
        output_size_per_partition = sum(output_partition_sizes) # 计算每个分区的总输出大小
        layer.logical_widths = output_partition_sizes # 保存逻辑宽度列表
        layer.weight_block_size = None # 初始化权重分块大小为None
        layer.orig_dtype = params_dtype # 保存原始数据类型

        if self.strategy == QuantizationStrategy.BLOCK: # 如果是分块量化策略
            assert self.weight_block_size is not None # 断言分块大小不为None
            layer.weight_block_size = self.weight_block_size # 保存分块大小到层
            # Validate block quantization shapes # 验证分块量化形状
            validate_fp8_block_shape( # 调用形状验证函数
                layer, # 目标层
                input_size, # 总输入大小
                output_size, # 总输出大小
                input_size_per_partition, # 每个分区的输入大小
                output_partition_sizes, # 输出分区大小列表
                self.weight_block_size, # 权重分块大小
            )

        # WEIGHT # 权重
        weight = ModelWeightParameter( # 创建模型权重参数
            data=torch.empty( # 创建空张量
                output_size_per_partition, # 输出维度
                input_size_per_partition, # 输入维度
                dtype=torch.float8_e4m3fn, # FP8 E4M3数据类型
            ),
            input_dim=1, # 输入维度索引
            output_dim=0, # 输出维度索引
            weight_loader=weight_loader, # 权重加载器
        )
        layer.register_parameter("weight", weight) # 在层中注册权重参数

        # WEIGHT SCALE # 权重缩放因子
        if self.strategy == QuantizationStrategy.CHANNEL: # 如果是逐通道策略
            weight_scale = ChannelQuantScaleParameter( # 创建逐通道量化缩放参数
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32), # 形状为[总输出大小, 1]
                output_dim=0, # 输出维度索引
                weight_loader=weight_loader, # 权重加载器
            )
            weight_scale[:] = torch.finfo(torch.float32).min # 初始化为float32最小值
        elif self.strategy == QuantizationStrategy.TENSOR: # 如果是逐张量策略
            weight_scale = PerTensorScaleParameter( # 创建逐张量缩放参数
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32), # 形状为[分区数量]
                weight_loader=weight_loader, # 权重加载器
            )
            weight_scale[:] = torch.finfo(torch.float32).min # 初始化为float32最小值
        elif self.strategy == QuantizationStrategy.BLOCK: # 如果是分块策略
            assert layer.weight_block_size is not None # 断言分块大小不为None
            block_n, block_k = layer.weight_block_size[0], layer.weight_block_size[1] # 提取N和K方向的分块大小
            output_size_per_partition = sum(output_partition_sizes) # 计算每个分区的总输出大小
            weight_scale = BlockQuantScaleParameter( # 创建分块量化缩放参数
                data=torch.empty( # 创建空张量
                    (output_size_per_partition + block_n - 1) // block_n, # N方向分块数（向上取整）
                    (input_size_per_partition + block_k - 1) // block_k, # K方向分块数（向上取整）
                    dtype=torch.float32, # float32数据类型
                ),
                input_dim=1, # 输入维度索引
                output_dim=0, # 输出维度索引
                weight_loader=weight_loader, # 权重加载器
            )
            weight_scale.format_ue8m0 = False # 设置格式为非UE8M0
            weight_scale[:] = torch.finfo(torch.float32).min # 初始化为float32最小值

        layer.register_parameter("weight_scale", weight_scale) # 在层中注册权重缩放参数
        # INPUT SCALE # 输入缩放因子
        if self.is_static_input_scheme: # 如果是静态输入方案
            input_scale = PerTensorScaleParameter( # 创建逐张量输入缩放参数
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32), # 形状为[分区数量]
                weight_loader=weight_loader, # 权重加载器
            )
            input_scale[:] = torch.finfo(torch.float32).min # 初始化为float32最小值
            layer.register_parameter("input_scale", input_scale) # 在层中注册输入缩放参数

    def process_weights_after_loading(self, layer) -> None: # 权重加载后处理方法
        """权重加载后的后处理：根据量化策略重新量化、归一化并封装参数"""
        if self.strategy == QuantizationStrategy.TENSOR: # 如果是逐张量策略
            max_w_scale, weight = requantize_with_max_scale( # 使用最大缩放值重新量化
                weight=layer.weight, # 权重
                weight_scale=layer.weight_scale, # 权重缩放
                logical_widths=layer.logical_widths, # 逻辑宽度
            )

            if is_fp8_fnuz(): # 如果是FNUZ格式
                input_scale = getattr(layer, "input_scale", None) # 获取输入缩放，默认为None

                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz( # 归一化E4M3FN到E4M3FNUZ格式
                    weight=weight, weight_scale=max_w_scale, input_scale=input_scale # 传入权重、缩放和输入缩放
                )
                if input_scale is not None: # 如果输入缩放不为None
                    layer.input_scale = Parameter(input_scale, requires_grad=False) # 封装为参数
            layer.weight = Parameter(weight.t(), requires_grad=False) # 转置权重并封装为参数
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False) # 封装缩放为参数

        elif self.strategy == QuantizationStrategy.CHANNEL: # 如果是逐通道策略
            weight = layer.weight # 获取权重

            if is_fp8_fnuz(): # 如果是FNUZ格式
                input_scale = getattr(layer, "input_scale", None) # 获取输入缩放，默认为None

                weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz( # 归一化E4M3FN到E4M3FNUZ格式
                    weight=weight, # 权重
                    weight_scale=layer.weight_scale, # 权重缩放
                    input_scale=input_scale, # 输入缩放
                )
                if input_scale is not None: # 如果输入缩放不为None
                    layer.input_scale = Parameter(input_scale, requires_grad=False) # 封装为参数
            else: # 否则（非FNUZ格式）
                weight_scale = layer.weight_scale.data # 直接获取缩放数据

            if _use_aiter: # 如果使用AITER
                # keep the weight as (N, K) # 保持权重形状为(N, K)
                layer.weight = Parameter( # 封装权重为参数
                    shuffle_weight(weight, (16, 16)), requires_grad=False # 使用AITER重排权重，块大小(16,16)
                )
            else: # 否则（不使用AITER）
                layer.weight = Parameter(weight.t(), requires_grad=False) # 转置权重并封装为参数

            # required by torch.compile to be torch.nn.Parameter
            # torch.compile要求必须是torch.nn.Parameter类型
            layer.weight_scale = Parameter(weight_scale, requires_grad=False) # 封装缩放为参数

        elif self.strategy == QuantizationStrategy.BLOCK: # 如果是分块策略
            assert self.is_static_input_scheme is False # 断言静态输入方案为False（分块量化不支持静态输入）
            weight = layer.weight # 获取权重
            weight_scale = layer.weight_scale # 获取权重缩放

            if is_fp8_fnuz(): # 如果是FNUZ格式
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz( # 归一化E4M3FN到E4M3FNUZ格式
                    weight=weight, weight_scale=weight_scale # 传入权重和缩放
                )
            layer.weight = Parameter(weight.data, requires_grad=False) # 封装权重数据为参数
            layer.weight_scale = Parameter(weight_scale.data, requires_grad=False) # 封装缩放数据为参数

        else: # 其他未知策略
            raise ValueError(f"Unknown quantization strategy {self.strategy}") # 抛出值错误，未知量化策略

        # INPUT SCALE # 输入缩放因子
        if self.is_static_input_scheme and hasattr(layer, "input_scale"): # 如果是静态输入方案且层有输入缩放属性
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False) # 取最大值并封装为参数
        else: # 否则
            layer.input_scale = None # 设置输入缩放为None

    def apply_weights( # 应用权重方法，执行量化线性计算
        self,
        layer: torch.nn.Module, # 目标神经网络层
        x: torch.Tensor, # 输入张量
        bias: Optional[torch.Tensor] = None, # 可选偏置张量
    ) -> torch.Tensor: # 返回输出张量
        """应用W8A8 FP8量化权重，根据量化策略选择合适的内核执行线性计算"""
        if self.weight_block_size is not None: # 如果存在分块大小（分块量化）
            return self.w8a8_block_fp8_linear( # 调用分块FP8线性计算函数
                input=x, # 输入张量
                weight=layer.weight, # 权重
                block_size=self.weight_block_size, # 分块大小
                weight_scale=layer.weight_scale, # 权重缩放因子
                input_scale=layer.input_scale, # 输入缩放因子
                bias=bias, # 偏置
            )

        if _use_aiter and self.strategy == QuantizationStrategy.CHANNEL: # 如果使用AITER且为逐通道策略
            return apply_fp8_ptpc_linear( # 调用逐token逐通道FP8线性计算
                input=x, # 输入张量
                weight=layer.weight, # 权重
                weight_scale=layer.weight_scale, # 权重缩放因子
                input_scale=layer.input_scale, # 输入缩放因子
                bias=bias, # 偏置
                use_per_token_if_dynamic=True, # 动态量化时使用逐token方式
                compressed_tensor_quant=True, # 标记为压缩张量量化
            )
        else: # 否则（通用情况）
            return apply_fp8_linear( # 调用通用FP8线性计算
                input=x, # 输入张量
                weight=layer.weight, # 权重
                weight_scale=layer.weight_scale, # 权重缩放因子
                input_scale=layer.input_scale, # 输入缩放因子
                bias=bias, # 偏置
                use_per_token_if_dynamic=True, # 动态量化时使用逐token方式
                compressed_tensor_quant=True, # 标记为压缩张量量化
            )
