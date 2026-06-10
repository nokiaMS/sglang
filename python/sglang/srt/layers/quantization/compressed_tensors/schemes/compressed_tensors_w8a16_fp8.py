# 压缩张量W8A16 FP8量化方案实现
# 本文件实现了CompressedTensors框架下W8A16（权重8比特FP8、激活16比特浮点）量化方案
# 支持逐通道和逐张量两种权重缩放策略，使用Marlin内核进行高效FP8矩阵乘法

# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# 适配自 vLLM 项目的压缩张量量化实现
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX文件版权文本：版权归属vLLM项目的贡献者
from typing import Callable, List, Optional # 导入类型提示工具

import torch # 导入PyTorch库
from compressed_tensors.quantization import QuantizationStrategy # 导入量化策略枚举

from sglang.srt.layers.parameter import ( # 导入参数类
    ChannelQuantScaleParameter, # 逐通道量化缩放参数
    ModelWeightParameter, # 模型权重参数
    PerTensorScaleParameter, # 逐张量缩放参数
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import ( # 导入压缩张量线性方案基类
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import ( # 导入FP8 Marlin工具函数
    apply_fp8_marlin_linear, # 应用FP8 Marlin线性计算
    prepare_fp8_layer_for_marlin, # 为Marlin准备FP8层
)
from sglang.srt.layers.quantization.utils import convert_to_channelwise # 导入逐通道转换工具

__all__ = ["CompressedTensorsW8A16Fp8"] # 模块公开接口列表

SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR] # 支持的量化策略列表：逐通道和逐张量


class CompressedTensorsW8A16Fp8(CompressedTensorsLinearScheme): # 压缩张量W8A16 FP8量化方案类
    """W8A16 FP8量化方案：权重使用FP8（8比特浮点），激活使用16比特浮点"""

    def __init__(self, strategy: str, is_static_input_scheme: bool): # 初始化方法，接收量化策略和是否静态输入方案标志
        self.strategy = strategy # 保存量化策略
        self.is_static_input_scheme = is_static_input_scheme # 保存是否为静态输入量化方案

    @classmethod
    def get_min_capability(cls) -> int: # 获取最低GPU计算能力要求
        """获取运行此量化方案所需的最低GPU计算能力版本"""
        # ampere and up # Ampere架构及以上
        return 80 # 返回8.0（Ampere架构）

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # W8A8-FP8内核仅支持逐张量和逐通道两种情况。
    # So if we have a fused module (QKV, MLP) with per tensor scales,
    # 因此如果融合模块（QKV、MLP）使用逐张量缩放，
    # we expand each scale to its shard's channels.
    # 我们将每个缩放值扩展到其对应分片的通道上。
    def process_weights_after_loading(self, layer) -> None: # 权重加载后处理方法
        """权重加载后的后处理：转换缩放策略、转置权重、准备Marlin内核"""
        if self.strategy == QuantizationStrategy.TENSOR: # 如果是逐张量策略
            ws_channelwise = convert_to_channelwise( # 将逐张量缩放转换为逐通道缩放
                layer.weight_scale, layer.logical_widths # 使用权重缩放和逻辑宽度
            )
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False) # 将转换后的缩放设置为参数
        else: # 否则（逐通道策略）
            # required by torch.compile to be torch.nn.Parameter
            # torch.compile要求必须是torch.nn.Parameter类型
            layer.weight_scale = torch.nn.Parameter( # 将权重缩放数据封装为参数
                layer.weight_scale.data, requires_grad=False # 不需要梯度
            )

        # Weights must be transposed for marlin # Marlin内核要求权重必须转置
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False) # 转置权重并封装为参数

        if self.is_static_input_scheme: # 如果是静态输入方案
            # required by torch.compile to be torch.nn.Parameter
            # torch.compile要求必须是torch.nn.Parameter类型
            layer.input_scale = torch.nn.Parameter( # 将输入缩放数据封装为参数
                layer.input_scale.data, requires_grad=False # 不需要梯度
            )
        prepare_fp8_layer_for_marlin(layer, size_k_first=True) # 为Marlin内核准备FP8层，K维度在前

    def create_weights( # 创建权重参数方法
        self,
        layer: torch.nn.Module, # 目标神经网络层
        input_size: int, # 输入大小
        output_partition_sizes: List[int], # 输出分区大小列表
        input_size_per_partition: int, # 每个分区的输入大小
        params_dtype: torch.dtype, # 参数数据类型
        weight_loader: Callable, # 权重加载器
        **kwargs, # 额外关键字参数
    ):
        """创建并注册W8A16 FP8量化所需的权重和缩放参数"""
        output_size_per_partition = sum(output_partition_sizes) # 计算每个分区的总输出大小
        layer.logical_widths = output_partition_sizes # 保存逻辑宽度列表
        layer.input_size_per_partition = input_size_per_partition # 保存每个分区的输入大小
        layer.output_size_per_partition = output_size_per_partition # 保存每个分区的输出大小
        layer.orig_dtype = params_dtype # 保存原始数据类型

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
        elif self.strategy == QuantizationStrategy.TENSOR: # 如果是逐张量策略
            weight_scale = PerTensorScaleParameter( # 创建逐张量缩放参数
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32), # 形状为[分区数量]
                weight_loader=weight_loader, # 权重加载器
            )
        else: # 其他不支持的策略
            raise ValueError( # 抛出值错误
                f"Unsupported weight strategy={self.strategy}, " # 不支持的权重策略
                f"supported strategies are {SUPPORTED_STRATEGIES}" # 支持的策略列表
            )

        weight_scale[:] = torch.finfo(torch.float32).min # 初始化缩放因子为float32最小值
        layer.register_parameter("weight_scale", weight_scale) # 在层中注册权重缩放参数

        # INPUT SCALE (to deal with converted checkpoints) # 输入缩放因子（用于处理转换后的检查点）
        if self.is_static_input_scheme: # 如果是静态输入方案
            input_scale = PerTensorScaleParameter( # 创建逐张量输入缩放参数
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32), # 形状为[分区数量]
                weight_loader=weight_loader, # 权重加载器
            )
            layer.register_parameter("input_scale", input_scale) # 在层中注册输入缩放参数

    def apply_weights( # 应用权重方法，执行量化线性计算
        self,
        layer: torch.nn.Module, # 目标神经网络层
        x: torch.Tensor, # 输入张量
        bias: Optional[torch.Tensor] = None, # 可选偏置张量
    ) -> torch.Tensor: # 返回输出张量
        """应用W8A16 FP8量化权重，使用Marlin内核执行线性计算"""
        return apply_fp8_marlin_linear( # 调用FP8 Marlin线性计算函数
            input=x, # 输入张量
            weight=layer.weight, # 权重
            weight_scale=layer.weight_scale, # 权重缩放因子
            workspace=layer.workspace, # Marlin工作空间
            size_n=layer.output_size_per_partition, # 输出维度大小N
            size_k=layer.input_size_per_partition, # 输入维度大小K
            bias=bias, # 偏置
        )
