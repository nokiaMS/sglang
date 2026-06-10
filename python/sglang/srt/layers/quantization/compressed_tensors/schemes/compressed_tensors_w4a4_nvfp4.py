# 压缩张量NV FP4 W4A4量化线性层方案模块
# 实现了CompressedTensorsW4A4Fp4类，用于Blackwell架构上的
# W4A4 NVFP4量化线性层推理，支持FlashInfer TRT-LLM和CUTLASS两种后端
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging  # 导入日志模块
from collections.abc import Callable  # 导入可调用类型
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch
from torch.nn.parameter import Parameter  # 导入参数类

from sglang.srt.layers.parameter import (  # 导入各类量化参数类
    GroupQuantScaleParameter,  # 分组量化缩放参数
    ModelWeightParameter,  # 模型权重参数
    PerTensorScaleParameter,  # 逐张量缩放参数
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # 导入压缩张量线性方案基类
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend  # 导入FP4 GEMM运行器后端获取函数
from sglang.srt.layers.quantization.modelopt_quant import (  # 导入ModelOpt量化相关函数
    enable_flashinfer_fp4_gemm,  # 是否启用FlashInfer FP4 GEMM
    fp4_gemm,  # FP4 GEMM运算
    fp4_quantize,  # FP4量化函数
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale  # 导入块缩放交错函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = ["CompressedTensorsW4A4Fp4"]  # 模块公开导出的类列表


class CompressedTensorsW4A4Fp4(CompressedTensorsLinearScheme):  # NV FP4 W4A4量化线性层方案类，继承自CompressedTensorsLinearScheme
    def __init__(self):  # 初始化方法
        self.group_size = 16  # 量化分组大小为16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低设备算力要求
        return 100  # 需要sm100(Blackwell)架构

    def create_weights(  # 创建权重参数
        self,
        layer: torch.nn.Module,  # 目标层模块
        output_partition_sizes: list[int],  # 输出分区大小列表
        input_size_per_partition: int,  # 每个分区的输入大小
        params_dtype: torch.dtype,  # 参数数据类型
        weight_loader: Callable,  # 权重加载器
        **kwargs,  # 其他关键字参数
    ):
        output_size_per_partition = sum(output_partition_sizes)  # 计算输出总大小
        layer.logical_widths = output_partition_sizes  # 保存逻辑宽度列表
        layer.input_size_per_partition = input_size_per_partition  # 保存每个分区的输入大小
        layer.output_size_per_partition = output_size_per_partition  # 保存每个分区的输出大小

        # Weight
        weight = ModelWeightParameter(  # 创建模型权重参数（FP4打包权重）
            data=torch.empty(  # 创建空张量
                sum(output_partition_sizes),  # 输出维度
                input_size_per_partition // 2,  # 输入维度除以2（2个FP4打包为1个uint8）
                dtype=torch.uint8,  # 使用uint8类型存储打包的FP4权重
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight_packed", weight)  # 注册打包权重参数

        # Global Weight Scale  # 全局权重缩放因子
        weight_global_scale = PerTensorScaleParameter(  # 创建逐张量缩放参数
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),  # 每个输出分区一个缩放值
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)  # 注册全局权重缩放因子

        # Per Group Weight Scale  # 逐分组权重缩放因子
        weight_scale = GroupQuantScaleParameter(  # 创建分组量化缩放参数
            data=torch.empty(  # 创建空张量
                sum(output_partition_sizes),  # 输出维度
                input_size_per_partition // self.group_size,  # 按分组数划分的维度
                dtype=torch.float8_e4m3fn,  # 使用FP8 E4M3类型存储缩放因子
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )

        layer.register_parameter("weight_scale", weight_scale)  # 注册逐分组权重缩放因子

        input_global_scale = PerTensorScaleParameter(  # 创建输入全局缩放参数
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),  # 每个输出分区一个缩放值
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("input_global_scale", input_global_scale)  # 注册输入全局缩放因子

    def process_weights_after_loading(self, layer) -> None:  # 权重加载后的后处理方法
        global_input_scale = layer.input_global_scale.max().to(torch.float32)  # 取输入全局缩放因子的最大值并转为float32
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)  # 替换为标量参数

        layer.weight_global_scale = Parameter(  # 取权重全局缩放因子的最大值并转为float32
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        if get_fp4_gemm_runner_backend().is_flashinfer_trtllm():  # 若使用FlashInfer TRT-LLM后端
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer TRTLLM FP4 GEMM需要不同的权重布局。
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            # FlashInfer提供nvfp4_quantize来量化+重排布局，
            # 但我们使用自己的量化，所以必须自己调用重排。
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a  # 导入矩阵和缩放因子重排函数

            weight = layer.weight_packed.data  # 获取打包权重数据
            weight_scale = layer.weight_scale.data  # 获取权重缩放因子数据

            epilogue_tile_m = 128  # epilogue分块大小
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)  # 对权重矩阵进行重排
            weight_scale = (  # 对缩放因子矩阵进行重排
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)  # 恢复原始形状
                .view(torch.float8_e4m3fn)  # 视图转为FP8类型
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)  # 替换缩放因子为处理后的版本
            layer.weight_packed = Parameter(weight, requires_grad=False)  # 替换权重为处理后的版本
        else:  # 否则使用CUTLASS后端
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)  # 对缩放因子进行交错处理
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)  # 替换为交错后的缩放因子
            layer.weight_packed = Parameter(  # 保持权重数据不变
                layer.weight_packed.data, requires_grad=False
            )

        layer.alpha = Parameter(  # 计算alpha = 1 / (input_global_scale * weight_global_scale)
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def apply_weights(  # 应用权重，执行前向传播
        self,
        layer: torch.nn.Module,  # 目标层模块
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置参数（可选）
    ) -> torch.Tensor:
        output_dtype = x.dtype  # 保存输出数据类型
        w_n, _ = layer.weight_packed.shape  # 获取权重输出维度
        output_shape = [x.shape[0], w_n]  # 计算输出形状

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        # 将BF16或FP16量化为（FP4和交错块缩放因子）
        x_fp4, x_blockscale = fp4_quantize(x, layer.input_global_scale)  # 对输入进行FP4量化

        assert x_fp4.dtype == torch.uint8  # 断言量化后的输入为uint8
        assert layer.weight_packed.dtype == torch.uint8  # 断言打包权重为uint8
        assert layer.weight_scale.dtype == torch.float8_e4m3fn  # 断言权重缩放因子为FP8
        assert layer.alpha.dtype == torch.float32  # 断言alpha为float32

        w = layer.weight_packed  # 获取权重
        w_blockscale = layer.weight_scale  # 获取权重块缩放因子
        if (  # 若启用FlashInfer FP4 GEMM且非CUTLASS后端
            enable_flashinfer_fp4_gemm
            and not get_fp4_gemm_runner_backend().is_cutlass()
        ):
            w = layer.weight_packed.T  # 转置权重
            w_blockscale = layer.weight_scale.T  # 转置缩放因子

        out = fp4_gemm(  # 执行FP4 GEMM运算
            x_fp4,  # 量化后的输入
            w,  # 权重
            x_blockscale,  # 输入块缩放因子
            w_blockscale,  # 权重块缩放因子
            layer.alpha,  # alpha缩放参数
            output_dtype,  # 输出数据类型
            w_n,  # 输出维度
        )
        if bias is not None:  # 若有偏置
            out = out + bias  # 加上偏置
        return out.view(*output_shape)  # 返回重塑后的输出
