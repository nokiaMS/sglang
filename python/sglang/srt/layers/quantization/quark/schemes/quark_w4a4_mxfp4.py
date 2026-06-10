# SPDX-License-Identifier: Apache-2.0
# Quark W4A4 MXFP4量化方案实现文件，实现了权重4bit激活4bit的MXFP4格式量化线性层

from typing import Any, Callable, Optional  # 导入类型提示工具

import torch  # 导入PyTorch库

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter  # 导入分组量化缩放参数和打包参数类
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme  # 导入Quark线性层量化方案基类
from sglang.srt.utils import is_hip  # 导入HIP平台检测函数

_is_hip = is_hip()  # 检测当前是否为HIP（AMD GPU）平台
if _is_hip:  # 如果是HIP平台
    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import (  # 导入融合GEMM分割拼接操作
        fused_gemm_afp4wfp4_split_cat,
    )
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4  # 导入AFP4WFP4矩阵乘法操作
    from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant  # 导入预量化GEMM操作
    from aiter.ops.triton.quant import dynamic_mxfp4_quant  # 导入动态MXFP4量化函数


__all__ = ["QuarkW4A4MXFP4"]  # 模块公开导出的类列表

OCP_MX_BLOCK_SIZE = 32  # OCP MX标准块大小，用于分组量化


class QuarkW4A4MXFP4(QuarkLinearScheme):  # Quark W4A4 MXFP4量化方案类，继承自QuarkLinearScheme

    def __init__(  # 初始化方法
        self, weight_quant_spec: dict[str, Any], input_quant_spec: dict[str, Any]
    ):
        self.out_dtype = torch.get_default_dtype()  # 获取默认输出数据类型
        self.qscheme = "per_group"  # 量化方案为按组量化
        self.weight_quant_spec = weight_quant_spec  # 保存权重量化规格
        self.input_quant_spec = input_quant_spec  # 保存输入量化规格

    @classmethod  # 类方法装饰器
    def get_min_capability(cls) -> int:  # 获取最低设备计算能力要求
        return 70  # 最低计算能力为7.0（Volta架构及以上）

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后处理（此方案无需额外处理）
        return  # 直接返回

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
        weight = PackedvLLMParameter(  # 创建打包的权重参数
            data=torch.empty(  # 创建空张量
                output_size_per_partition,  # 输出维度大小
                input_size_per_partition // 2,  # 输入维度除以2（因为4bit打包，两个值存在一个uint8中）
                dtype=torch.uint8,  # 数据类型为uint8
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            packed_dim=1,  # 打包维度索引
            packed_factor=2,  # 打包因子为2（两个4bit值打包为一个uint8）
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight", weight)  # 注册权重参数到层

        # WEIGHT SCALE  # 权重缩放因子
        weight_scale = GroupQuantScaleParameter(  # 创建分组量化缩放参数
            data=torch.empty(  # 创建空张量
                output_size_per_partition,  # 输出维度大小
                input_size_per_partition // OCP_MX_BLOCK_SIZE,  # 每组缩放因子数量（输入大小除以块大小）
                dtype=torch.uint8,  # 数据类型为uint8（E8M0格式）
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("weight_scale", weight_scale)  # 注册权重缩放参数到层

    def apply_weights(  # 应用权重执行前向传播的方法
        self,
        layer: torch.nn.Module,  # 目标层模块
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置张量（可选）
    ) -> torch.Tensor:  # 返回输出张量
        # This path does not have support for bias currently  # 当前路径不支持偏置
        assert bias is None, "bias is not supported"  # 断言偏置必须为None，bias不支持

        three_d = False  # 是否为三维输入的标志
        fused_gemm_split_cat = False  # 是否使用融合GEMM分割拼接的标志
        x_s = None  # 输入缩放因子
        y = None  # 输出张量

        if isinstance(x, tuple):  # 如果输入是元组
            assert len(x) in [  # 断言元组长度只能是2、3或5
                2,
                3,
                5,
            ], "For tuple input, only (x, x_s), (x, x_s, y), or (x, y, S1, S2, out_dtype) formats are accepted"  # 对于元组输入，只接受(x,x_s)、(x,x_s,y)或(x,y,S1,S2,out_dtype)格式
            if len(x) == 2:  # 长度为2时
                x, x_s = x  # 解包为输入和缩放因子
            elif len(x) == 3:  # 长度为3时
                x, x_s, y = x  # 解包为输入、缩放因子和输出
            elif len(x) == 5:  # 长度为5时
                x, y, S1, S2, out_dtype = x  # 解包为输入、输出、分割参数和输出类型
                fused_gemm_split_cat = True  # 启用融合GEMM分割拼接模式

        use_fused_quant_gemm = (  # 判断是否使用融合量化GEMM
            not fused_gemm_split_cat  # 不使用分割拼接模式
            and x_s is None  # 输入缩放因子为None
            and y is not None  # 输出张量已提供
            and layer.weight.shape[0] == y.shape[1]  # 权重输出维度与输出张量维度匹配
        )

        if x.dim() == 3:  # 如果输入是三维的
            three_d = True  # 标记为三维输入
            x = x.view(-1, x.shape[-1])  # 将三维输入展平为二维
            output_shape = [*x.shape[:-1], layer.weight.shape[0]]  # 记录输出形状用于恢复

        # use_fused_quant_gemm = true, x_q is a bf16/fp16 num  # 使用融合量化GEMM时，x_q是bf16/fp16数
        # x_s is not None = true, x_q is uint8 num  # x_s不为None时，x_q是uint8数
        if use_fused_quant_gemm or x_s is not None:  # 如果使用融合量化GEMM或输入缩放因子已存在
            x_q = x  # 直接使用输入作为量化输入
        else:
            x_q, x_s = dynamic_mxfp4_quant(x)  # 对输入进行动态MXFP4量化，得到量化值和缩放因子

        if y is None:  # 如果输出张量未提供
            y = torch.empty(  # 创建空的输出张量
                x_q.shape[0],  # 批次大小
                layer.weight.shape[0],  # 输出维度
                device=x_q.device,  # 设备
                dtype=self.out_dtype,  # 输出数据类型
            )

        if use_fused_quant_gemm:  # 如果使用融合量化GEMM
            gemm_afp4wfp4_pre_quant(x_q, layer.weight, layer.weight_scale, y.dtype, y)  # 执行预量化融合GEMM
            y = y.to(x.dtype)  # 将输出转换为输入的数据类型
        elif fused_gemm_split_cat:  # 如果使用融合GEMM分割拼接模式
            k, v = fused_gemm_afp4wfp4_split_cat(  # 执行融合GEMM分割拼接操作，返回k和v
                x=x_q,  # 量化输入
                w=layer.weight,  # 权重
                y=y,  # 输出缓冲区
                x_scale=x_s,  # 输入缩放因子
                w_scale=layer.weight_scale,  # 权重缩放因子
                S1=S1,  # 分割参数1
                S2=S2,  # 分割参数2
                dtype=out_dtype,  # 输出数据类型
            )
        else:
            gemm_afp4wfp4(x_q, layer.weight, x_s, layer.weight_scale, self.out_dtype, y)  # 执行标准AFP4WFP4矩阵乘法

        if fused_gemm_split_cat:  # 如果使用了融合GEMM分割拼接模式
            return k, v  # 返回k和v
        elif three_d:  # 如果原始输入是三维的
            return y.view(*output_shape)  # 将输出恢复为三维形状
        else:
            return y  # 直接返回二维输出
