# 压缩张量W8A8 INT8量化方案实现
# 本文件实现了CompressedTensors框架下W8A8（权重8比特INT8、激活8比特INT8）量化方案
# 支持CUDA和NPU平台，包含对称和非对称量化、静态和动态输入缩放

# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# 适配自 vLLM 项目的压缩张量量化实现
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX文件版权文本：版权归属vLLM项目的贡献者

from typing import Callable, Optional # 导入类型提示工具

import torch # 导入PyTorch库
from compressed_tensors.quantization import QuantizationStrategy # 导入量化策略枚举
from torch.nn import Parameter # 导入神经网络参数类

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import ( # 导入NPU W8A8 INT8动态线性方法
    NPUW8A8Int8DynamicLinearMethod,
)
from sglang.srt.layers.parameter import ( # 导入参数类
    ChannelQuantScaleParameter, # 逐通道量化缩放参数
    ModelWeightParameter, # 模型权重参数
    PerTensorScaleParameter, # 逐张量缩放参数
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import ( # 导入压缩张量线性方案基类
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8 # 导入逐token INT8量化函数
from sglang.srt.layers.quantization.utils import requantize_with_max_scale # 导入使用最大缩放重新量化工具
from sglang.srt.utils import is_cuda # 导入CUDA平台检测函数

__all__ = ["CompressedTensorsW8A8Int8", "NPUCompressedTensorsW8A8Int8"] # 模块公开接口列表

_is_cuda = is_cuda() # 检测是否为CUDA平台
if _is_cuda: # 如果是CUDA平台
    from sgl_kernel import int8_scaled_mm # 导入INT8缩放矩阵乘法内核


class CompressedTensorsW8A8Int8(CompressedTensorsLinearScheme): # 压缩张量W8A8 INT8量化方案类（CUDA版本）
    """W8A8 INT8量化方案：权重和激活均使用INT8（8比特整数）格式"""

    def __init__( # 初始化方法
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool # 接收量化策略、是否静态输入方案、是否对称输入标志
    ):
        self.strategy = strategy # 保存量化策略
        self.is_static_input_scheme = is_static_input_scheme # 保存是否为静态输入量化方案
        self.input_symmetric = input_symmetric # 保存是否为对称输入量化

    def create_weights( # 创建权重参数方法
        self,
        layer: torch.nn.Module, # 目标神经网络层
        output_partition_sizes: list[int], # 输出分区大小列表
        input_size_per_partition: int, # 每个分区的输入大小
        params_dtype: torch.dtype, # 参数数据类型
        weight_loader: Callable, # 权重加载器
        **kwargs, # 额外关键字参数
    ):
        """创建并注册W8A8 INT8量化所需的权重、缩放因子和零点参数"""
        output_size_per_partition = sum(output_partition_sizes) # 计算每个分区的总输出大小
        layer.logical_widths = output_partition_sizes # 保存逻辑宽度列表

        # WEIGHT # 权重
        weight = ModelWeightParameter( # 创建模型权重参数
            data=torch.empty( # 创建空张量
                output_size_per_partition, input_size_per_partition, dtype=torch.int8 # 形状为[输出大小, 输入大小]，int8类型
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
        else: # 否则
            assert self.strategy == QuantizationStrategy.TENSOR # 断言为逐张量策略
            weight_scale = PerTensorScaleParameter( # 创建逐张量缩放参数
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32), # 形状为[分区数量]
                weight_loader=weight_loader, # 权重加载器
            )
        layer.register_parameter("weight_scale", weight_scale) # 在层中注册权重缩放参数

        # INPUT SCALE # 输入缩放因子
        if self.is_static_input_scheme: # 如果是静态输入方案
            input_scale = PerTensorScaleParameter( # 创建逐张量输入缩放参数
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader # 形状为[1]
            )
            layer.register_parameter("input_scale", input_scale) # 在层中注册输入缩放参数

            if not self.input_symmetric: # 如果不是对称输入量化
                # Note: compressed-tensors stores the zp using the same dtype
                # as the weights
                # 注意：compressed-tensors使用与权重相同的数据类型存储零点
                # AZP loaded as int8 but used as int32
                # 零点以int8加载但以int32使用
                input_zero_point = PerTensorScaleParameter( # 创建输入零点参数
                    data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader # 形状为[1]，int8类型
                )
                layer.register_parameter("input_zero_point", input_zero_point) # 在层中注册输入零点参数

    @classmethod
    def get_min_capability(cls) -> int: # 获取最低GPU计算能力要求
        """获取运行此量化方案所需的最低GPU计算能力版本"""
        # ampere and up # Ampere架构及以上
        return 80 # 返回8.0（Ampere架构）

    def process_weights_after_loading(self, layer) -> None: # 权重加载后处理方法
        """权重加载后的后处理：重新量化、转置权重、处理输入缩放和零点、计算AZP调整项"""
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per channel
        # 如果是逐张量量化，当融合模块（如QKV）使用逐张量缩放（因此有N个缩放值传递给内核）时，
        # 重新量化以便始终以逐通道方式运行
        if self.strategy == QuantizationStrategy.TENSOR: # 如果是逐张量策略
            max_w_scale, weight = requantize_with_max_scale( # 使用最大缩放值重新量化
                weight=layer.weight, # 权重
                weight_scale=layer.weight_scale, # 权重缩放
                logical_widths=layer.logical_widths, # 逻辑宽度
            )

            layer.weight = Parameter(weight.t(), requires_grad=False) # 转置权重并封装为参数
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False) # 封装最大缩放为参数

        # If channelwise, scales are already lined up, so just transpose.
        # 如果是逐通道量化，缩放值已对齐，只需转置。
        elif self.strategy == QuantizationStrategy.CHANNEL: # 如果是逐通道策略
            weight = layer.weight # 获取权重
            weight_scale = layer.weight_scale.data # 获取缩放数据

            layer.weight = Parameter(weight.t(), requires_grad=False) # 转置权重并封装为参数
            # required by torch.compile to be torch.nn.Parameter
            # torch.compile要求必须是torch.nn.Parameter类型
            layer.weight_scale = Parameter(weight_scale, requires_grad=False) # 封装缩放为参数

        else: # 其他未知策略
            raise ValueError(f"Unknown quantization strategy {self.strategy}") # 抛出值错误，未知量化策略

        # INPUT SCALE # 输入缩放因子
        if self.is_static_input_scheme and hasattr(layer, "input_scale"): # 如果是静态输入方案且层有输入缩放属性
            if self.input_symmetric: # 如果是对称量化
                layer.input_scale = Parameter( # 封装输入缩放最大值为参数
                    layer.input_scale.max(), requires_grad=False # 取最大值，不需要梯度
                )
            else: # 否则（非对称量化）
                input_scale = layer.input_scale # 获取输入缩放
                input_zero_point = layer.input_zero_point # 获取输入零点

                # reconstruct the ranges # 重建量化范围
                int8_traits = torch.iinfo(torch.int8) # 获取int8类型的数值特征
                azps = input_zero_point.to(dtype=torch.int32) # 将零点转换为int32
                range_max = (input_scale * (int8_traits.max - azps)).max() # 计算范围最大值
                range_min = (input_scale * (int8_traits.min - azps)).min() # 计算范围最小值

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min) # 计算新的缩放值

                # AZP loaded as int8 but used as int32
                # 零点以int8加载但以int32使用
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32) # 计算非对称零点值

                layer.input_scale = Parameter(scale, requires_grad=False) # 封装新缩放值为参数
                layer.input_zero_point = Parameter(azp, requires_grad=False) # 封装新零点为参数
        else: # 否则（非静态输入方案）
            layer.input_scale = None # 设置输入缩放为None
            layer.input_zero_point = None # 设置输入零点为None

        # azp_adj is the AZP adjustment term, used to account for weights.
        # azp_adj是AZP（非对称零点）调整项，用于补偿权重的影响。
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # 它不依赖于缩放值或零点，因此在静态和动态量化中相同。
        # For more details, see csrc/quantization/cutlass_w8a8/Epilogues.md
        # 更多详情请参见 csrc/quantization/cutlass_w8a8/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/8d59dbb00044a588cab96bcdc028006ed922eb06/csrc/quantization/cutlass_w8a8/Epilogues.md
        if not self.input_symmetric: # 如果不是对称输入量化
            weight = layer.weight # 获取权重
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32) # 沿输入维度求和计算AZP调整项
            if self.is_static_input_scheme: # 如果是静态输入方案
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                # cutlass_w8a8要求在逐张量情况下将azp折叠到azp_adj中
                azp_adj = layer.input_zero_point * azp_adj # 将零点乘入调整项
            layer.azp_adj = Parameter(azp_adj, requires_grad=False) # 封装AZP调整项为参数
        else: # 否则（对称量化）
            layer.azp_adj = None # AZP调整项为None

    def apply_weights( # 应用权重方法，执行量化线性计算
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] # 接收层、输入和偏置
    ) -> torch.Tensor: # 返回输出张量
        """应用W8A8 INT8量化权重，对输入逐token量化后执行INT8缩放矩阵乘法"""
        # TODO: add cutlass_scaled_mm_azp support # 待办：添加cutlass_scaled_mm_azp支持
        x_q, x_scale = per_token_quant_int8(x) # 对输入执行逐token INT8量化

        return int8_scaled_mm( # 调用INT8缩放矩阵乘法
            x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias # 传入量化输入、权重、缩放值和偏置
        )


class NPUCompressedTensorsW8A8Int8(CompressedTensorsW8A8Int8): # NPU平台上的压缩张量W8A8 INT8量化方案类
    """NPU平台上的W8A8 INT8量化方案，继承自CUDA版本，使用NPU专用内核"""

    def __init__( # 初始化方法
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool # 接收量化策略、是否静态输入方案、是否对称输入标志
    ):
        super().__init__(strategy, is_static_input_scheme, input_symmetric) # 调用父类初始化方法
        # TODO: Currently, NPU kernel for static quant requires quant_bias field,
        # which can't be replicated in compressed-tensors.
        # 待办：目前NPU静态量化内核需要quant_bias字段，
        # 该字段无法在compressed-tensors中复制。
        if self.is_static_input_scheme: # 如果是静态输入方案
            raise NotImplementedError( # 抛出未实现错误
                "Static compressed-tensors scheme is not yet supported on NPU." # NPU尚不支持静态compressed-tensors方案
            )
        self.kernel = NPUW8A8Int8DynamicLinearMethod() # 初始化NPU W8A8 INT8动态线性方法内核

    @classmethod
    def get_min_capability(cls) -> int: # 获取最低计算能力要求
        """获取最低计算能力要求（NPU平台不适用）"""
        return NotImplementedError # 返回未实现错误

    def process_weights_after_loading(self, layer): # 权重加载后处理方法
        """权重加载后的后处理，委托给NPU内核执行"""
        return self.kernel.process_weights_after_loading(layer) # 调用NPU内核的权重后处理方法

    def apply_weights(self, layer, x, bias): # 应用权重方法，执行量化线性计算
        """应用W8A8 INT8量化权重，委托给NPU内核执行"""
        return self.kernel.apply(layer, x, bias) # 调用NPU内核的权重应用方法
