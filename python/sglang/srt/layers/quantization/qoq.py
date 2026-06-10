# QoQ (Quantization for Quantization) 量化配置与线性层方法实现，支持 W4A8 逐通道/逐分组量化推理
from __future__ import annotations  # 启用延迟类型注解求值

from typing import Any, Dict, List, Optional  # 导入类型提示

import torch  # 导入 PyTorch
from torch.nn.parameter import Parameter  # 导入参数类

from sglang.srt.layers.parameter import (  # 导入量化参数类
    ChannelQuantScaleParameter,  # 通道量化缩放参数
    GroupQuantScaleParameter,  # 分组量化缩放参数
    ModelWeightParameter,  # 模型权重参数
)
from sglang.srt.layers.quantization.base_config import (  # 导入量化基础配置类
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8  # 导入逐 token INT8 量化函数
from sglang.srt.utils import is_cuda  # 导入 CUDA 平台判断工具

_is_cuda = is_cuda()  # 判断当前是否为 CUDA 平台
if _is_cuda:  # 如果是 CUDA 平台
    from sgl_kernel import qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm  # 导入 QServe W4A8 GEMM 内核


QoQ_SUPPORTED_WEIGHT_BITS = [4]  # QoQ 支持的权重量化位数
QoQ_SUPPORTED_GROUP_SIZES = [-1, 128]  # QoQ 支持的分组大小，-1 表示逐通道


class QoQConfig(QuantizationConfig):  # QoQ 量化配置类
    """Config class for QoQ Quantization.  # QoQ 量化配置类

    - Weight: static, per-channel/group, asymmetric  # 权重：静态，逐通道/分组，非对称
    - Activation: dynamic, per-token, symmetric  # 激活：动态，逐 token，对称

    Reference: https://arxiv.org/abs/2405.04532  # 参考论文
    https://github.com/mit-han-lab/omniserve  # 参考代码仓库
    """

    def __init__(self, weight_bits: int, group_size: int) -> None:  # 初始化方法
        self.weight_bits = weight_bits  # 保存权重量化位数
        self.group_size = group_size  # 保存分组大小

        # Verify  # 验证参数合法性
        if self.weight_bits not in QoQ_SUPPORTED_WEIGHT_BITS:  # 如果权重位数不支持
            raise ValueError(  # 抛出值错误
                f"QoQ does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {QoQ_SUPPORTED_WEIGHT_BITS} "
                "are supported."
            )  # QoQ 不支持该权重量化位数
        if self.group_size not in QoQ_SUPPORTED_GROUP_SIZES:  # 如果分组大小不支持
            raise ValueError(  # 抛出值错误
                f"QoQ does not support group_size = {self.group_size}. "
                f"Only group_sizes = {QoQ_SUPPORTED_GROUP_SIZES} "
                "are supported."
            )  # QoQ 不支持该分组大小

        # 4 bits packed into 8 bit datatype.  # 4 位打包到 8 位数据类型中
        self.pack_factor = 8 // self.weight_bits  # 计算打包因子，4 位时为 2

    def __repr__(self) -> str:  # 返回配置的字符串表示
        return "QoQConfig(weight_bits={}, group_size={})".format(  # 格式化输出
            self.weight_bits, self.group_size
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.float16]  # 仅支持 float16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低 GPU 计算能力要求
        return 80  # 最低计算能力为 80（Ampere 及以上）

    @classmethod
    def get_name(cls) -> str:  # 获取量化方法名称
        return "qoq"  # 返回 qoq

    @classmethod
    def get_config_filenames(cls) -> List[str]:  # 获取配置文件名列表
        """List of filenames to search for in the model directory."""  # 在模型目录中搜索的文件名列表
        return [  # 返回可能的配置文件名
            "quant_config.json",  # 量化配置文件
            "quantize_config.json",  # 量化配置文件（备选名称）
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QoQConfig:  # 从配置字典创建实例
        weight_bits = cls.get_from_keys(config, ["wbits"])  # 获取权重量化位数
        group_size = cls.get_from_keys(config, ["group_size"])  # 获取分组大小
        return cls(weight_bits, group_size)  # 返回配置实例

    def get_quant_method(  # 获取量化方法
        self,
        layer: torch.nn.Module,  # 目标层
        prefix: str,  # 层名前缀
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase  # 导入线性层基类

        if isinstance(layer, LinearBase):  # 如果是线性层
            return QoQLinearMethod(self)  # 返回 QoQ 线性方法
        return None  # 非线性层返回 None

    def get_scaled_act_names(self) -> List[str]:  # 获取需要缩放的激活名称列表
        return []  # 返回空列表


class QoQLinearMethod(LinearMethodBase):  # QoQ 线性层方法
    """Linear method for QoQ.  # QoQ 线性方法

    Args:  # 参数
        quant_config: The QoQ quantization config.  # QoQ 量化配置
    """

    def __init__(self, quant_config: QoQConfig):  # 初始化方法
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建权重参数
        self,
        layer: torch.nn.Module,  # 目标层
        input_size_per_partition: int,  # 每个分区的输入大小
        output_partition_sizes: List[int],  # 输出分区大小列表
        input_size: int,  # 输入总大小
        output_size: int,  # 输出总大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")  # 获取权重加载器

        # Validate output_size_per_partition  # 验证输出分区大小
        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出总大小
        if output_size_per_partition % 32 != 0:  # 如果输出大小不是 32 的倍数
            raise ValueError(  # 抛出值错误
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by 32."
            )  # 输出分区大小不能被 32 整除

        # Validate input_size_per_partition  # 验证输入分区大小
        if input_size_per_partition % self.quant_config.pack_factor != 0:  # 如果输入大小不能被打包因子整除
            raise ValueError(  # 抛出值错误
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"pack_factor = {self.quant_config.pack_factor}."
            )  # 输入分区大小不能被打包因子整除
        if (  # 如果使用分组量化
            self.quant_config.group_size != -1
            and input_size_per_partition % self.quant_config.group_size != 0  # 且输入大小不能被分组大小整除
        ):
            raise ValueError(  # 抛出值错误
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"group_size = {self.quant_config.group_size}."
            )  # 输入分区大小不能被分组大小整除

        qweight = ModelWeightParameter(  # 创建量化权重参数
            data=torch.empty(  # 分配空张量
                output_size_per_partition,  # 输出维度
                input_size_per_partition // self.quant_config.pack_factor,  # 输入维度除以打包因子
                dtype=torch.int8,  # 数据类型为 int8
            ),
            input_dim=1,  # 输入维度索引
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("qweight", qweight)  # 注册量化权重参数

        s1_scales = ChannelQuantScaleParameter(  # 创建通道级缩放参数
            data=torch.empty(output_size_per_partition, dtype=torch.float16),  # 每个输出通道一个缩放值
            output_dim=0,  # 输出维度索引
            weight_loader=weight_loader,  # 权重加载器
        )
        layer.register_parameter("s1_scales", s1_scales)  # 注册通道级缩放参数

        if self.quant_config.group_size == -1:  # 如果是逐通道量化模式
            s1_szeros = ChannelQuantScaleParameter(  # 创建通道级零点参数
                data=torch.empty(output_size_per_partition, dtype=torch.float16),  # 每个输出通道一个零点值
                output_dim=0,  # 输出维度索引
                weight_loader=weight_loader,  # 权重加载器
            )
            layer.register_parameter("s1_szeros", s1_szeros)  # 注册通道级零点参数
        else:  # 分组量化模式
            s2_scales = GroupQuantScaleParameter(  # 创建分组级缩放参数
                data=torch.empty(  # 分配空张量
                    (
                        input_size_per_partition // self.quant_config.group_size,  # 分组数量
                        output_size_per_partition,  # 输出维度
                    ),
                    dtype=torch.int8,  # 数据类型为 int8
                ),
                input_dim=0,  # 输入维度索引
                output_dim=1,  # 输出维度索引
                weight_loader=weight_loader,  # 权重加载器
            )
            layer.register_parameter("s2_scales", s2_scales)  # 注册分组级缩放参数

            s2_zeros = GroupQuantScaleParameter(  # 创建分组级零点参数
                data=torch.empty(  # 分配空张量
                    (
                        input_size_per_partition // self.quant_config.group_size,  # 分组数量
                        output_size_per_partition,  # 输出维度
                    ),
                    dtype=torch.int8,  # 数据类型为 int8
                ),
                input_dim=0,  # 输入维度索引
                output_dim=1,  # 输出维度索引
                weight_loader=weight_loader,  # 权重加载器
            )
            layer.register_parameter("s2_zeros", s2_zeros)  # 注册分组级零点参数

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 加载权重后处理
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)  # 将量化权重设为不可训练参数
        layer.s1_scales = Parameter(layer.s1_scales.data, requires_grad=False)  # 将通道缩放设为不可训练参数
        if self.quant_config.group_size == -1:  # 如果是逐通道量化模式
            layer.s1_szeros = Parameter(layer.s1_szeros.data, requires_grad=False)  # 将通道零点设为不可训练参数
        else:  # 分组量化模式
            layer.s2_scales = Parameter(layer.s2_scales.data, requires_grad=False)  # 将分组缩放设为不可训练参数
            layer.s2_zeros = Parameter(layer.s2_zeros.data, requires_grad=False)  # 将分组零点设为不可训练参数

    def apply(  # 应用线性层计算
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: Optional[torch.Tensor] = None,  # 偏置项
    ):
        assert x.dtype == torch.float16, "QoQ only supports float16 input now"  # 断言输入必须为 float16，QoQ 当前仅支持 float16 输入
        if self.quant_config.group_size == -1:  # 逐通道量化模式
            x_q, x_scale, x_sum = per_token_quant_int8(  # 逐 token 量化为 INT8，同时计算求和
                x, scale_dtype=x.dtype, cal_sum=True  # 指定缩放类型和是否计算求和
            )
            out = qserve_w4a8_per_chn_gemm(  # 调用 QServe 逐通道 W4A8 GEMM 内核
                x_q, layer.qweight, layer.s1_scales, x_scale, layer.s1_szeros, x_sum  # 传入量化输入、权重、缩放因子等
            )
        else:  # 分组量化模式
            x_q, x_scale = per_token_quant_int8(x, scale_dtype=x.dtype)  # 逐 token 量化为 INT8
            out = qserve_w4a8_per_group_gemm(  # 调用 QServe 逐分组 W4A8 GEMM 内核
                x_q,  # 量化输入
                layer.qweight,  # 量化权重
                layer.s2_zeros,  # 分组零点
                layer.s2_scales,  # 分组缩放
                layer.s1_scales,  # 通道缩放
                x_scale,  # 输入缩放
            )
        if bias is not None:  # 如果有偏置项
            out = out + bias  # 加上偏置
        return out  # 返回输出
