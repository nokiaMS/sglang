# 压缩张量WNA16量化方案（用于线性层）
# 本文件实现了WNA16（宽N位权重A16位激活）量化方案，
# 使用Marlin内核进行高效推理，支持4位和8位权重量化。
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0  # Apache-2.0许可证声明

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # vLLM项目版权声明
import logging  # 导入日志模块
from typing import Callable, Optional  # 导入类型提示

import torch  # 导入PyTorch深度学习框架
from compressed_tensors.quantization import ActivationOrdering  # 导入激活排序枚举

# yapf conflicts with isort for this block  # yapf与isort在此块冲突
# yapf: disable  # 禁用yapf格式化
from sglang.srt.layers.parameter import (  # 导入参数类
    BasevLLMParameter,  # 基础vLLM参数类
    ChannelQuantScaleParameter,  # 通道量化缩放参数类
    GroupQuantScaleParameter,  # 分组量化缩放参数类
    PackedColumnParameter,  # 打包列参数类
    PackedvLLMParameter,  # 打包vLLM参数类
    RowvLLMParameter,  # 行vLLM参数类
    permute_param_layout_,  # 参数布局置换函数
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # 导入压缩张量线性方案基类
    CompressedTensorsLinearScheme,  # 压缩张量线性量化方案基类
)
from sglang.srt.layers.quantization.marlin_utils import (  # 导入Marlin工具函数
    MarlinLinearLayerConfig,  # Marlin线性层配置类
    apply_gptq_marlin_linear,  # 应用GPTQ Marlin线性运算
    check_marlin_supports_shape,  # 检查Marlin是否支持给定形状
    marlin_is_k_full,  # 判断Marlin的K是否为完整
    marlin_make_empty_g_idx,  # 创建空的分组索引
    marlin_make_workspace,  # 创建Marlin工作空间
    marlin_permute_scales,  # Marlin缩放因子置换
    marlin_repeat_scales_on_all_ranks,  # Marlin在所有rank上重复缩放因子
    marlin_sort_g_idx,  # Marlin分组索引排序
    marlin_zero_points,  # Marlin零点处理
)
from sglang.srt.layers.quantization.utils import (  # 导入量化工具函数
    get_scalar_types,  # 获取标量类型
    replace_parameter,  # 替换参数
    unpack_cols,  # 解包列
)
from sglang.srt.utils import is_cuda  # 导入CUDA检测函数

_is_cuda = is_cuda()  # 检测当前是否为CUDA环境

if _is_cuda:  # 如果是CUDA环境
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack  # 导入GPTQ Marlin重打包内核


ScalarType, scalar_types = get_scalar_types()  # 获取标量类型和标量类型字典

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = ["CompressedTensorsWNA16"]  # 模块公开导出列表
WNA16_SUPPORTED_TYPES_MAP = {  # WNA16支持的量化类型映射
    4: scalar_types.uint4b8,  # 4位：无符号4位（偏移128）
    8: scalar_types.uint8b128  # 8位：无符号8位（偏移128）
}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}  # WNA16支持零点的量化类型映射
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())  # WNA16支持的位宽列表


class CompressedTensorsWNA16(CompressedTensorsLinearScheme):
    """压缩张量WNA16量化方案，支持4/8位权重量化和16位激活，使用Marlin内核加速推理。"""
    _kernel_backends_being_used: set[str] = set()  # 正在使用的内核后端集合

    def __init__(self,  # 初始化方法
                 strategy: str,  # 量化策略（如"group"或"channel"）
                 num_bits: int,  # 量化位宽
                 group_size: Optional[int] = None,  # 分组大小，None表示通道级量化
                 symmetric: Optional[bool] = True,  # 是否对称量化
                 actorder: Optional[ActivationOrdering] = None):  # 激活排序方式
        """初始化WNA16量化方案，配置量化参数并验证合法性。"""

        self.pack_factor = 32 // num_bits  # 计算打包因子（32位中可以打包多少个量化值）
        self.strategy = strategy  # 保存量化策略
        self.symmetric = symmetric  # 保存是否对称量化
        self.group_size = -1 if group_size is None else group_size  # None转换为-1表示通道级量化
        self.has_g_idx = actorder == ActivationOrdering.GROUP  # 是否有分组索引（激活重排序）

        if self.group_size == -1 and self.strategy != "channel":  # 如果没有分组大小且策略不是通道级
            raise ValueError("Marlin kernels require group quantization or "  # Marlin内核需要分组量化
                             "channelwise quantization, but found no group "  # 或通道级量化，但未找到分组
                             "size and strategy is not channelwise.")  # 大小且策略不是通道级

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:  # 如果位宽不在支持列表中
            raise ValueError(  # 抛出值错误
                f"Unsupported num_bits = {num_bits}. "  # 不支持的位宽
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")  # 支持的位宽

        self.quant_type = (WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]  # 如果非对称量化，使用零点类型
                           if not self.symmetric else  # 否则
                           WNA16_SUPPORTED_TYPES_MAP[num_bits])  # 使用标准类型

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低GPU算力要求
        """获取运行此量化方案所需的最低GPU计算能力版本。"""
        # ampere and up  # Ampere架构及以上
        return 80  # 返回SM 8.0（Ampere架构）

    def create_weights(self, layer: torch.nn.Module, output_size: int,  # 创建权重参数方法
                       input_size: int, output_partition_sizes: list[int],  # 输入大小和输出分区大小
                       input_size_per_partition: int,  # 每个分区的输入大小
                       params_dtype: torch.dtype, weight_loader: Callable,  # 参数数据类型和权重加载器
                       **kwargs):  # 额外关键字参数
        """为WNA16量化线性层创建并注册所有权重参数（打包权重、缩放因子、零点、分组索引等）。"""

        output_size_per_partition = sum(output_partition_sizes)  # 计算分区输出大小之和

        self.kernel_config = MarlinLinearLayerConfig(  # 创建Marlin线性层配置
            full_weight_shape=(input_size, output_size),  # 完整权重形状
            partition_weight_shape=(  # 分区权重形状
                input_size_per_partition,  # 每个分区的输入大小
                output_size_per_partition,  # 每个分区的输出大小
            ),
            weight_type=self.quant_type,  # 权重量化类型
            act_type=params_dtype,  # 激活数据类型
            group_size=self.group_size,  # 分组大小
            zero_points=not self.symmetric,  # 是否有零点
            has_g_idx=self.has_g_idx  # 是否有分组索引
        )

        # If group_size is -1, we are in channelwise case.  # 如果group_size为-1，则处于通道级量化情况
        group_size = self.group_size if self.group_size != -1 else input_size  # 通道级量化时使用input_size作为分组大小
        row_parallel = (input_size != input_size_per_partition)  # 是否为行并行
        partition_scales = not marlin_repeat_scales_on_all_ranks(  # 是否分区缩放因子
            self.has_g_idx, self.group_size, row_parallel)  # 传入分组索引、分组大小和行并行标志

        scales_and_zp_size = input_size // group_size  # 计算缩放因子和零点的大小

        if partition_scales:  # 如果需要分区缩放
            assert input_size_per_partition % group_size == 0  # 断言分区输入大小可被分组大小整除
            scales_and_zp_size = input_size_per_partition // group_size  # 重新计算缩放因子和零点大小

        weight = PackedvLLMParameter(input_dim=1,  # 创建打包权重参数，输入维度为1
                                     output_dim=0,  # 输出维度为0
                                     weight_loader=weight_loader,  # 权重加载器
                                     packed_factor=self.pack_factor,  # 打包因子
                                     packed_dim=1,  # 打包维度
                                     data=torch.empty(  # 创建空数据张量
                                         output_size_per_partition,  # 分区输出大小
                                         input_size_per_partition //  # 分区输入大小除以
                                         self.pack_factor,  # 打包因子
                                         dtype=torch.int32,  # 使用int32类型
                                     ))

        weight_scale_args = {  # 权重缩放因子参数字典
            "weight_loader":  # 权重加载器
            weight_loader,  # 权重加载器函数
            "data":  # 数据
            torch.empty(  # 创建空张量
                output_size_per_partition,  # 分区输出大小
                scales_and_zp_size,  # 缩放因子和零点大小
                dtype=params_dtype,  # 参数数据类型
            )
        }

        zeros_args = {  # 零点参数字典
            "weight_loader":  # 权重加载器
            weight_loader,  # 权重加载器函数
            "data":  # 数据
            torch.zeros(  # 创建全零张量
                output_size_per_partition // self.pack_factor,  # 分区输出大小除以打包因子
                scales_and_zp_size,  # 缩放因子和零点大小
                dtype=torch.int32,  # 使用int32类型
            )
        }

        if not partition_scales:  # 如果不需要分区缩放（缩放因子在所有rank上重复）
            weight_scale = ChannelQuantScaleParameter(output_dim=0,  # 创建通道量化缩放参数，输出维度为0
                                                       **weight_scale_args)  # 传入缩放因子参数

            if not self.symmetric:  # 如果非对称量化
                qzeros = PackedColumnParameter(output_dim=0,  # 创建打包列参数用于零点
                                                packed_dim=0,  # 打包维度为0
                                                packed_factor=self.pack_factor,  # 打包因子
                                                **zeros_args)  # 传入零点参数
        else:  # 如果需要分区缩放
            weight_scale = GroupQuantScaleParameter(output_dim=0,  # 创建分组量化缩放参数
                                                     input_dim=1,  # 输入维度为1
                                                     **weight_scale_args)  # 传入缩放因子参数
            if not self.symmetric:  # 如果非对称量化
                qzeros = PackedvLLMParameter(input_dim=1,  # 创建打包参数用于零点，输入维度为1
                                              output_dim=0,  # 输出维度为0
                                              packed_dim=0,  # 打包维度为0
                                              packed_factor=self.pack_factor,  # 打包因子
                                              **zeros_args)  # 传入零点参数

        # A 2D array defining the original shape of the weights  # 定义权重原始形状的2D数组
        # before packing  # 打包之前的形状
        weight_shape = BasevLLMParameter(data=torch.empty(2,  # 创建基础参数存储原始权重形状（2个元素）
                                                           dtype=torch.int64),  # 使用int64类型
                                          weight_loader=weight_loader)  # 权重加载器

        layer.register_parameter("weight_packed", weight)  # 注册打包权重参数
        layer.register_parameter("weight_scale", weight_scale)  # 注册权重缩放因子参数
        layer.register_parameter("weight_shape", weight_shape)  # 注册权重形状参数

        if not self.symmetric:  # 如果非对称量化
            layer.register_parameter("weight_zero_point", qzeros)  # 注册零点参数

        # group index (for activation reordering)  # 分组索引（用于激活重排序）
        if self.has_g_idx:  # 如果有分组索引
            weight_g_idx = RowvLLMParameter(data=torch.empty(  # 创建行参数用于分组索引
                input_size_per_partition,  # 每个分区的输入大小
                dtype=torch.int32,  # 使用int32类型
            ),
                                             input_dim=0,  # 输入维度为0
                                             weight_loader=weight_loader)  # 权重加载器
            layer.register_parameter("weight_g_idx", weight_g_idx)  # 注册分组索引参数

    # Checkpoints are serialized in compressed-tensors format, which is  # 检查点以压缩张量格式序列化，
    # different from the format the kernel may want. Handle repacking here.  # 与内核所需格式不同。在此处理重打包。
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # 权重加载后的后处理方法
        """权重加载后的后处理，将压缩张量格式重打包为Marlin内核所需格式。"""
        # Default names since marlin requires empty parameters for these,  # 默认名称，因为Marlin需要这些空参数，
        # TODO: remove this requirement from marlin (allow optional tensors)  # TODO: 从Marlin中移除此要求（允许可选张量）
        self.w_q_name = "weight_packed"  # 打包权重参数名称
        self.w_s_name = "weight_scale"  # 权重缩放因子参数名称
        self.w_zp_name = "weight_zero_point"  # 零点参数名称
        self.w_gidx_name = "weight_g_idx"  # 分组索引参数名称

        device = getattr(layer, self.w_q_name).device  # 获取权重所在设备
        c = self.kernel_config  # 获取内核配置

        check_marlin_supports_shape(  # 检查Marlin是否支持该形状
            c.partition_weight_shape[1],  # out_features  # 输出特征数
            c.partition_weight_shape[0],  # in_features  # 输入特征数
            c.full_weight_shape[0],  # in_features  # 完整输入特征数
            c.group_size,  # 分组大小
        )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]  # 是否为行并行
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)  # 判断K维度是否为完整

        # Allocate marlin workspace.  # 分配Marlin工作空间
        self.workspace = marlin_make_workspace(device)  # 创建Marlin工作空间

        def _transform_param(  # 定义参数转换内部函数
            layer: torch.nn.Module, name: Optional[str], fn: Callable  # 目标层、参数名、转换函数
        ) -> None:  # 无返回值
            """内部函数：对层中的指定参数应用转换函数。"""
            if name is not None and getattr(layer, name, None) is not None:  # 如果参数名不为空且参数存在

                old_param = getattr(layer, name)  # 获取旧参数
                new_param = fn(old_param)  # 应用转换函数得到新参数
                # replace the parameter with torch.nn.Parameter for TorchDynamo  # 用torch.nn.Parameter替换参数以兼容TorchDynamo
                # compatibility  # 兼容性
                replace_parameter(  # 替换参数
                    layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)  # 创建不需要梯度的Parameter
                )

        def transform_w_q(x):  # 定义权重转换函数
            """将打包权重从压缩张量格式转换为Marlin格式。"""
            assert isinstance(x, BasevLLMParameter)  # 断言参数为BasevLLMParameter类型
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)  # 置换参数布局
            x.data = gptq_marlin_repack(  # 使用GPTQ Marlin重打包
                x.data.contiguous(),  # 确保数据连续
                perm=layer.g_idx_sort_indices,  # 排序索引排列
                size_k=c.partition_weight_shape[0],  # K维度大小
                size_n=c.partition_weight_shape[1],  # N维度大小
                num_bits=c.weight_type.size_bits,  # 量化位宽
            )
            return x  # 返回转换后的参数

        def transform_w_s(x):  # 定义缩放因子转换函数
            """将缩放因子从压缩张量格式转换为Marlin格式。"""
            assert isinstance(x, BasevLLMParameter)  # 断言参数为BasevLLMParameter类型
            permute_param_layout_(x, input_dim=0, output_dim=1)  # 置换参数布局
            x.data = marlin_permute_scales(  # 使用Marlin缩放因子置换
                x.data.contiguous(),  # 确保数据连续
                size_k=c.partition_weight_shape[0],  # K维度大小
                size_n=c.partition_weight_shape[1],  # N维度大小
                group_size=c.group_size,  # 分组大小
            )
            return x  # 返回转换后的参数

        if c.has_g_idx:  # 如果有分组索引
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(  # 对分组索引排序
                getattr(layer, self.w_gidx_name)  # 获取分组索引
            )
            _transform_param(layer, self.w_gidx_name, lambda _: g_idx)  # 转换分组索引参数
            layer.g_idx_sort_indices = g_idx_sort_indices  # 保存排序索引
        else:  # 如果没有分组索引
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))  # 设置空的分组索引
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)  # 设置空的排序索引

        if c.zero_points:  # 如果有零点
            grouped_k = (  # 计算分组后的K维度
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1  # 如果有分组大小则除以，否则为1
            )
            _transform_param(  # 转换零点参数
                layer,  # 目标层
                self.w_zp_name,  # 零点参数名称
                lambda x: marlin_zero_points(  # 使用Marlin零点处理
                    unpack_cols(  # 先解包列
                        x.t(),  # 转置
                        c.weight_type.size_bits,  # 量化位宽
                        grouped_k,  # 分组后的K维度
                        c.partition_weight_shape[1],  # 分区N维度大小
                    ),
                    size_k=grouped_k,  # 分组后的K维度大小
                    size_n=c.partition_weight_shape[1],  # 分区N维度大小
                    num_bits=c.weight_type.size_bits,  # 量化位宽
                ),
            )
        else:  # 如果没有零点
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))  # 设置空的零点
        _transform_param(layer, self.w_q_name, transform_w_q)  # 转换权重参数
        _transform_param(layer, self.w_s_name, transform_w_s)  # 转换缩放因子参数

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,  # 应用权重进行前向计算方法
                      bias: Optional[torch.Tensor]) -> torch.Tensor:  # 输入张量和偏置
        """应用WNA16量化权重和Marlin内核执行线性层前向计算。"""
        c = self.kernel_config  # 获取内核配置

        def _get_weight_params(  # 定义获取权重参数的内部函数
            layer: torch.nn.Module,  # 目标层
        ) -> tuple[  # 返回元组
            torch.Tensor,  # w_q  # 量化权重
            torch.Tensor,  # w_s  # 权重缩放因子
            Optional[torch.Tensor],  # w_zp,  # 零点
            Optional[torch.Tensor],  # w_gidx  # 分组索引
        ]:
            """内部函数：从层中获取所有权重相关参数。"""
            return (  # 返回参数元组
                getattr(layer, self.w_q_name),  # 获取量化权重
                getattr(layer, self.w_s_name),  # 获取权重缩放因子
                getattr(layer, self.w_zp_name or "", None),  # 获取零点（可能为None）
                getattr(layer, self.w_gidx_name or "", None),  # 获取分组索引（可能为None）
            )

        w_q, w_s, w_zp, w_gidx = _get_weight_params(layer)  # 获取所有权重参数

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not  # `process_weights_after_loading`会确保w_zp和w_gidx对Marlin不是
        #  None for marlin  # None
        return apply_gptq_marlin_linear(  # 调用GPTQ Marlin线性运算
            input=x,  # 输入张量
            weight=w_q,  # 量化权重
            weight_scale=w_s,  # 权重缩放因子
            weight_zp=w_zp,  # type: ignore  # 零点（忽略类型检查）
            g_idx=w_gidx,  # type: ignore  # 分组索引（忽略类型检查）
            g_idx_sort_indices=layer.g_idx_sort_indices,  # 分组索引排序结果
            workspace=self.workspace,  # Marlin工作空间
            wtype=c.weight_type,  # 权重量化类型
            input_size_per_partition=c.partition_weight_shape[0],  # 每个分区的输入大小
            output_size_per_partition=c.partition_weight_shape[1],  # 每个分区的输出大小
            is_k_full=self.is_k_full,  # K维度是否完整
            bias=bias,  # 偏置
        )
