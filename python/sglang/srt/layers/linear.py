# 线性层模块，实现模型并行线性层，包括列并行、行并行、QKV并行和合并列并行等变体
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/linear.py
改编自vLLM项目的线性层实现"""

from __future__ import annotations  # 启用延迟注解评估

import itertools  # 导入迭代工具
import logging  # 导入日志模块
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch核心库
from torch import nn  # 导入神经网络模块
from torch.nn.parameter import Parameter, UninitializedParameter  # 导入参数类

from sglang.kernel_api_logging import wrap_method_with_debug_kernel_once  # 导入调试内核包装器
from sglang.srt.distributed import (  # 导入分布式通信函数
    divide,  # 整除函数
    get_tensor_model_parallel_rank,  # 获取TP rank
    get_tensor_model_parallel_world_size,  # 获取TP世界大小
    get_tp_group,  # 获取TP组
    split_tensor_along_last_dim,  # 沿最后维度分割张量
    tensor_model_parallel_all_gather,  # TP全收集
    tensor_model_parallel_all_reduce,  # TP全归约
    tensor_model_parallel_quant_all_reduce,  # TP量化全归约
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入对称内存
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (  # 导入DP注意力工具
    get_attention_tp_group,  # 获取注意力TP组
    is_allocation_symmetric,  # 判断分配是否对称
)
from sglang.srt.layers.parameter import (  # 导入参数类
    BasevLLMParameter,  # 基础vLLM参数
    BlockQuantScaleParameter,  # 块量化缩放参数
    PackedColumnParameter,  # 打包列参数
    PackedvLLMParameter,  # 打包vLLM参数
    PerTensorScaleParameter,  # 逐张量缩放参数
    RowvLLMParameter,  # 行vLLM参数
    _ColumnvLLMParameter,  # 列vLLM参数
)
from sglang.srt.layers.utils import pad_or_narrow_weight  # 导入权重填充或收窄工具
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import get_bool_env_var, is_cpu, is_hip, is_npu, set_weight_attrs  # 导入工具函数

if TYPE_CHECKING:  # 类型检查时
    from sglang.srt.layers.quantization.base_config import (  # 导入量化配置
        QuantizationConfig,  # 量化配置类
        QuantizeMethodBase,  # 量化方法基类
    )

_is_hip = is_hip()  # 检测是否为HIP(AMD GPU)平台
_disable_hip_linear_quant = _is_hip and get_bool_env_var(  # HIP平台禁用线性量化的标志
    "SGLANG_ROCM_DISABLE_LINEARQUANT"
)

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

WEIGHT_LOADER_V2_SUPPORTED = [  # 支持V2权重加载器的量化方法列表
    "CompressedTensorsLinearMethod",  # 压缩张量线性方法
    "AWQLinearMethod",  # AWQ线性方法
    "GPTQMarlinLinearMethod",  # GPTQ Marlin线性方法
    "Fp8LinearMethod",  # FP8线性方法
    "BlockInt8LinearMethod",  # 块INT8线性方法
    "MarlinLinearMethod",  # Marlin线性方法
    "QQQLinearMethod",  # QQQ线性方法
    "GPTQMarlin24LinearMethod",  # GPTQ Marlin 24线性方法
    "TPUInt8LinearMethod",  # TP UINT8线性方法
    "GPTQLinearMethod",  # GPTQ线性方法
    "FBGEMMFp8LinearMethod",  # FBGEMM FP8线性方法
    "GPTQLinearAscendMethod",  # GPTQ线性Ascend方法
    "GPTQLinearIntelAMXMethod",  # GPTQ线性Intel AMX方法
    "GPTQMoEAscendMethod",  # GPTQ MoE Ascend方法
    "GPTQMoEIntelAMXMethod",  # GPTQ MoE Intel AMX方法
    "ModelOptFp8LinearMethod",  # ModelOpt FP8线性方法
    "ModelOptFp4LinearMethod",  # ModelOpt FP4线性方法
    "IPEXAWQLinearMethod",  # IPEX AWQ线性方法
    "PetitNvFp4LinearMethod",  # Petit NV FP4线性方法
    "QuarkInt4Fp8LinearMethod",  # Quark INT4 FP8线性方法
]

_is_cpu = is_cpu()  # 检测是否为CPU平台
_is_npu = is_npu()  # 检测是否为NPU平台


def adjust_marlin_shard(param, shard_size, shard_offset):  # 调整Marlin分片大小和偏移
    marlin_tile_size = getattr(param, "marlin_tile_size", None)  # 获取Marlin瓦片大小
    if marlin_tile_size is None:  # 如果没有Marlin瓦片大小
        return shard_size, shard_offset  # 返回原始值

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size  # 乘以瓦片大小返回


def adjust_bitsandbytes_4bit_shard(  # 调整BitsAndBytes 4位量化的分片
    param: Parameter, shard_offsets: Dict[str, Tuple[int, int]], loaded_shard_id: str
) -> Tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding.
    调整BitsAndBytes分片的量化偏移和大小。"""

    total, _ = shard_offsets["total"]  # 获取总量
    orig_offset, orig_size = shard_offsets[loaded_shard_id]  # 获取原始偏移和大小

    quantized_total = param.data.shape[0]  # 获取量化后的总量
    quantized_offset = orig_offset * quantized_total // total  # 计算量化偏移
    quantized_size = orig_size * quantized_total // total  # 计算量化大小

    return quantized_size, quantized_offset  # 返回量化大小和偏移


def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):  # 调整标量到融合数组
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    对于融合模块（QKV和MLP），我们有一个长度为N的数组，
    每个逻辑矩阵对应1个缩放值。param是长度为N的数组。
    loaded_weight对应磁盘上的一个分片。这里根据shard_id切片param进行加载。
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}  # QKV索引映射

    if isinstance(shard_id, str):  # 如果shard_id是字符串
        shard_id = qkv_idxs[shard_id]  # 转换为整数索引
    elif not isinstance(shard_id, int):  # 如果既不是字符串也不是整数
        raise ValueError(f"Unknown Shard Id {shard_id}")  # 抛出错误

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    # AutoFP8缩放值没有形状
    # compressed-tensors缩放值有形状
    if len(loaded_weight.shape) != 0:  # 如果有形状
        assert loaded_weight.shape[0] == 1  # 断言第一维为1
        loaded_weight = loaded_weight[0]  # 去掉第一维

    return param[shard_id], loaded_weight  # 返回切片后的参数和权重


def adjust_shard_offsets(shard_offsets, loaded_weight, dim):  # 调整分片偏移
    actual_weight_size = loaded_weight.size(dim)  # 获取实际权重大小
    target_weight_size = shard_offsets[-1][-1] + shard_offsets[-1][-2]  # 获取目标权重大小
    if actual_weight_size != target_weight_size:  # 如果不匹配
        new_shard_offsets = []  # 新的分片偏移列表
        new_offset = 0  # 新偏移
        for shard_id, shard_offset, shard_size in shard_offsets:  # 遍历每个分片
            actual_shard_size = actual_weight_size * shard_size // target_weight_size  # 计算实际分片大小
            new_shard_offsets.append((shard_id, new_offset, actual_shard_size))  # 添加新偏移
            new_offset += actual_shard_size  # 更新偏移
        return new_shard_offsets  # 返回新偏移列表
    return shard_offsets  # 返回原始偏移


class LinearBase(torch.nn.Module):  # 线性层基类
    """Base linear layer.
    基础线性层。

    Args:
        input_size: input dimension of the linear layer.
        input_size: 线性层的输入维度。
        output_size: output dimension of the linear layer.
        output_size: 线性层的输出维度。
        bias: If true, add bias.
        bias: 如果为True，添加偏置。
        skip_bias_add: If true, skip adding bias but instead return it.
        skip_bias_add: 如果为True，跳过添加偏置，而是返回它。
        params_dtype: Data type for the parameters.
        params_dtype: 参数的数据类型。
        quant_config: Quantization configure.
        quant_config: 量化配置。
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度
        skip_bias_add: bool = False,  # 是否跳过偏置添加
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 层前缀
    ):
        super().__init__()  # 调用父类初始化

        # Keep input parameters
        # 保存输入参数
        self.input_size = input_size  # 输入维度
        self.output_size = output_size  # 输出维度
        self.skip_bias_add = skip_bias_add  # 跳过偏置添加标志
        if params_dtype is None:  # 如果未指定数据类型
            params_dtype = torch.get_default_dtype()  # 使用默认数据类型
        self.params_dtype = params_dtype  # 保存参数数据类型
        self.quant_config = quant_config  # 保存量化配置
        if quant_config is None:  # 如果没有量化配置
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化方法

            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()  # 使用未量化方法
        else:  # 有量化配置
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)  # 获取量化方法

        if self.quant_method is not None:  # 如果有量化方法
            wrap_method_with_debug_kernel_once(  # 包装调试内核
                self.quant_method,
                "apply",
                op_name=f"sglang.quant_method.{self.quant_method.__class__.__name__}.apply",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播（需子类实现）
        raise NotImplementedError  # 抛出未实现异常


class ReplicatedLinear(LinearBase):  # 复制线性层（无并行）
    """Replicated linear layer.
    复制线性层。

    Args:
        input_size: input dimension of the linear layer.
        input_size: 线性层的输入维度。
        output_size: output dimension of the linear layer.
        output_size: 线性层的输出维度。
        bias: If true, add bias.
        bias: 如果为True，添加偏置。
        skip_bias_add: If true, skip adding bias but instead return it.
        skip_bias_add: 如果为True，跳过添加偏置，而是返回它。
        params_dtype: Data type for the parameters.
        params_dtype: 参数的数据类型。
        quant_config: Quantization configure.
        quant_config: 量化配置。
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        prefix: 在状态字典中的层名称，包含所有父级
                        （例如 model.layers.0.qkv_proj）
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度
        bias: bool = True,  # 是否添加偏置
        skip_bias_add: bool = False,  # 是否跳过偏置添加
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 层前缀
    ):
        super().__init__(  # 调用父类初始化
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
        )

        # All the linear layer supports quant method.
        # 所有线性层都支持量化方法。
        assert self.quant_method is not None  # 断言量化方法存在
        self.quant_method.create_weights(  # 创建权重
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,  # 权重加载器
        )

        if bias:  # 如果需要偏置
            self.bias = Parameter(  # 创建偏置参数
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(  # 设置权重属性
                self.bias,
                {
                    "output_dim": 0,  # 输出维度索引
                    "weight_loader": self.weight_loader,  # 权重加载器
                },
            )
        else:  # 不需要偏置
            self.register_parameter("bias", None)  # 注册None偏置

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):  # 权重加载器
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        # 如果磁盘上的权重没有形状，给它一个
        # （例如AutoFp8的缩放值）。
        if len(loaded_weight.shape) == 0:  # 如果权重没有形状
            loaded_weight = loaded_weight.reshape(1)  # 重塑为1维

        # The per-tensor quant-scale must be 1 dimension
        # 逐张量量化缩放值必须是1维
        if _is_npu:  # NPU特殊处理
            if param.size() != loaded_weight.size() and param.size(0) == 1:  # 如果大小不匹配且参数第一维为1
                if torch.allclose(loaded_weight, loaded_weight[0]):  # 如果所有值相同
                    loaded_weight = loaded_weight[:1]  # 只取第一个
                else:  # 值不相同
                    raise ValueError(f"{loaded_weight} are not all equal")  # 抛出错误

            if param.dtype == torch.int8 or loaded_weight.dtype == torch.int8:  # INT8类型检查
                assert (
                    param.dtype == loaded_weight.dtype
                ), "init para dtype and loaded weight dtype should be the same"  # 断言类型一致

        assert (  # 断言参数和权重大小一致
            param.size() == loaded_weight.size()
        ), f"{param.shape=} {param.dtype=} {loaded_weight.shape=} {loaded_weight.dtype=}"
        param.data.copy_(loaded_weight)  # 复制权重数据

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 前向传播
        bias = self.bias if not self.skip_bias_add else None  # 获取偏置
        assert self.quant_method is not None  # 断言量化方法存在
        output = self.quant_method.apply(self, x, bias)  # 应用量化方法
        output_bias = self.bias if self.skip_bias_add else None  # 获取输出偏置
        return output, output_bias  # 返回输出和偏置

    def extra_repr(self) -> str:  # 额外表示方法
        s = f"in_features={self.input_size}"  # 输入特征数
        s += f", output_features={self.output_size}"  # 输出特征数
        s += f", bias={self.bias is not None}"  # 是否有偏置
        return s  # 返回字符串


class ColumnParallelLinear(LinearBase):  # 列并行线性层
    """Linear layer with column parallelism.
    列并行线性层。

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    线性层定义为Y = XA + b。A沿第二维度并行化为A = [A_1, ..., A_p]。

    Args:
        input_size: first dimension of matrix A.
        input_size: 矩阵A的第一维度。
        output_size: second dimension of matrix A.
        output_size: 矩阵A的第二维度。
        bias: If true, add bias.
        bias: 如果为True，添加偏置。
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        gather_output: 如果为True，对输出调用全收集使Y对所有GPU可见；
                       否则每个GPU有自己的输出Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        skip_bias_add: 添加此选项以启用性能优化，偏置可与其他逐元素操作融合。
                       跳过添加偏置，而是返回它。
        params_dtype: Data type for the parameters.
        params_dtype: 参数的数据类型。
        quant_config: Quantization configure.
        quant_config: 量化配置。
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        output_sizes: 打包到一个输出中的输出大小列表，如QKV的大小为3。
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        prefix: 在状态字典中的层名称，包含所有父级
                        （例如 model.layers.0.qkv_proj）
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度
        bias: bool = True,  # 是否添加偏置
        gather_output: bool = False,  # 是否全收集输出
        skip_bias_add: bool = False,  # 是否跳过偏置添加
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        output_sizes: Optional[List[int]] = None,  # 输出大小列表
        prefix: str = "",  # 层前缀
        tp_rank: Optional[int] = None,  # TP rank
        tp_size: Optional[int] = None,  # TP大小
        use_presharded_weights: bool = False,  # 是否使用预分片权重
        skip_block_quant_check: bool = False,  # 是否跳过块量化检查
    ):
        super().__init__(  # 调用父类初始化
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.gather_output = gather_output  # 保存全收集标志
        self.use_presharded_weights = use_presharded_weights  # 保存预分片权重标志

        # Divide the weight matrix along the last dimension.
        # 沿最后维度分割权重矩阵。
        if tp_rank is None:  # 如果未指定TP rank
            tp_rank = get_tensor_model_parallel_rank()  # 获取当前TP rank
        if tp_size is None:  # 如果未指定TP大小
            tp_size = get_tensor_model_parallel_world_size()  # 获取当前TP世界大小
        self.tp_rank, self.tp_size = tp_rank, tp_size  # 保存TP rank和大小
        assert self.quant_method is not None  # 断言量化方法存在
        self.output_size_per_partition = divide(self.output_size, tp_size)  # 计算每个分区的输出大小
        self.output_partition_sizes = [self.output_size_per_partition]  # 输出分区大小列表
        # If QKV or MergedColumn, use output size of each partition.
        # 如果是QKV或MergedColumn，使用每个分区的输出大小。
        if hasattr(self, "output_sizes"):  # 如果有output_sizes属性
            self.output_partition_sizes = [
                divide(output_size, tp_size) for output_size in self.output_sizes
            ]

        if output_sizes is None:  # 如果没有输出大小列表
            output_sizes = [output_size]  # 使用总输出大小

        self.quant_method.create_weights(  # 创建权重
            layer=self,
            input_size_per_partition=self.input_size,  # 输入大小
            output_partition_sizes=self.output_partition_sizes,  # 输出分区大小
            input_size=self.input_size,  # 输入大小
            output_size=self.output_size,  # 输出大小
            params_dtype=self.params_dtype,  # 参数数据类型
            skip_block_quant_check=skip_block_quant_check,  # 跳过块量化检查
            weight_loader=(  # 权重加载器选择
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:  # 如果需要偏置
            self.bias = Parameter(  # 创建偏置参数
                torch.zeros(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(  # 设置权重属性
                self.bias,
                {
                    "output_dim": 0,  # 输出维度索引
                    "weight_loader": self.weight_loader,  # 权重加载器
                },
            )
        else:  # 不需要偏置
            self.register_parameter("bias", None)  # 注册None偏置

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):  # V1权重加载器
        output_dim = getattr(param, "output_dim", None)  # 获取输出维度
        param_data = param.data  # 获取参数数据

        # Special case for GGUF
        # GGUF特殊情况
        is_gguf_weight = getattr(param, "is_gguf_weight", False)  # 是否为GGUF权重
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)  # 是否为GGUF权重类型
        if is_gguf_weight_type:  # 如果是GGUF权重类型
            param.weight_type = loaded_weight.item()  # 设置权重类型

        # Materialize GGUF UninitializedParameter
        # 物化GGUF未初始化参数
        if is_gguf_weight and isinstance(param, UninitializedParameter):  # GGUF未初始化参数
            weight_shape = list(loaded_weight.shape)  # 获取权重形状
            if output_dim is not None:  # 如果有输出维度
                weight_shape[output_dim] = weight_shape[output_dim] // self.tp_size  # 按TP大小分割
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)  # 物化参数
            param_data = param.data  # 更新参数数据

        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        # bitsandbytes加载特定部分的权重，无需在此收窄
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)  # 是否使用BNB 4位
        if output_dim is not None and not use_bitsandbytes_4bit:  # 如果有输出维度且非BNB
            shard_size = param_data.shape[output_dim]  # 获取分片大小
            start_idx = self.tp_rank * shard_size  # 计算起始索引

            if _is_cpu:  # CPU平台
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(  # 收窄填充的参数
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,  # 起始索引
                    output_dim,  # 输出维度
                    shard_size,  # 分片大小
                    not self.use_presharded_weights,  # 是否需要收窄加载权重
                )
            else:  # 非CPU平台
                if not self.use_presharded_weights:  # 如果未使用预分片权重
                    loaded_weight = loaded_weight.narrow(  # 收窄加载权重
                        output_dim, start_idx, shard_size
                    )

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 从磁盘加载缩放值的特殊情况，通常没有形状（如AutoFP8）。
        if len(loaded_weight.shape) == 0:  # 如果权重没有形状
            loaded_weight = loaded_weight.reshape(1)  # 重塑为1维

        assert (  # 断言参数和权重形状一致
            param_data.shape == loaded_weight.shape
        ), f"param_data.shape={param_data.shape} != loaded_weight.shape={loaded_weight.shape}"
        param_data.copy_(loaded_weight)  # 复制权重数据

    def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):  # V2权重加载器
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 从磁盘加载缩放值的特殊情况，通常没有形状（如AutoFP8）。
        if len(loaded_weight.shape) == 0:  # 如果权重没有形状
            assert loaded_weight.numel() == 1  # 断言只有一个元素
            loaded_weight = loaded_weight.reshape(1)  # 重塑为1维

        if isinstance(param, _ColumnvLLMParameter):  # 如果是列vLLM参数
            param.load_column_parallel_weight(  # 加载列并行权重
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:  # 其他参数类型
            # FIXME: This branch is needed to load deepseek v3 awq.
            # However, we should fix this and avoid the branching here.
            # After QuantizedRL reload, params might still need tp_rank
            # FIXME: 此分支用于加载deepseek v3 awq。
            # 但我们应修复此问题并避免分支。
            # QuantizedRL重载后，参数可能仍需要tp_rank
            try:  # 尝试带TP参数加载
                param.load_column_parallel_weight(
                    loaded_weight,
                    tp_rank=self.tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            except TypeError:  # 如果参数不支持额外参数
                # Fallback for parameters that don't accept additional args
                # 不接受额外参数的回退方案
                param.load_column_parallel_weight(loaded_weight)  # 不带TP参数加载

    def forward(self, input_):  # 前向传播
        bias = self.bias if not self.skip_bias_add else None  # 获取偏置

        # Matrix multiply.
        # 矩阵乘法。
        assert self.quant_method is not None  # 断言量化方法存在
        output_parallel = self.quant_method.apply(self, input_, bias)  # 应用量化方法
        if self.gather_output:  # 如果需要全收集输出
            # All-gather across the partitions.
            # 跨分区全收集。
            output = tensor_model_parallel_all_gather(output_parallel)  # 执行全收集
        else:  # 不需要全收集
            output = output_parallel  # 使用并行输出
        output_bias = self.bias if self.skip_bias_add else None  # 获取输出偏置
        return output, output_bias  # 返回输出和偏置

    def extra_repr(self) -> str:  # 额外表示方法
        s = f"in_features={self.input_size}"  # 输入特征数
        s += f", output_features={self.output_size_per_partition}"  # 输出特征数（每分区）
        s += f", bias={self.bias is not None}"  # 是否有偏置
        s += f", tp_size={self.tp_size}"  # TP大小
        s += f", gather_output={self.gather_output}"  # 是否全收集
        return s  # 返回字符串


class MergedColumnParallelLinear(ColumnParallelLinear):  # 合并列并行线性层
    """Packed linear layers with column parallelism.
    带列并行的打包线性层。

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.
    类似于ColumnParallelLinear，但权重矩阵沿输出维度拼接。
    加载权重矩阵时，不同分区分别进行分片。

    Args:
        input_size: input dimension of the linear layer.
        input_size: 线性层的输入维度。
        output_sizes: list of output dimensions of the linear layer.
        output_sizes: 线性层的输出维度列表。
        bias: If true, add bias.
        bias: 如果为True，添加偏置。
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        gather_output: 如果为True，对输出调用全收集使输出对所有GPU可见；
                       否则每个GPU有自己的输出。
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        skip_bias_add: 添加此选项以启用性能优化，偏置可与其他逐元素操作融合。
                       跳过添加偏置，而是返回它。
        params_dtype: Data type for the parameters.
        params_dtype: 参数的数据类型。
        quant_config: Quantization configure.
        quant_config: 量化配置。
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        prefix: 在状态字典中的层名称，包含所有父级
                        （例如 model.layers.0.qkv_proj）
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度
        output_sizes: List[int],  # 输出维度列表
        bias: bool = True,  # 是否添加偏置
        gather_output: bool = False,  # 是否全收集输出
        skip_bias_add: bool = False,  # 是否跳过偏置添加
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 层前缀
        tp_rank: Optional[int] = None,  # TP rank
        tp_size: Optional[int] = None,  # TP大小
        use_presharded_weights: bool = False,  # 是否使用预分片权重
    ):
        self.output_sizes = output_sizes  # 保存输出大小列表
        if tp_rank is None:  # 如果未指定TP rank
            tp_rank = get_tensor_model_parallel_rank()  # 获取当前TP rank
        if tp_size is None:  # 如果未指定TP大小
            tp_size = get_tensor_model_parallel_world_size()  # 获取当前TP世界大小
        self.tp_rank, self.tp_size = tp_rank, tp_size  # 保存TP rank和大小
        assert all(output_size % tp_size == 0 for output_size in output_sizes)  # 断言所有输出大小可被TP整除
        self.use_presharded_weights = use_presharded_weights  # 保存预分片权重标志
        super().__init__(  # 调用父类初始化
            input_size=input_size,
            output_size=sum(output_sizes),  # 总输出大小为所有输出之和
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=use_presharded_weights,
        )
        self.prefix = prefix  # 保存前缀

    def weight_loader(  # V1权重加载器
        self,
        param: Parameter,  # 参数
        loaded_weight: torch.Tensor,  # 加载的权重
        loaded_shard_id: tuple[int, ...] | int | None = None,  # 加载的分片ID
    ):
        if isinstance(loaded_shard_id, tuple):  # 如果分片ID是元组（多索引）
            if hasattr(param, "load_merged_column_weight"):  # 如果参数支持V2加载
                return self.weight_loader_v2(param, loaded_weight, loaded_shard_id)  # 使用V2
            raise NotImplementedError(  # 抛出未实现错误
                "Shard id with multiple indices is not supported in weight_loader, "
                "please use weight_loader_v2 instead."
            )

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        # GGUF特殊情况
        # 在知道量化类型后初始化GGUF参数
        is_gguf_weight = getattr(param, "is_gguf_weight", False)  # 是否为GGUF权重
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)  # 是否为GGUF权重类型
        if is_gguf_weight_type:  # 如果是GGUF权重类型
            param.data[loaded_shard_id].copy_(loaded_weight)  # 复制权重
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()  # 设置权重类型
            return  # 返回

        if is_gguf_weight:  # 如果是GGUF权重
            output_dim = getattr(param, "output_dim", None)  # 获取输出维度
            shard_size = loaded_weight.size(output_dim) // self.tp_size  # 计算分片大小
            start_idx = self.tp_rank * shard_size  # 计算起始索引

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)  # 收窄权重

            param.shard_id.append(loaded_shard_id)  # 添加分片ID
            param.shard_id_map[loaded_shard_id] = len(param.data_container)  # 映射分片ID
            param.data_container.append(loaded_weight)  # 添加到数据容器
            return  # 返回

        param_data = param.data  # 获取参数数据
        output_dim = getattr(param, "output_dim", None)  # 获取输出维度
        # Special case for AQLM codebooks.
        # AQLM码本的特殊情况。
        is_metadata = getattr(param, "is_metadata", False)  # 是否为元数据
        # Special case for per-tensor scale to load scalar into fused array.
        # 逐张量缩放值加载标量到融合数组的特殊情况。
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)  # 是否需要标量转数组

        if loaded_shard_id is None:  # 如果没有分片ID（已在磁盘上融合）
            # Loaded weight is already fused on disk (qkv/mlp).
            # 加载的权重已在磁盘上融合（qkv/mlp）。
            if output_dim is None:  # 如果没有输出维度
                if needs_scalar_to_array:  # 如果需要标量转数组
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape  # 断言形状一致
                param_data.copy_(loaded_weight)  # 复制数据
                return  # 返回
            current_shard_offset = 0  # 当前分片偏移
            shard_offsets: List[Tuple[int, int, int]] = []  # 分片偏移列表
            for i, output_size in enumerate(self.output_sizes):  # 遍历输出大小
                effective_size = (  # 计算有效大小
                    output_size // self.tp_size
                    if self.use_presharded_weights
                    else output_size
                )
                shard_offsets.append((i, current_shard_offset, effective_size))  # 添加偏移
                current_shard_offset += effective_size  # 更新偏移
            packed_dim = getattr(param, "packed_dim", None)  # 获取打包维度

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)  # BNB 4位标志
            if _is_cpu:  # CPU平台
                shard_offsets = adjust_shard_offsets(  # 调整分片偏移
                    shard_offsets, loaded_weight, output_dim
                )

            for shard_id, shard_offset, shard_size in shard_offsets:  # 遍历每个分片
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                # 量化的特殊情况。
                # 如果量化，需要调整偏移和大小以考虑打包。
                if packed_dim == output_dim:  # 如果打包维度等于输出维度
                    shard_size = shard_size // param.pack_factor  # 调整大小
                    shard_offset = shard_offset // param.pack_factor  # 调整偏移
                    # Special case for Marlin.
                    # Marlin特殊情况。
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:  # BNB 4位量化
                    index = list(itertools.accumulate([0] + self.output_sizes))  # 累积索引
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)  # 总量
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id)
                    )

                loaded_weight_shard = loaded_weight.narrow(  # 收窄到当前分片
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)  # 递归加载
            return  # 返回

        assert loaded_shard_id < len(self.output_sizes)  # 断言分片ID有效
        if output_dim is not None:  # 如果有输出维度
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size  # 计算分片偏移
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size  # 计算分片大小
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 量化的特殊情况。
            # 如果量化，需要调整偏移和大小以考虑打包。
            packed_dim = getattr(param, "packed_dim", None)  # 获取打包维度
            if packed_dim == output_dim:  # 如果打包维度等于输出维度
                shard_size = shard_size // param.pack_factor  # 调整大小
                shard_offset = shard_offset // param.pack_factor  # 调整偏移
                # Special case for Marlin.
                # Marlin特殊情况。
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)  # BNB 4位
            if use_bitsandbytes_4bit:  # BNB 4位量化
                shard_size = loaded_weight.shape[output_dim]  # 使用加载权重大小
                shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id  # 计算偏移

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)  # 收窄参数
            start_idx = self.tp_rank * shard_size  # 计算起始索引

            if _is_cpu:  # CPU平台
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(  # 收窄填充参数
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    output_dim,
                    shard_size,
                    not use_bitsandbytes_4bit and not self.use_presharded_weights,
                )
            else:  # 非CPU平台
                # bitsandbytes loads the weights of the specific portion
                # no need to narrow here
                # bitsandbytes加载特定部分的权重，无需在此收窄
                if not use_bitsandbytes_4bit and not self.use_presharded_weights:  # 需要收窄
                    # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                    # 对qwen2_5_VL的mlp等非8对齐的特殊情况进行填充
                    end_idx = start_idx + shard_size  # 计算结束索引
                    if end_idx > loaded_weight.shape[output_dim]:  # 如果超出范围
                        loaded_weight = pad_or_narrow_weight(  # 填充或收窄权重
                            loaded_weight, output_dim, start_idx, shard_size
                        )
                    else:  # 未超出范围
                        loaded_weight = loaded_weight.narrow(  # 直接收窄
                            output_dim, start_idx, shard_size
                        )

        # Special case for AQLM codebooks.
        # AQLM码本的特殊情况。
        elif is_metadata:  # 如果是元数据
            # metadata indicates fixed size concatenated along dim 0
            # 元数据表示固定大小沿维度0拼接
            shard_size = loaded_weight.shape[0]  # 获取分片大小
            shard_offset = loaded_shard_id * shard_size  # 计算偏移
            param_data = param_data.narrow(0, shard_offset, shard_size)  # 收窄参数

        # Special case for per-tensor scales in fused case.
        # 融合情况下逐张量缩放值的特殊情况。
        elif needs_scalar_to_array:  # 如果需要标量转数组
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:  # 其他情况
            ignore_warning = getattr(param, "ignore_warning", False)  # 是否忽略警告
            if not ignore_warning:  # 如果不忽略
                logger.warning(  # 记录警告
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape  # 断言形状一致
        param_data.copy_(loaded_weight)  # 复制权重数据

    def _load_fused_module_from_checkpoint(  # 从检查点加载融合模块
        self,
        param: BasevLLMParameter,  # 参数
        loaded_weight: torch.Tensor,  # 加载的权重
        output_sizes: list[int] | None = None,  # 输出大小列表
    ):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.
        处理MLP层已在磁盘上融合的模型的特殊情况。
        此时没有分片ID。此函数通过分割这些层确定分片ID，
        然后使用分片ID调用权重加载器。

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        具有这些融合层的模型示例：
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0  # 当前分片偏移
        shard_offsets: List[Tuple[int, int, int]] = []  # 分片偏移列表
        output_sizes = output_sizes or self.output_sizes  # 使用指定或默认输出大小
        for i, output_size in enumerate(output_sizes):  # 遍历输出大小
            shard_offsets.append((i, current_shard_offset, output_size))  # 添加偏移
            current_shard_offset += output_size  # 更新偏移
        if _is_cpu:  # CPU平台
            from sglang.srt.model_loader.weight_utils import (
                pad_loaded_weight,
            )

            loaded_weight = pad_loaded_weight(  # 填充加载权重
                loaded_weight, param.output_dim, output_sizes
            )

        for shard_id, shard_offset, shard_size in shard_offsets:  # 遍历每个分片
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 量化的特殊情况。
            # 如果量化，需要调整偏移和大小以考虑打包。
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )
            loaded_weight_shard = loaded_weight.narrow(  # 收窄到当前分片
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)  # 使用V2加载

    def _load_merged_block_scale(  # 加载合并的块缩放
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle block-wise scale loading for MergedColumnParallelLinear.
        Similar to QKVParallelLinear._load_qkv_block_scale, but for merged column layers.
        处理MergedColumnParallelLinear的块级缩放加载。
        类似于QKVParallelLinear._load_qkv_block_scale，但用于合并列层。
        """
        weight_block_size = self.quant_method.quant_config.weight_block_size  # 获取权重块大小
        block_n, _ = weight_block_size[0], weight_block_size[1]  # 获取块维度
        block_n = 1 if getattr(param, "format_ue8m0", False) else block_n  # UE8M0格式使用1

        # Calculate block sizes for each shard
        # 计算每个分片的块大小
        shard_block_sizes = []  # 分片块大小列表
        shard_block_offsets = []  # 分片块偏移列表
        current_block_offset = 0  # 当前块偏移
        for output_size in self.output_sizes:  # 遍历输出大小
            shard_block_size = (output_size + block_n - 1) // block_n  # 计算块大小
            shard_block_sizes.append(shard_block_size)  # 添加块大小
            shard_block_offsets.append(current_block_offset)  # 添加块偏移
            current_block_offset += shard_block_size  # 更新偏移

        if _is_cpu:  # CPU平台
            from sglang.srt.model_loader.weight_utils import (
                pad_loaded_weight,
            )

            loaded_weight = pad_loaded_weight(  # 填充加载权重
                loaded_weight, param.output_dim, shard_block_sizes
            )

        # Load each shard
        # 加载每个分片
        for shard_id, (shard_block_offset, shard_block_size) in enumerate(
            zip(shard_block_offsets, shard_block_sizes)
        ):
            # Extract the shard from loaded_weight
            # 从加载权重中提取分片
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_block_offset, shard_block_size
            )

            # Calculate per-rank offset and size (considering TP)
            # 计算每个rank的偏移和大小（考虑TP）
            rank_shard_offset = shard_block_offset // self.tp_size
            rank_shard_size = shard_block_size // self.tp_size

            # Load into the parameter
            # 加载到参数中
            param.load_merged_column_weight(
                loaded_weight=loaded_weight_shard,
                shard_id=shard_id,
                shard_offset=rank_shard_offset,
                shard_size=rank_shard_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                use_presharded_weights=self.use_presharded_weights,
            )

    def weight_loader_v2(  # V2权重加载器
        self,
        param: BasevLLMParameter,  # 参数
        loaded_weight: torch.Tensor,  # 加载的权重
        loaded_shard_id: tuple[int, ...] | int | None = None,  # 加载的分片ID
    ):
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):  # 无分片ID或多索引
            if isinstance(param, PerTensorScaleParameter):  # 逐张量缩放参数
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    shard_id=0,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return  # 返回
            elif isinstance(param, BlockQuantScaleParameter):  # 块量化缩放参数
                self._load_merged_block_scale(param, loaded_weight)
                return  # 返回
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):  # 行或基础参数
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return  # 返回
            output_sizes = (  # 计算输出大小列表
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            # TODO: @dsikka - move to parameter.py
            # TODO: @dsikka - 移到parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return  # 返回

        assert loaded_shard_id < len(self.output_sizes)  # 断言分片ID有效

        if isinstance(param, BlockQuantScaleParameter):  # 块量化缩放参数
            weight_block_size = self.quant_method.quant_config.weight_block_size
            raw_block_n, _ = weight_block_size[0], weight_block_size[1]
            block_n = 1 if getattr(param, "format_ue8m0", False) else raw_block_n
            shard_offset = (
                (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) // block_n
            ) // self.tp_size
            shard_size = (
                (self.output_sizes[loaded_shard_id] + block_n - 1)
                // block_n
                // self.tp_size
            )
        else:  # 普通参数
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size  # 分片偏移
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size  # 分片大小

        param.load_merged_column_weight(  # 加载合并列权重
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            use_presharded_weights=self.use_presharded_weights,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )


class QKVParallelLinear(ColumnParallelLinear):  # QKV并行线性层
    """Linear layers for the attention's QKV transformation.
    注意力QKV变换的线性层。

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    用于注意力层中查询、键和值向量线性变换的线性层。
    权重矩阵沿输出维度拼接。层沿头维度并行化。
    当键/值头数小于查询头数（如多查询/分组查询注意力）时，
    键/值头可能被复制而查询头被分区。

    Args:
        hidden_size: input hidden state size of the transformer.
        hidden_size: 变换器的输入隐藏状态大小。
        head_size: size of each attention head.
        head_size: 每个注意力头的大小。
        total_num_heads: total number of attention query heads.
        total_num_heads: 注意力查询头的总数。
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        total_num_kv_heads: 注意力键/值头的总数。如果为None，假设等于total_num_heads。
        bias: If true, add bias.
        bias: 如果为True，添加偏置。
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        skip_bias_add: 添加此选项以启用性能优化，偏置可与其他逐元素操作融合。
                       跳过添加偏置，而是返回它。
        params_dtype: Data type for the parameters.
        params_dtype: 参数的数据类型。
        quant_config: Quantization configure.
        quant_config: 量化配置。
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        prefix: 在状态字典中的层名称，包含所有父级
                        （例如 model.layers.0.qkv_proj）
    """

    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏状态大小
        head_size: int,  # 头大小
        total_num_heads: int,  # 查询头总数
        total_num_kv_heads: Optional[int] = None,  # 键值头总数
        bias: bool = True,  # 是否添加偏置
        skip_bias_add: bool = False,  # 是否跳过偏置添加
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 层前缀
        tp_rank: Optional[int] = None,  # TP rank
        tp_size: Optional[int] = None,  # TP大小
        load_presharded_attn: bool = False,  # 是否加载预分片注意力
        v_head_size: Optional[int] = None,  # V头大小
        skip_block_quant_check: bool = False,  # 是否跳过块量化检查
    ):
        self.hidden_size = hidden_size  # 保存隐藏大小
        self.head_size = head_size  # 保存头大小
        self.v_head_size = v_head_size if v_head_size is not None else head_size  # V头大小
        self.total_num_heads = total_num_heads  # 保存查询头总数
        if total_num_kv_heads is None:  # 如果未指定KV头数
            total_num_kv_heads = total_num_heads  # 默认与查询头数相同
        self.total_num_kv_heads = total_num_kv_heads  # 保存KV头总数
        # Divide the weight matrix along the last dimension.
        # 沿最后维度分割权重矩阵。
        if tp_rank is None:  # 如果未指定TP rank
            tp_rank = get_tensor_model_parallel_rank()  # 获取当前TP rank
        if tp_size is None:  # 如果未指定TP大小
            tp_size = get_tensor_model_parallel_world_size()  # 获取当前TP世界大小
        self.tp_rank, self.tp_size = tp_rank, tp_size  # 保存TP rank和大小
        self.num_heads = divide(self.total_num_heads, tp_size)  # 当前rank的头数
        if tp_size >= self.total_num_kv_heads:  # 如果TP大小大于等于KV头数
            self.num_kv_heads = 1  # 每个rank至少1个KV头
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)  # KV头复制数
        else:  # TP大小小于KV头数
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)  # 当前rank的KV头数
            self.num_kv_head_replicas = 1  # 无复制
        self.q_proj_shard_size = self.num_heads * self.head_size  # Q投影分片大小
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size  # KV投影分片大小
        self.v_proj_shard_size = self.num_kv_heads * self.v_head_size  # V投影分片大小
        input_size = self.hidden_size  # 输入大小
        output_size = (  # 总输出大小
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [  # 各投影的输出大小
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]
        self.use_presharded_weights = load_presharded_attn  # 预分片权重标志
        quant_config = None if _disable_hip_linear_quant else quant_config  # HIP禁用线性量化

        super().__init__(  # 调用父类初始化
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,  # QKV不收集输出
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=self.use_presharded_weights,
            skip_block_quant_check=skip_block_quant_check,
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str):  # 获取分片偏移映射
        shard_offset_mapping = {  # 分片偏移映射表
            "q": 0,  # Q偏移为0
            "k": self.num_heads * self.head_size,  # K偏移在Q之后
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,  # V偏移在K之后
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,  # 总大小
        }
        return shard_offset_mapping.get(loaded_shard_id)  # 返回偏移值

    def _get_shard_size_mapping(self, loaded_shard_id: str):  # 获取分片大小映射
        shard_size_mapping = {  # 分片大小映射表
            "q": self.num_heads * self.head_size,  # Q大小
            "k": self.num_kv_heads * self.head_size,  # K大小
            "v": self.num_kv_heads * self.v_head_size,  # V大小
        }
        return shard_size_mapping.get(loaded_shard_id)  # 返回大小值

    def _load_fused_module_from_checkpoint(  # 从检查点加载融合QKV模块
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.
        处理QKV层已在磁盘上融合的模型的特殊情况。
        此时没有分片ID。此函数通过分割这些层确定分片ID，
        然后使用分片ID调用权重加载器。

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        具有这些融合层的模型示例：
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [  # 分片偏移列表
            # (shard_id, shard_offset, shard_size)
            # (分片ID, 分片偏移, 分片大小)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:  # 遍历每个分片
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 量化的特殊情况。
            # 如果量化，需要调整偏移和大小以考虑打包。
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            if not self.use_presharded_weights:  # 如果未使用预分片权重
                loaded_weight_shard = loaded_weight.narrow(
                    param.output_dim, shard_offset, shard_size
                )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)  # 使用V2加载

    def _load_qkv_block_scale(  # 加载QKV块缩放
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        block_n, _ = self.quant_method.quant_config.weight_block_size  # 获取块大小
        q_size = self.total_num_heads * self.head_size // block_n  # Q的块数
        k_size = self.total_num_kv_heads * self.head_size // block_n  # K的块数
        v_size = self.total_num_kv_heads * self.v_head_size // block_n  # V的块数
        shard_offsets = [  # 分片偏移列表
            # (shard_id, shard_offset, shard_size)
            # (分片ID, 分片偏移, 分片大小)
            ("q", 0, q_size),
            ("k", q_size, k_size),
            ("v", q_size + k_size, v_size),
        ]
        for shard_id, shard_offset, shard_size in shard_offsets:  # 遍历每个分片
            loaded_weight_shard = loaded_weight.narrow(  # 收窄到当前分片
                param.output_dim, shard_offset, shard_size
            )
            rank_shard_offset = self._get_shard_offset_mapping(shard_id) // block_n  # rank级偏移
            rank_shard_size = self._get_shard_size_mapping(shard_id) // block_n  # rank级大小
            param.load_qkv_weight(  # 加载QKV权重
                loaded_weight=loaded_weight_shard,
                num_heads=self.num_kv_head_replicas,
                shard_id=shard_id,
                shard_offset=rank_shard_offset,
                shard_size=rank_shard_size,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )

    def weight_loader_v2(  # V2权重加载器
        self,
        param: BasevLLMParameter,  # 参数
        loaded_weight: torch.Tensor,  # 加载的权重
        loaded_shard_id: Optional[str] = None,  # 加载的分片ID
    ):
        if loaded_shard_id is None:  # special case for certain models
            # 某些模型的特殊情况
            if isinstance(param, PerTensorScaleParameter):  # 逐张量缩放
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return  # 返回
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):  # 行或基础参数
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return  # 返回
            elif isinstance(param, BlockQuantScaleParameter):  # 块量化缩放
                self._load_qkv_block_scale(param, loaded_weight)
                return  # 返回
            # TODO: @dsikka - move to parameter.py
            # TODO: @dsikka - 移到parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return  # 返回

        assert loaded_shard_id in ["q", "k", "v"]  # 断言分片ID有效

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)  # 获取偏移
        shard_size = self._get_shard_size_mapping(loaded_shard_id)  # 获取大小

        if isinstance(param, BlockQuantScaleParameter):  # 块量化缩放
            weight_block_size = self.quant_method.quant_config.weight_block_size
            raw_block_n, _ = weight_block_size[0], weight_block_size[1]
            block_n = 1 if getattr(param, "format_ue8m0", False) else raw_block_n
            shard_offset = (shard_offset + block_n - 1) // block_n  # 调整偏移
            shard_size = (shard_size + block_n - 1) // block_n  # 调整大小

        param.load_qkv_weight(  # 加载QKV权重
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
            use_presharded_weights=self.use_presharded_weights,
        )

    def weight_loader(  # V1权重加载器
        self,
        param: Parameter,  # 参数
        loaded_weight: torch.Tensor,  # 加载的权重
        loaded_shard_id: Optional[str] = None,  # 加载的分片ID
    ):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        # GGUF特殊情况
        # 在知道量化类型后初始化GGUF参数
        is_gguf_weight = getattr(param, "is_gguf_weight", False)  # 是否为GGUF权重
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)  # 是否为GGUF权重类型
        if is_gguf_weight_type and loaded_shard_id is not None:  # GGUF权重类型
            idx_map = {"q": 0, "k": 1, "v": 2}  # QKV索引映射
            param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)  # 复制权重
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()  # 设置权重类型
            return  # 返回

        if is_gguf_weight:  # GGUF权重
            output_dim = getattr(param, "output_dim", None)  # 获取输出维度
            shard_size = loaded_weight.size(output_dim) // self.tp_size  # 分片大小
            start_idx = self.tp_rank * shard_size  # 起始索引

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)  # 收窄

            param.shard_id.append(loaded_shard_id)  # 添加分片ID
            param.shard_id_map[loaded_shard_id] = len(param.data_container)  # 映射
            param.data_container.append(loaded_weight)  # 添加到容器
            return  # 返回

        param_data = param.data  # 参数数据
        output_dim = getattr(param, "output_dim", None)  # 输出维度
        # Special case for AQLM codebooks.
        # AQLM码本特殊情况。
        is_metadata = getattr(param, "is_metadata", False)  # 是否为元数据

        # Special case for per-tensor scales in fused case.
        # 融合情况逐张量缩放值的特殊情况。
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)  # 是否需要标量转数组

        if loaded_shard_id is None:  # 无分片ID（已融合）
            # Loaded weight is already fused on disk (qkv/mlp).
            # 加载的权重已在磁盘上融合（qkv/mlp）。
            if output_dim is None:  # 无输出维度
                if needs_scalar_to_array:  # 需要标量转数组
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape  # 断言形状一致
                param_data.copy_(loaded_weight)  # 复制数据
                return  # 返回
            shard_offsets = [  # 分片偏移列表
                # (shard_id, shard_offset, shard_size)
                # (分片ID, 分片偏移, 分片大小)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)  # BNB 4位

            packed_dim = getattr(param, "packed_dim", None)  # 打包维度
            if _is_cpu:  # CPU平台
                shard_offsets = adjust_shard_offsets(
                    shard_offsets, loaded_weight, output_dim
                )

            for shard_id, shard_offset, shard_size in shard_offsets:  # 遍历每个分片
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                # 量化权重的特殊情况。
                # 如果量化，需要调整偏移和大小以考虑打包。
                if packed_dim == output_dim:  # 打包维度等于输出维度
                    shard_size = shard_size // param.pack_factor  # 调整大小
                    shard_offset = shard_offset // param.pack_factor  # 调整偏移

                    # Special case for Marlin.
                    # Marlin特殊情况。
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:  # BNB 4位量化
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size,
                            self.total_num_kv_heads * self.v_head_size,
                        ),
                        "total": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size
                            + self.total_num_kv_heads * self.v_head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                if not self.use_presharded_weights:  # 未使用预分片权重
                    loaded_weight_shard = loaded_weight.narrow(
                        output_dim, shard_offset, shard_size
                    )
                self.weight_loader(param, loaded_weight_shard, shard_id)  # 递归加载
            return  # 返回

        assert loaded_shard_id in ["q", "k", "v"]  # 断言分片ID有效

        # If output dim is defined, use the default loading process.
        # 如果定义了输出维度，使用默认加载过程。
        if output_dim is not None:  # 有输出维度
            if loaded_shard_id == "q":  # Q分片
                shard_offset = 0  # Q偏移为0
                shard_size = self.num_heads * self.head_size  # Q大小
            elif loaded_shard_id == "k":  # K分片
                shard_offset = self.num_heads * self.head_size  # K偏移在Q之后
                shard_size = self.num_kv_heads * self.head_size  # K大小
            elif loaded_shard_id == "v":  # V分片
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size  # V偏移在K之后
                shard_size = self.num_kv_heads * self.v_head_size  # V大小
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 量化权重的特殊情况。
            # 如果量化，需要调整偏移和大小以考虑打包。
            packed_dim = getattr(param, "packed_dim", None)  # 打包维度
            if packed_dim == output_dim:  # 打包维度等于输出维度
                shard_size = shard_size // param.pack_factor  # 调整大小
                shard_offset = shard_offset // param.pack_factor  # 调整偏移

                # Special case for Marlin.
                # Marlin特殊情况。
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)  # BNB 4位
            if use_bitsandbytes_4bit:  # BNB 4位量化
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.v_head_size,
                    ),
                    "total": (
                        (self.num_heads + self.num_kv_heads) * self.head_size
                        + self.num_kv_heads * self.v_head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)  # 收窄参数
            if loaded_shard_id == "q":  # Q分片使用TP rank
                shard_id = self.tp_rank
            else:  # K/V分片考虑KV头复制
                shard_id = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size  # 起始索引

            if _is_cpu:  # CPU平台
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(  # 收窄填充参数
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    output_dim,
                    shard_size,
                    not use_bitsandbytes_4bit and not self.use_presharded_weights,
                )
            else:  # 非CPU平台
                # bitsandbytes loads the weights of the specific portion
                # no need to narrow here
                # bitsandbytes加载特定部分的权重，无需在此收窄
                if not use_bitsandbytes_4bit and not self.use_presharded_weights:  # 需要收窄
                    loaded_weight = loaded_weight.narrow(
                        output_dim, start_idx, shard_size
                    )

        # Special case for AQLM codebooks.
        # AQLM码本特殊情况。
        elif is_metadata:  # 元数据
            # metadata indicates fixed size concatenated along dim 0
            # 元数据表示固定大小沿维度0拼接
            shard_size = loaded_weight.shape[0]  # 分片大小
            shard_index = ["q", "k", "v"].index(loaded_shard_id)  # 分片索引
            param_data = param_data.narrow(0, shard_index * shard_size, shard_size)  # 收窄
        # Special case for per-tensor scales in fused case.
        # 融合情况逐张量缩放值的特殊情况。
        elif needs_scalar_to_array:  # 需要标量转数组
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:  # 其他情况
            ignore_warning = getattr(param, "ignore_warning", False)  # 忽略警告标志
            if not ignore_warning:  # 不忽略
                logger.warning(  # 记录警告
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert (  # 断言形状一致
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)  # 复制权重数据


class RowParallelLinear(LinearBase):  # 行并行线性层
    """Linear layer with row parallelism.
    行并行线性层。

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
    线性层定义为Y = XA + b。A沿第一维度并行化，X沿第二维度并行化：
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        input_size: 矩阵A的第一维度。
        output_size: second dimension of matrix A.
        output_size: 矩阵A的第二维度。
        bias: If true, add bias. Note that bias is not parallelized.
        bias: 如果为True，添加偏置。注意偏置不被并行化。
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        input_is_parallel: 如果为True，假设输入已在GPU间分割，不再分割。
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        skip_bias_add: 添加此选项以启用性能优化，偏置可与其他逐元素操作融合。
                       跳过添加偏置，而是返回它。
        params_dtype: Data type for the parameters.
        params_dtype: 参数的数据类型。
        quant_config: Quantization configure.
        quant_config: 量化配置。
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度
        bias: bool = True,  # 是否添加偏置
        input_is_parallel: bool = True,  # 输入是否已并行
        skip_bias_add: bool = False,  # 是否跳过偏置添加
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        reduce_results: bool = True,  # 是否归约结果
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 层前缀
        tp_rank: Optional[int] = None,  # TP rank
        tp_size: Optional[int] = None,  # TP大小
        use_presharded_weights: bool = False,  # 是否使用预分片权重
        use_dp_attention_reduce: bool = False,  # 是否使用DP注意力归约
    ):
        quant_config = None if _disable_hip_linear_quant else quant_config  # HIP禁用线性量化
        super().__init__(  # 调用父类初始化
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.input_is_parallel = input_is_parallel  # 保存输入并行标志
        self.reduce_results = reduce_results  # 保存归约结果标志
        self.use_dp_attention_reduce = use_dp_attention_reduce  # 保存DP注意力归约标志

        # Divide the weight matrix along the last dimension.
        # 沿最后维度分割权重矩阵。
        if tp_rank is None:  # 如果未指定TP rank
            tp_rank = get_tensor_model_parallel_rank()  # 获取当前TP rank
        if tp_size is None:  # 如果未指定TP大小
            tp_size = get_tensor_model_parallel_world_size()  # 获取当前TP世界大小
        self.tp_rank, self.tp_size = tp_rank, tp_size  # 保存TP rank和大小
        self.input_size_per_partition = divide(input_size, self.tp_size)  # 每个分区的输入大小
        assert self.quant_method is not None  # 断言量化方法存在
        self.use_presharded_weights = use_presharded_weights  # 保存预分片权重标志

        self.quant_method.create_weights(  # 创建权重
            layer=self,
            input_size_per_partition=self.input_size_per_partition,  # 每分区输入大小
            output_partition_sizes=[self.output_size],  # 输出分区大小
            input_size=self.input_size,  # 输入大小
            output_size=self.output_size,  # 输出大小
            params_dtype=self.params_dtype,  # 参数数据类型
            weight_loader=(  # 权重加载器选择
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )

        if bias:  # 如果需要偏置
            self.bias = Parameter(torch.zeros(self.output_size, dtype=params_dtype))  # 创建偏置
            set_weight_attrs(  # 设置权重属性
                self.bias,
                {
                    "output_dim": 0,  # 输出维度索引
                    "weight_loader": self.weight_loader,  # 权重加载器
                },
            )
        else:  # 不需要偏置
            self.register_parameter("bias", None)  # 注册None偏置

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):  # V1权重加载器
        input_dim = getattr(param, "input_dim", None)  # 获取输入维度
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)  # BNB 4位标志

        # Special case for GGUF
        # GGUF特殊情况
        is_gguf_weight = getattr(param, "is_gguf_weight", False)  # 是否为GGUF权重
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)  # 是否为GGUF权重类型
        if is_gguf_weight_type:  # GGUF权重类型
            param.weight_type = loaded_weight.item()  # 设置权重类型

        # Materialize GGUF UninitializedParameter
        # 物化GGUF未初始化参数
        if is_gguf_weight and isinstance(param, UninitializedParameter):  # GGUF未初始化
            weight_shape = list(loaded_weight.shape)  # 获取权重形状
            if input_dim:  # 如果有输入维度
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size  # 按TP大小分割
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)  # 物化参数

        param_data = param.data  # 获取参数数据
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        # bitsandbytes加载特定部分的权重，无需在此收窄
        if (  # 需要收窄的情况
            input_dim is not None
            and not use_bitsandbytes_4bit
            and not self.use_presharded_weights
        ):
            shard_size = param_data.shape[input_dim]  # 分片大小
            start_idx = self.tp_rank * shard_size  # 起始索引

            if _is_cpu:  # CPU平台
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(  # 收窄填充参数
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    input_dim,
                    shard_size,
                )
            else:  # 非CPU平台
                # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                # qwen2_5_VL的mlp等非8对齐的特殊情况的填充
                end_idx = start_idx + shard_size  # 结束索引
                if end_idx > loaded_weight.shape[input_dim]:  # 超出范围
                    loaded_weight = pad_or_narrow_weight(  # 填充或收窄权重
                        loaded_weight, input_dim, start_idx, shard_size
                    )
                else:  # 未超出范围
                    loaded_weight = loaded_weight.narrow(  # 直接收窄
                        input_dim, start_idx, shard_size
                    )

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 从磁盘加载缩放值的特殊情况，通常没有形状（如AutoFP8）。
        if len(loaded_weight.shape) == 0:  # 如果没有形状
            loaded_weight = loaded_weight.reshape(1)  # 重塑为1维

        assert (  # 断言形状一致
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)  # 复制权重数据

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):  # V2权重加载器

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 从磁盘加载缩放值的特殊情况，通常没有形状（如AutoFP8）。
        if len(loaded_weight.shape) == 0:  # 如果没有形状
            assert loaded_weight.numel() == 1  # 断言只有一个元素
            loaded_weight = loaded_weight.reshape(1)  # 重塑为1维

        if isinstance(param, RowvLLMParameter):  # 如果是行vLLM参数
            # This `BasevLLMParameter` is defined in sglang/srt/layers/parameter.py,
            # It supports additional parameters like tp_rank and use_presharded_weights.
            # 此`BasevLLMParameter`定义在sglang/srt/layers/parameter.py中，
            # 支持额外参数如tp_rank和use_presharded_weights。
            param.load_row_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:  # 其他参数类型
            # `params` is defined in `vllm/model_executor/parameter.py`,
            # It does not support additional parameters.
            # However, after QuantizedRL reload, params might still need tp_rank
            # `params`定义在`vllm/model_executor/parameter.py`中，
            # 不支持额外参数。但QuantizedRL重载后，参数可能仍需要tp_rank
            try:  # 尝试带TP参数加载
                param.load_row_parallel_weight(
                    loaded_weight,
                    tp_rank=self.tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            except TypeError:  # 如果参数不支持额外参数
                # Fallback for parameters that don't accept additional args
                # 不接受额外参数的回退方案
                param.load_row_parallel_weight(loaded_weight)  # 不带TP参数加载

    def forward(self, input_, skip_all_reduce=False, forward_batch=None):  # 前向传播
        if self.input_is_parallel:  # 如果输入已并行
            input_parallel = input_  # 直接使用输入
        else:  # 输入未并行
            splitted_input = split_tensor_along_last_dim(  # 沿最后维度分割
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()  # 取当前rank的分片

        # Matrix multiply.
        # 矩阵乘法。
        assert self.quant_method is not None  # 断言量化方法存在
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        # 仅对rank 0将偏置加法融合到GEMM中（确保TP>1时偏置不会被多次添加）
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias  # 计算偏置
        if self.use_dp_attention_reduce:  # 使用DP注意力归约
            symm_ctx = use_symmetric_memory(get_attention_tp_group())  # 使用注意力TP组
        else:  # 不使用DP注意力归约
            symm_ctx = use_symmetric_memory(  # 使用普通TP组
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with symm_ctx:  # 使用对称内存上下文
            output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)  # 应用量化方法

        if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:  # 需要归约且TP>1
            if self.use_dp_attention_reduce:  # DP注意力归约
                output = get_attention_tp_group().all_reduce(output_parallel)  # 注意力TP全归约
            else:  # 普通归约
                quantize_communications = (  # 是否量化通信
                    (
                        not forward_batch.forward_mode.is_decode_or_idle()
                        and get_global_server_args().enable_quant_communications
                    )
                    if forward_batch is not None
                    else False
                )
                if quantize_communications:  # 量化通信
                    output = tensor_model_parallel_quant_all_reduce(output_parallel)  # 量化全归约
                else:  # 非量化通信
                    output = tensor_model_parallel_all_reduce(output_parallel)  # 普通全归约
        else:  # 不需要归约
            output = output_parallel  # 直接使用并行输出

        output_bias = self.bias if self.skip_bias_add else None  # 获取输出偏置

        return output, output_bias  # 返回输出和偏置

    def extra_repr(self) -> str:  # 额外表示方法
        s = f"input_features={self.input_size_per_partition}"  # 输入特征数（每分区）
        s += f", output_features={self.output_size}"  # 输出特征数
        s += f", bias={self.bias is not None}"  # 是否有偏置
        s += f", tp_size={self.tp_size}"  # TP大小
        s += f", reduce_results={self.reduce_results}"  # 是否归约结果
        return s  # 返回字符串


class MergedColumnParallelRepeatedLinear(LinearBase):  # 合并列并行重复线性层
    """Merged column parallel linear and repeated linear layer.
    合并列并行线性层和重复线性层。

    TODO: quantization is not supported yet.
    TODO: 尚不支持量化。
    Args:
        input_size: input dimension of the linear layer.
        input_size: 线性层的输入维度。
        column_output_sizes: output dimension of the column linear layers.
        column_output_sizes: 列线性层的输出维度。
        repeated_output_sizes: output dimension of the repeated linear layers.
        repeated_output_sizes: 重复线性层的输出维度。
        skip_bias_add: If true, skip adding bias but instead return it.
        skip_bias_add: 如果为True，跳过添加偏置，而是返回它。
        params_dtype: Data type for the parameters.
        params_dtype: 参数的数据类型。
        quant_config: Quantization configure.
        quant_config: 量化配置。
    """

    def __init__(  # 初始化方法
        self,
        input_size: int,  # 输入维度
        column_output_sizes: List[int],  # 列线性层输出大小列表
        repeated_output_sizes: List[int],  # 重复线性层输出大小列表
        skip_bias_add: bool = False,  # 是否跳过偏置添加
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 层前缀
    ):
        output_size = sum(column_output_sizes) + sum(repeated_output_sizes)  # 总输出大小
        super().__init__(  # 调用父类初始化
            input_size=input_size,
            output_size=output_size,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.num_column_parallel = len(column_output_sizes)  # 列并行数量
        self.tp_rank = get_tensor_model_parallel_rank()  # 获取TP rank
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小

        self.output_partition_sizes = [  # 输出分区大小
            divide(x, self.tp_size) for x in column_output_sizes  # 列并行分区
        ] + repeated_output_sizes  # 加上重复分区
        self.quant_method.create_weights(  # 创建权重
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            skip_block_quant_check=True,  # 跳过块量化检查
            weight_loader=self.weight_loader,
        )

        self.prefix = prefix  # 保存前缀

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # 前向传播
        return self.quant_method.apply(self, input_)  # 应用量化方法

    def weight_loader(  # 权重加载器
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ) -> torch.Tensor:
        output_dim = param.output_dim  # 获取输出维度
        shard_offset = sum(self.output_partition_sizes[:loaded_shard_id])  # 分片偏移
        shard_size = self.output_partition_sizes[loaded_shard_id]  # 分片大小
        param_data = param.data.narrow(output_dim, shard_offset, shard_size)  # 收窄参数

        if loaded_shard_id < self.num_column_parallel:  # 列并行分片
            start_idx = self.tp_rank * shard_size  # 起始索引
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)  # 收窄权重

        param_data.copy_(loaded_weight)  # 复制权重数据


class ColumnParallelBatchedLinear(nn.Module):  # 列并行批处理线性层
    """Column parallel batched linear layer.
    列并行批处理线性层。

    TODO: quantization is not supported yet.
    TODO: 尚不支持量化。
    Args:
        batch: batch dimension of the linear layer.
        batch: 线性层的批次维度。
        input_size: input dimension of the linear layer.
        input_size: 线性层的输入维度。
        output_size: output dimension of the linear layer.
        output_size: 线性层的输出维度。
        dtype: Data type for the parameters.
        dtype: 参数的数据类型。
    """

    def __init__(  # 初始化方法
        self, batch: int, input_size: int, output_size: int, dtype: torch.dtype
    ):
        super().__init__()  # 调用父类初始化
        self.tp_rank = get_tensor_model_parallel_rank()  # 获取TP rank
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.weight = nn.Parameter(  # 创建权重参数
            torch.empty(batch, output_size // self.tp_size, input_size, dtype=dtype),
            requires_grad=False,  # 不需要梯度
        )
        setattr(self.weight, "weight_loader", self.weight_loader)  # 设置权重加载器

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # 前向传播
        return torch.bmm(input, self.weight.transpose(-1, -2))  # 批量矩阵乘法

    def weight_loader(  # 权重加载器
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ) -> torch.Tensor:
        shard_size = self.weight.shape[-2]  # 分片大小
        start_idx = self.tp_rank * shard_size  # 起始索引
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)  # 收窄权重
        param.data[loaded_shard_id].copy_(loaded_weight)  # 复制到参数
