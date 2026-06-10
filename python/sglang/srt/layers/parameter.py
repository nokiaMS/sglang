# 参数层实现：定义了vLLM线性层的各种参数类型，包括基础参数、
# 列并行参数、行并行参数、模型权重参数、量化缩放参数、打包参数等，
# 以及参数布局置换和分片索引调整的工具函数
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/parameter.py"""

import logging
from fractions import Fraction
from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.utils import pad_or_narrow_weight
from sglang.srt.utils import is_cpu

__all__ = [
    "BasevLLMParameter",
    "PackedvLLMParameter",
    "PerTensorScaleParameter",
    "ModelWeightParameter",
    "ChannelQuantScaleParameter",
    "GroupQuantScaleParameter",
    "BlockQuantScaleParameter",
    "PackedColumnParameter",
    "RowvLLMParameter",
]

logger = logging.getLogger(__name__)

_is_cpu = is_cpu()


# 计算数据类型的精度等级，用于判断是否允许类型转换
def _dtype_rank(dtype: torch.dtype) -> Optional[int]:
    if dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    ):
        return 0
    if dtype in (torch.float16, torch.bfloat16):
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.float64:
        return 3
    return None


# 带检查的权重复制：禁止降精度转换
def copy_with_check(target: torch.Tensor, loaded_weight: torch.Tensor):
    """
    Copy `loaded_weight` into `target` while forbidding downcasts.
    bf16/fp16 share the same rank, and all fp8 variants share the same rank.
    """

    assert (
        target.shape == loaded_weight.shape
    ), f"{target.shape=}, {loaded_weight.shape=}"

    if target.dtype == loaded_weight.dtype:
        target.copy_(loaded_weight)
        return

    # 检查精度等级，禁止降精度转换
    target_rank = _dtype_rank(target.dtype)
    loaded_rank = _dtype_rank(loaded_weight.dtype)

    if target_rank is None or loaded_rank is None:
        raise ValueError(
            f"Unsupported copy between dtypes: {target.dtype=}, {loaded_weight.dtype=}"
        )
    if target_rank < loaded_rank and not envs.SGLANG_QUANT_ALLOW_DOWNCASTING.get():
        raise ValueError(
            f"Downcasting not allowed: {target.dtype=}, {loaded_weight.dtype=}"
        )
    if loaded_rank == torch.float8_e8m0fnu:
        assert target_rank in {torch.float8_e8m0fnu, torch.float32}

    target.copy_(loaded_weight)


# vLLM线性层的基础参数类，扩展了torch.nn.parameter
class BasevLLMParameter(Parameter):
    """
    Base parameter for vLLM linear layers. Extends the torch.nn.parameter
    by taking in a linear weight loader. Will copy the loaded weight
    into the parameter when the provided weight loader is called.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):

        return super().__new__(cls, data=data, requires_grad=False)

    # 初始化基础参数
    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        """
        Initialize the BasevLLMParameter

        :param data: torch tensor with the parameter data
        :param weight_loader: weight loader callable

        :returns: a torch.nn.parameter
        """

        self._weight_loader = weight_loader

    @property
    def weight_loader(self):
        return self._weight_loader

    # 断言并加载权重：检查形状一致后复制
    def _assert_and_load(self, loaded_weight: torch.Tensor):
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    # 加载列并行权重
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    # 加载行并行权重
    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    # 加载合并列权重
    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    # 加载QKV权重
    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)


# 列并行参数私有类：定义列并行线性层的权重加载功能
class _ColumnvLLMParameter(BasevLLMParameter):
    """
    Private class defining weight loading functionality
    (load_merged_column_weight, load_qkv_weight)
    for parameters being loaded into linear layers with column
    parallelism. This includes QKV and MLP layers which are
    not already fused on disk. Requires an output dimension
    to be defined. Called within the weight loader of
    each of the column parallel linear layers.
    """

    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self):
        return self._output_dim

    # 加载列并行权重：按TP秩分片
    def load_column_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.output_dim]

            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            if _is_cpu:
                # CPU路径：使用窄化函数处理填充参数
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    self.data,
                    loaded_weight,
                    0,  # param_data_start
                    tp_rank * shard_size,
                    self.output_dim,
                    shard_size,
                )
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            else:
                # GPU路径：使用narrow获取分片
                loaded_weight = loaded_weight.narrow(
                    self.output_dim, tp_rank * shard_size, shard_size
                )

        copy_with_check(self.data, loaded_weight)

    # 加载合并列权重：用于MLP融合列线性层
    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        tp_rank = kwargs.get("tp_rank")
        use_presharded_weights = kwargs.get("use_presharded_weights")
        # 如果参数在输出维度上被打包，调整分片索引
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data

        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)

        from sglang.srt.model_loader.weight_utils import (
            narrow_padded_param_and_loaded_weight,
        )

        if _is_cpu:
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,  # param_data_start
                tp_rank * shard_size,
                self.output_dim,
                shard_size,
                not use_presharded_weights,
            )
        else:
            if not use_presharded_weights:
                # 针对qwen2_5_VL的mlp等非8对齐特殊情况的填充
                # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                start_idx = tp_rank * shard_size
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[self.output_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, self.output_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        self.output_dim, start_idx, shard_size
                    )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    # 加载QKV权重：用于注意力层的QKV融合线性层
    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
        **kwargs,
    ):

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")

        # 如果参数在输出维度上被打包，调整分片索引
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        # 计算分片ID：Q按tp_rank，K/V按tp_rank//num_heads
        shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)

        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,  # param_data_start
                shard_id * shard_size,
                self.output_dim,
                shard_size,
                not use_presharded_weights,
            )
        else:
            if not use_presharded_weights:
                loaded_weight = loaded_weight.narrow(
                    self.output_dim, shard_id * shard_size, shard_size
                )

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=}, {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)


# 行并行参数类：定义行并行线性层的权重加载功能
class RowvLLMParameter(BasevLLMParameter):
    """
    Parameter class defining weight_loading functionality
    (load_row_parallel_weight) for parameters being loaded
    into linear layers with row parallel functionality.
    Requires an input_dim to be defined.
    """

    def __init__(self, input_dim: int, **kwargs):
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self):
        return self._input_dim

    # 加载行并行权重：按TP秩在输入维度上分片
    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.input_dim]

            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            if _is_cpu:
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    self.data,
                    loaded_weight,
                    0,  # param_data_start
                    tp_rank * shard_size,
                    self.input_dim,
                    shard_size,
                )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)

                return
            else:
                # 针对qwen2_5_VL的mlp等非8对齐特殊情况的填充
                # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                start_idx = tp_rank * shard_size
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[self.input_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, self.input_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        self.input_dim, start_idx, shard_size
                    )

        # 处理标量权重
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


# 模型权重参数类：同时支持列并行和行并行
class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for linear layer weights. Uses both column and
    row parallelism.
    """

    pass


# 分组量化缩放参数类：用于分组量化权重的缩放因子
class GroupQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    grouped quantization. Uses both column and row parallelism.
    """

    pass


# 通道量化缩放参数类：用于通道级量化权重的缩放因子
class ChannelQuantScaleParameter(_ColumnvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    channel-wise quantization. Equivalent to _ColumnvLLMParameter.
    """

    pass


# 块量化缩放参数类：用于块级量化权重的缩放因子
class BlockQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    block-wise quantization. Uses both column and row parallelism.
    """

    pass


# 逐张量缩放参数类：用于融合线性层中逐张量量化的缩放因子
class PerTensorScaleParameter(BasevLLMParameter):
    """
    Parameter class for scales where the number of scales is
    equivalent to the number of logical matrices in fused linear
    layers (e.g. for QKV, there are 3 scales loaded from disk).
    This is relevant to weights with per-tensor quantization.
    Adds functionality to map the scalers to a shard during
    weight loading.

    Note: additional parameter manipulation may be handled
    for each quantization config specifically, within
    process_weights_after_loading
    """

    def __init__(self, **kwargs):
        # QKV分片ID映射
        self.qkv_idxs = {"q": 0, "k": 1, "v": 2}
        super().__init__(**kwargs)

    # 将分片ID转换为整数
    def _shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id

        # 如果不是整数，假设是QKV的分片ID
        # 映射为整数并返回
        # if not int, assume shard_id for qkv
        # map to int and return
        assert isinstance(shard_id, str)
        assert shard_id in self.qkv_idxs
        return self.qkv_idxs[shard_id]

    # 对于行并行层，不需要分片，直接加载权重
    # For row parallel layers, no sharding needed
    # load weight into parameter as is
    def load_row_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    # 加载合并列权重：按分片ID加载
    def load_merged_column_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    # 加载QKV权重：按分片ID加载
    def load_qkv_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    # 加载列并行权重：不需要分片
    def load_column_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    # 按分片ID加载权重到参数数据
    def _load_into_shard_id(
        self, loaded_weight: torch.Tensor, shard_id: Union[str, int], **kwargs
    ):
        """
        Slice the parameter data based on the shard id for
        loading.
        """

        param_data = self.data
        shard_id = self._shard_id_as_int(shard_id)

        # AutoFP8缩放没有形状
        # compressed-tensors缩放有形状
        # AutoFP8 scales do not have a shape
        # compressed-tensors scales do have a shape
        if len(loaded_weight.shape) != 0:
            assert loaded_weight.shape[0] == 1
            loaded_weight = loaded_weight[0]

        # 按分片ID切片参数数据并复制
        param_data = param_data[shard_id]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# 打包列参数类：用于在磁盘上打包且仅支持列并行的参数
class PackedColumnParameter(_ColumnvLLMParameter):
    """
    Parameter for model parameters which are packed on disk
    and support column parallelism only. See PackedvLLMParameter
    for more details on the packed properties.
    """

    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    # 调整分片索引以考虑打包因子和marlin瓦片大小
    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


# 打包vLLM参数类：用于在磁盘上打包的模型权重（如GPTQ Marlin权重）
class PackedvLLMParameter(ModelWeightParameter):
    """
    Parameter for model weights which are packed on disk.
    Example: GPTQ Marlin weights are int4 or int8, packed into int32.
    Extends the ModelWeightParameter to take in the
    packed factor, the packed dimension, and optionally, marlin
    tile size for marlin kernels. Adjusts the shard_size and
    shard_offset for fused linear layers model weight loading
    by accounting for packing and optionally, marlin tile size.
    """

    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    # 调整分片索引以考虑打包因子和marlin瓦片大小
    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


# 置换参数布局：将参数的输入/输出维度置换到指定位置
def permute_param_layout_(
    param: BasevLLMParameter, input_dim: int, output_dim: int, **kwargs
) -> BasevLLMParameter:
    """
    Permute a parameter's layout to the specified input and output dimensions,
    useful for forcing the parameter into a known layout, for example, if I need
    a packed (quantized) weight matrix to be in the layout
        {input_dim = 0, output_dim = 1, packed_dim = 0}
    then I can call:
        permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
    to ensure x is in the correct layout (permuting it to the correct layout if
    required, asserting if it cannot get it to the correct layout)
    """

    curr_input_dim = getattr(param, "input_dim", None)
    curr_output_dim = getattr(param, "output_dim", None)

    if curr_input_dim is None or curr_output_dim is None:
        assert param.data.dim() == 2, (
            "permute_param_layout_ only supports 2D parameters when either "
            "input_dim or output_dim is not set"
        )

    # 如果某个维度未设置，设置为另一个维度的对侧
    # 我们只能在上面断言参数是2D的情况下才能这样做
    # if one of the dimensions is not set, set it to the opposite of the other
    #  we can only do this since we asserted the parameter is 2D above
    if curr_input_dim is None:
        assert curr_output_dim is not None, "either input or output dim must be set"
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        assert curr_input_dim is not None, "either input or output dim must be set"
        curr_output_dim = (curr_input_dim + 1) % 2

    # 从当前布局创建置换，使self.input_dim在input_dim位置，
    # self.output_dim在output_dim位置，保留其他维度
    # create permutation from the current layout to the layout with
    # self.input_dim at input_dim and self.output_dim at output_dim preserving
    # other dimensions
    perm = [
        i for i in range(param.data.dim()) if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    # 检查packed_dim是否与置换兼容
    if "packed_dim" in kwargs:
        assert (
            hasattr(param, "packed_dim")
            and param.packed_dim == perm[kwargs["packed_dim"]]
        ), "permute_param_layout_ currently doesn't support repacking"

    # 应用置换并更新维度属性
    param.data = param.data.permute(*perm)
    if hasattr(param, "_input_dim"):
        param._input_dim = input_dim
    if hasattr(param, "_output_dim"):
        param._output_dim = output_dim
    if "packed_dim" in kwargs and hasattr(param, "_packed_dim"):
        param._packed_dim = kwargs["packed_dim"]

    return param


# 调整Marlin瓦片大小的分片索引
def _adjust_shard_indexes_for_marlin(shard_size, shard_offset, marlin_tile_size):
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


# 调整打包的分片索引：考虑打包因子和可选的marlin瓦片大小
def _adjust_shard_indexes_for_packing(
    shard_size, shard_offset, packed_factor, marlin_tile_size
):
    shard_size = shard_size // packed_factor
    shard_offset = shard_offset // packed_factor
    if marlin_tile_size is not None:
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size,
        )
    return shard_size, shard_offset
