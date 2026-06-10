# 模型并行通用工具模块，提供张量并行的分片、列并行、行并行等工具函数和类
# SPDX-License-Identifier: Apache-2.0
"""
Common utilities for torch model parallelism.
通用工具函数，用于PyTorch模型并行。
"""

from typing import Optional, Sequence  # 导入类型提示：可选类型和序列类型

import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed.device_mesh import DeviceMesh  # 导入设备网格类，用于分布式设备管理

try:  # 尝试导入分布式张量模块
    import torch.distributed.tensor as dt  # 导入PyTorch分布式张量模块
except ImportError:  # 如果导入失败（旧版本PyTorch）
    # torch 2.4 or older
    # torch 2.4或更早版本
    import torch.distributed._tensor as dt  # 导入旧版分布式张量模块

from torch.distributed.tensor.parallel import (  # 导入张量并行相关工具
    ColwiseParallel,  # 列并行策略
    RowwiseParallel,  # 行并行策略
    parallelize_module,  # 模块并行化函数
)


def _shard_tensor(  # 定义分片张量的私有函数
    full_tensor: torch.Tensor,  # 完整张量参数
    device_mesh: DeviceMesh,  # 设备网格参数
    placements: Sequence[dt.Shard],  # 分片放置策略序列
) -> "dt.DTensor":  # 返回分布式张量
    """
    Locally shards a full tensor based on indicated sharding arrangement, and
    returns a DTensor containing the local shard.
    根据指定的分片策略在本地对完整张量进行分片，并返回包含本地分片的分布式张量。

    .. warning:: This is a private API that is subject to change. It skips the
        communication otherwise required by `distribute_tensor`. It is only
        applicable to cases where all ranks have the same `full_tensor`. For
        example, in distributed inference all ranks load from the same
        checkpoint. This API will not check for data equality between ranks, it
        is thus user's responsibility to ensure the `full_tensor` is the same
        across ranks.
    警告：这是一个可能变更的私有API。它跳过了distribute_tensor所需的通信。
        仅适用于所有rank拥有相同full_tensor的情况，例如分布式推理中所有rank从同一检查点加载。
        此API不会检查rank间数据一致性，用户需自行确保full_tensor在各rank间相同。

    Args:
        full_tensor (torch.Tensor): the full tensor to be sharded.
        full_tensor (torch.Tensor): 待分片的完整张量。
        device_mesh (:class:`DeviceMesh`): DeviceMesh to place the
            DTensor.  Must have same dimension as the number of placements.
        device_mesh (:class:`DeviceMesh`): 用于放置分布式张量的设备网格，维度必须与放置策略数相同。
        placements (Sequence[:class:`Shard`]): the placements that
            describes how to place the local tensor on DeviceMesh.
        placements (Sequence[:class:`Shard`]): 描述如何在设备网格上放置本地张量的策略。

    Returns:
        A :class:`DTensor` object with the shard as its local tensor.
        返回一个以分片作为本地张量的分布式张量对象。

    Examples:
        >>> # xdoctest: +SKIP("need world_size and rank")
        >>> device_mesh = dist.init_device_mesh("cuda", (world_size,))
        >>> full_tensor = torch.arange(world_size, device=f"cuda:{rank}")
        >>> dtensor = _shard_tensor(full_tensor, device_mesh, [Shard(1)])
    """
    shape, offset = dt._utils.compute_local_shape_and_global_offset(  # 计算本地形状和全局偏移
        full_tensor.shape, device_mesh, placements  # 使用完整张量形状、设备网格和放置策略
    )
    slices = [  # 构建切片列表
        slice(cur_offset, cur_offset + cur_shape)  # 为每个维度创建切片对象
        for cur_shape, cur_offset in zip(shape, offset)  # 遍历形状和偏移
    ]
    local_tensor = full_tensor[slices]  # 从完整张量中提取本地分片
    return dt.DTensor.from_local(local_tensor, device_mesh, placements)  # 从本地张量创建分布式张量并返回


class ColwiseParallelSharded(ColwiseParallel):  # 定义已分片的列并行类，继承自ColwiseParallel
    """
    A version of ColwiseParallel where the local weight has been already
    sharded.  This is used for the fused wqkv case, where during loading, we
    already sharded wq, wk, wv before fusing them.
    ColwiseParallel的一个变体，其中本地权重已经分片。用于融合wqkv的场景，
    即在加载时，wq、wk、wv在融合之前已经分片。
    """

    # Override the _partition_linear_fn in ColwiseParallel
    # 重写ColwiseParallel中的_partition_linear_fn方法
    def _partition_linear_fn(self, name, module, device_mesh):  # 定义线性层分区函数
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        # 列并行将权重/偏置分片为Shard(0)，权重为Shard(0)
        # 意味着列并行中Linear为 input * weight^T + bias，权重会变为Shard(1)
        for name, param in module.named_parameters():  # 遍历模块的所有命名参数
            dtensor = dt.DTensor.from_local(param, device_mesh, [dt.Shard(0)])  # 从本地参数创建列并行分布式张量
            dist_param = torch.nn.Parameter(dtensor, requires_grad=False)  # 创建不需要梯度的分布式参数
            module.register_parameter(name, dist_param)  # 注册分布式参数到模块


class RowwiseParallelMaybeWait(RowwiseParallel):  # 定义带等待的行并行类，继承自RowwiseParallel
    """
    A version of RowwiseParallel that waits for the output (establish dependency
    between comm stream and compute stream in CUDA sense) before going into the
    next op. This is needed to workaround the current interaction between
    AsyncCollectiveTensor and multi-platform ops, such as `RMSNorm`.
    RowwiseParallel的一个变体，在进入下一个操作前等待输出完成（在CUDA意义上建立
    通信流和计算流之间的依赖）。这是为了解决AsyncCollectiveTensor与
    多平台操作（如RMSNorm）之间的交互问题。
    """

    def _partition_linear_fn(self, name, module, device_mesh):  # 定义线性层分区函数
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        # 行并行将权重分片为Shard(1)，偏置为Replicate()，权重为Shard(1)
        # 意味着行并行中nn.Linear为 input * weight^T + bias，权重会变为Shard(0)
        module.register_parameter(  # 注册权重参数
            "weight",  # 参数名称为weight
            nn.Parameter(_shard_tensor(module.weight, device_mesh, [dt.Shard(1)])),  # 行并行分片权重
        )
        if getattr(module, "bias", None) is not None:  # 如果模块有偏置
            # The Linear module has bias
            # 线性模块有偏置
            module.register_parameter(  # 注册偏置参数
                "bias",  # 参数名称为bias
                nn.Parameter(  # 创建参数
                    dt.distribute_tensor(module.bias, device_mesh, [dt.Replicate()])  # 偏置使用复制策略分布
                ),
            )

    @staticmethod  # 静态方法
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):  # 定义输出准备函数
        outputs = super(  # 调用父类的输出准备函数
            RowwiseParallelMaybeWait, RowwiseParallelMaybeWait  # 使用当前类名引用
        )._prepare_output_fn(  # 调用父类方法
            output_layouts, use_local_output, mod, outputs, device_mesh  # 传递所有参数
        )
        return torch.distributed._functional_collectives.wait_tensor(outputs)  # 等待张量完成并返回


def tensor_parallel(  # 定义张量并行化函数
    module: torch.nn.Module,  # 待并行化的模块
    device_mesh: Optional[DeviceMesh] = None,  # 设备网格，默认为None
):  # 无返回值注解
    """
    Tensor parallelize the model across the given device mesh.
    在给定的设备网格上对模型进行张量并行化。
    Args:
        module (`torch.nn.Module`):
            The module to tensor parallelize.
            待张量并行化的模块。
        device_mesh (`torch.distributed.DeviceMesh`):
            The device mesh to use for tensor parallelism.
            用于张量并行的设备网格。
    """

    # Tensor parallelize a nn.Module based on the `_tp_plan` attribute of the module.
    # No op if `_tp_plan` attribute does not exist under the module.
    # This is a helper function to be used with `model.apply` to recursively
    # parallelize a model.
    # 根据模块的`_tp_plan`属性对nn.Module进行张量并行化。
    # 如果模块不存在`_tp_plan`属性，则不执行任何操作。
    # 这是一个辅助函数，用于配合`model.apply`递归地并行化模型。
    def tplize(mod: torch.nn.Module) -> None:  # 定义模块并行化内部函数
        tp_plan = getattr(mod, "_tp_plan", None)  # 获取模块的张量并行计划
        if tp_plan is None:  # 如果没有并行计划
            return  # 直接返回
        for child_name, tp_style in tp_plan.items():  # 遍历并行计划中的每个子模块
            submod = mod.get_submodule(child_name)  # 获取子模块
            if tp_style == "Colwise":  # 如果是列并行策略
                parallelize_module(submod, device_mesh, ColwiseParallel())  # 使用列并行策略并行化
            elif tp_style == "Rowwise":  # 如果是行并行策略
                parallelize_module(submod, device_mesh, RowwiseParallelMaybeWait())  # 使用带等待的行并行策略并行化
            elif tp_style == "Colwise_Sharded":  # 如果是已分片的列并行策略
                parallelize_module(submod, device_mesh, ColwiseParallelSharded())  # 使用已分片列并行策略并行化
            else:  # 未知策略
                raise ValueError(f"Unknown TP style {tp_style}")  # 抛出值错误异常

    # `apply` is a native method of `nn.Module` that recursively applies a
    # function to every submodule.
    # `apply`是`nn.Module`的原生方法，递归地将函数应用到每个子模块。
    module.apply(tplize)  # 对模块递归应用并行化函数
