# 通用层工具模块 - 提供层权重处理、参数绑定、内存连续性检查等通用功能
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging  # 导入日志模块
import re  # 导入正则表达式模块

import torch  # 导入PyTorch
from torch.nn.parameter import Parameter  # 导入参数类

logger = logging.getLogger(__name__)  # 创建日志记录器


def get_layer_id(weight_name):  # 从权重名称中提取层ID
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    # 示例权重名称：model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)  # 用正则匹配层编号
    if match:  # 如果匹配成功
        return int(match.group(1))  # 返回层ID（整数）
    return None  # 未匹配则返回None


def pad_or_narrow_weight(  # 填充或截断权重张量以适配分片大小
    loaded_weight: torch.Tensor, input_dim: int, start_idx: int, shard_size: int
) -> torch.Tensor:
    # Padding with zeros for special case such as qwen2_5_VL's mlp which is not 8-aligned
    # 对特殊情况（如qwen2_5_VL的MLP不是8对齐的）用零填充
    valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)  # 计算有效大小，不小于0

    if valid_size > 0:  # 如果有效大小大于0
        loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)  # 截取有效部分
        pad_shape = list(loaded_weight.shape)  # 复制原始形状
        pad_shape[input_dim] = shard_size - valid_size  # 计算需要填充的大小
        pad = torch.zeros(  # 创建零填充张量
            pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
        )
        return torch.cat([loaded_slice, pad], dim=input_dim)  # 拼接有效部分和填充部分

    # All padding
    # 全部填充
    pad_shape = list(loaded_weight.shape)  # 复制原始形状
    pad_shape[input_dim] = shard_size  # 填充大小等于分片大小
    return torch.zeros(  # 返回全零张量
        pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
    )


def is_strict_contiguous(x: torch.Tensor) -> bool:  # 严格检查张量是否连续
    expected_stride = 1  # 期望的步长从1开始
    for size, stride in zip(reversed(x.shape), reversed(x.stride())):  # 从最内层维度开始遍历
        if stride != expected_stride:  # 如果步长不符合期望
            return False  # 不连续
        expected_stride *= size  # 更新期望步长
    return True  # 所有维度步长都符合，连续


def strict_contiguous(x: torch.Tensor) -> torch.Tensor:  # 确保张量严格连续，不连续则克隆
    if is_strict_contiguous(x):  # 如果已经严格连续
        return x  # 直接返回
    return x.clone(memory_format=torch.contiguous_format)  # 否则克隆为连续格式


def copy_or_rebind_param(  # 复制或重新绑定参数到模块
    module: torch.nn.Module, name: str, new_value: torch.Tensor
) -> None:
    """Keep parameter identities stable for CUDA graph reuse and hot reload."""
    # 保持参数身份稳定，以便CUDA图复用和热重载。
    new_value = new_value.detach()  # 分离梯度
    param = getattr(module, name, None)  # 获取当前参数
    if isinstance(param, Parameter):  # 如果是Parameter类型
        if param.data.shape == new_value.shape and param.data.dtype == new_value.dtype:  # 形状和类型匹配
            param.data.copy_(new_value)  # 原地复制数据
        else:  # 形状或类型不匹配
            param.data = new_value  # 替换数据
        param.requires_grad_(False)  # 禁用梯度
    else:  # 不是Parameter类型
        setattr(module, name, Parameter(new_value, requires_grad=False))  # 设置为新的Parameter


def alias_or_bind_derived_param(  # 将派生参数别名绑定或独立绑定到模块
    module: torch.nn.Module,
    source_name: str,  # 源参数名
    derived_name: str,  # 派生参数名
    derived_value: torch.Tensor,  # 派生参数值
) -> None:
    """Bind a post-processed (derived) tensor to a derived attribute name.
    将后处理（派生）张量绑定到派生属性名。

    When `derived_value` is broadcastable to the source Parameter's shape (and
    dtype matches), write it broadcast-filled into the source's storage in
    place and register `derived_name` as an alias of the source Parameter. The
    two attribute names then share one underlying buffer, so:
    当`derived_value`可以广播到源Parameter的形状（且类型匹配）时，
    将其广播填充写入源的存储中，并将`derived_name`注册为源Parameter的别名。
    两个属性名共享同一个底层缓冲区，因此：
      - apply() can read via `derived_name`
      - apply()可以通过`derived_name`读取
      - update_weights_from_disk can keep refilling `source_name` (the loader
        re-runs process_weights_after_loading which re-derives in place)
      - update_weights_from_disk可以持续填充`source_name`（加载器重新运行
        process_weights_after_loading，原地重新派生）
      - peak GPU memory is the source size, not source + derived.
      - 峰值GPU内存是源大小，而不是源+派生。

    When the shapes are not broadcast-compatible, fall back to allocating a
    separate Parameter under `derived_name` via copy_or_rebind_param.
    当形状不兼容广播时，回退到通过copy_or_rebind_param在`derived_name`下
    分配独立的Parameter。
    """
    derived_value = derived_value.detach()  # 分离派生值的梯度
    source = getattr(module, source_name, None)  # 获取源参数
    if isinstance(source, Parameter) and source.data.dtype == derived_value.dtype:  # 源是Parameter且类型匹配
        try:
            broadcast = torch.broadcast_to(derived_value, source.data.shape)  # 尝试广播到源形状
        except RuntimeError:  # 广播失败
            broadcast = None  # 设为None
        if broadcast is not None:  # 广播成功
            source.data.copy_(broadcast)  # 将广播值写入源数据
            source.requires_grad_(False)  # 禁用梯度
            setattr(module, derived_name, source)  # 将派生名设为源的别名
            return  # 返回
    copy_or_rebind_param(module, derived_name, derived_value)  # 广播失败，独立绑定派生参数


class PPMissingLayer(torch.nn.Identity):  # 流水线并行缺失层占位符
    # Adapted from
    # 适配自
    # https://github.com/vllm-project/vllm/blob/18ed3132d2bfe1df9a74729457b69243955221e8/vllm/model_executor/models/utils.py#L468C1-L486C1
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    流水线并行模型中缺失层的占位符层。
    """

    def __init__(self, *args, **kwargs):  # 初始化
        super().__init__()  # 调用父类初始化
        self.return_tuple = kwargs.get("return_tuple", False)  # 是否返回元组

    def forward(self, *args, **kwargs):  # 前向传播
        """
        Return the first arg from args or the first value from kwargs.
        返回args的第一个参数或kwargs的第一个值。

        Wraps the input in a tuple if `self.return_tuple` is True.
        如果`self.return_tuple`为True，则将输入包装在元组中。
        """
        input = args[0] if args else next(iter(kwargs.values()))  # 获取输入
        return (input,) if self.return_tuple else input  # 根据配置返回元组或原始值
