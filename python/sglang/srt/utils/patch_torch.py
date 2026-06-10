# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 本文件对 PyTorch 进行猴子补丁（monkey patch），修复以下问题：
# 1. 修复 torch.multiprocessing 中张量序列化/反序列化时设备索引不正确的问题，
#    通过将设备索引替换为 UUID 来确保跨进程传输张量时设备映射正确。
# 2. 修复 torch.compile 中 auto_functionalized 操作不可缓存的问题。
# 3. 提供条件注册自定义算子 fake 实现的装饰器工具。

from typing import Callable, Union

import torch
from torch.multiprocessing import reductions

from sglang.srt.utils.common import is_musa, is_npu, torch_release

# 检测当前是否运行在 NPU（华为昇腾）或 MUSA（摩尔线程）设备上
_is_npu = is_npu()
_is_musa = is_musa()

if _is_npu:
    from torch_npu.multiprocessing import reductions as npu_reductions

    # NPU 上重建张量时的修改版本，将设备索引替换为 SGLang 张量并行秩
    def _rebuild_npu_tensor_modified(*args):
        args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, npu_verl_to_sglang)
        return npu_reductions._rebuild_npu_tensor_original(*args)

    # 将 verl 的设备索引映射到 SGLang 的张量并行秩
    def npu_verl_to_sglang(device: int):
        assert (
            SGLANG_TP_RANK is not None
        ), "SGLANG_TP_RANK is not registered. Please call register_sgl_tp_rank() first."
        return SGLANG_TP_RANK


# 全局变量，存储当前 SGLang 进程的张量并行秩
SGLANG_TP_RANK = None


def monkey_patch_torch_reductions():
    """Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed"""
    # 对 torch.multiprocessing.reductions 打补丁，修复张量跨进程传输时的设备索引问题

    if not _is_npu:
        # CUDA 设备路径：保存原始函数并替换为修改版本
        if hasattr(reductions, "_reduce_tensor_original"):
            return
        # 保存原始的 reduce_tensor 和 rebuild_cuda_tensor 函数
        reductions._reduce_tensor_original = reductions.reduce_tensor
        reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

        # 替换为修改后的版本，在序列化时将设备索引转为 UUID
        reductions.reduce_tensor = _reduce_tensor_modified
        # 替换为修改后的版本，在反序列化时将 UUID 转回设备索引
        reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified
        # 重新初始化 reductions 以使补丁生效
        reductions.init_reductions()
    else:
        # FIXME: This is a temp patch for npu as HDK does not support device uuid for now
        # NPU 设备路径：HDK 暂不支持设备 UUID，因此使用张量并行秩替代
        if hasattr(npu_reductions, "_rebuild_npu_tensor_original"):
            return

        npu_reductions._rebuild_npu_tensor_original = npu_reductions.rebuild_npu_tensor
        npu_reductions.rebuild_npu_tensor = _rebuild_npu_tensor_modified


# The signature has not been changed for years, and we will not need this when the next version is released,
# so it looks safe to use a constant.
# reduce_tensor 参数元组中设备索引所在的位置（第 7 个参数，索引为 6）
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def register_sgl_tp_rank(rank: int):
    """注册当前进程的 SGLang 张量并行秩，供 NPU 设备映射使用"""
    global SGLANG_TP_RANK
    SGLANG_TP_RANK = rank


def _reduce_tensor_modified(*args, **kwargs):
    """修改版的 reduce_tensor，在序列化张量时将设备索引替换为 UUID"""
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    # 将输出参数中的设备索引替换为设备的 UUID 字符串
    output_args = _modify_tuple(
        output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
    )
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    """修改版的 rebuild_cuda_tensor，在反序列化张量时将 UUID 还原为设备索引"""
    # 将参数中的 UUID 替换回设备索引
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def _device_to_uuid(device: int) -> str:
    """将 CUDA 设备索引转换为设备 UUID 字符串"""
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device_maybe_uuid: Union[int, str]) -> int:
    """将设备 UUID 字符串还原为 CUDA 设备索引；如果已经是整数则直接返回"""
    if isinstance(device_maybe_uuid, int):
        # 如果已经是整数索引，无需转换
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        # 遍历所有 CUDA 设备，找到 UUID 匹配的设备索引
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier: Callable):
    """对元组中指定位置的元素应用 modifier 函数，返回新元组"""
    return *t[:index], modifier(t[index]), *t[index + 1 :]


def monkey_patch_torch_compile():
    """对 torch.compile 打补丁，将 auto_functionalized 操作标记为可缓存，以提升编译性能"""
    if torch_release < (2, 8):
        # These things are cacheable by torch.compile. torch.compile just doesn't know it.
        # This was fixed in PyTorch 2.8, but until then, we monkey patch.
        import torch._higher_order_ops.auto_functionalize as af

        # 标记 auto_functionalized_v2 和 auto_functionalized 为可缓存
        af.auto_functionalized_v2._cacheable = True
        af.auto_functionalized._cacheable = True


def register_fake_if_exists(op_name):
    """
    Decorator factory to conditionally register a fake for a custom op if it exists.
    Parses op_name (e.g., 'sgl_kernel::gptq_gemm'), checks if the op exists via hasattr
    on the namespace attribute of torch.ops. Registers the fake if present; otherwise,
    returns the function unchanged.
    Args:
        op_name (str): Full operator name (e.g., 'sgl_kernel::gptq_gemm').
    Returns:
        callable: Decorator for the fake function.
    Example:
        @register_fake_if_exists('sgl_kernel::gptq_gemm')
        def fake_gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit):
            return a.new_empty((a.shape[0], b_q_weight.shape[-1]), dtype=a.dtype)
    """
    # 条件注册自定义算子 fake 实现的装饰器工厂函数。
    # 仅当算子存在时才注册 fake 实现，否则原样返回函数。

    def decorator(func):
        # 解析算子名，格式为 "命名空间::算子名"
        namespace, bare_op = op_name.split("::")
        # 获取 torch.ops 中对应的命名空间
        ops_namespace = getattr(torch.ops, namespace, None)
        if ops_namespace and hasattr(ops_namespace, bare_op):
            # 算子存在，注册 fake 实现（用于 torch.compile 推断输出形状和类型）
            torch.library.register_fake(op_name, func)
        return func

    return decorator
