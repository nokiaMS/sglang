# 模型工具函数模块
# 本文件提供了模型相关的工具函数和类，包括权重映射器(WeightsMapper)、自动权重加载器(AutoWeightsLoader)、
# 融合KV缓冲区设置、QK归一化、2D旋转位置编码辅助等功能。

# Copyright 2023-2025 SGLang Team
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
from __future__ import annotations  # 启用延迟注解求值

import itertools  # 导入迭代工具
from collections.abc import Iterable, Mapping  # 导入集合抽象基类
from dataclasses import dataclass, field  # 导入数据类装饰器
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import TYPE_CHECKING, Any, Optional, Tuple  # 导入类型提示

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm  # 导入融合QK归一化
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled  # 导入上下文并行工具
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool  # 导入滑动窗口KV池
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 导入CUDA图捕获模式
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool  # 导入KV池获取函数
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import get_current_device_stream_fast, is_cuda, is_hip  # 导入工具函数
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

if TYPE_CHECKING:
    from sglang.srt.layers.layernorm import RMSNorm  # 仅类型检查时导入

_is_cuda = is_cuda()  # 是否为CUDA设备
_is_hip = is_hip()  # 是否为HIP设备

WeightsMapping = Mapping[str, Optional[str]]  # 权重映射类型
"""If a key maps to a value of `None`, the corresponding weight is ignored."""  # 如果键映射到None，则忽略对应权重


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""  # 根据模式映射权重名称

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)  # 子串替换映射
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)  # 前缀替换映射
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)  # 后缀替换映射

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """合并两个权重映射器"""
        return WeightsMapper(
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},  # 合并子串映射
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},  # 合并前缀映射
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},  # 合并后缀映射
        )

    def _map_name(self, key: str) -> Optional[str]:
        """根据映射规则转换权重名称"""
        for substr, new_key in sorted(  # 按长度降序排列子串映射
            self.orig_to_new_substr.items(), key=lambda i: len(i[0]), reverse=True
        ):
            if substr in key:  # 如果键包含子串
                if new_key is None:  # 映射到None表示忽略
                    return None

                key = key.replace(substr, new_key, 1)  # 替换子串
                break  # 只替换第一个匹配

        for prefix, new_key in sorted(  # 按长度降序排列前缀映射
            self.orig_to_new_prefix.items(), key=lambda i: len(i[0]), reverse=True
        ):
            if key.startswith(prefix):  # 如果键以前缀开头
                if new_key is None:  # 映射到None表示忽略
                    return None

                key = key.replace(prefix, new_key, 1)  # 替换前缀
                break  # 只替换第一个匹配

        for suffix, new_key in sorted(  # 按长度降序排列后缀映射
            self.orig_to_new_suffix.items(), key=lambda i: len(i[0]), reverse=True
        ):
            if key.endswith(suffix):  # 如果键以后缀结尾
                if new_key is None:  # 映射到None表示忽略
                    return None

                key = new_key.join(key.rsplit(suffix, 1))  # 替换后缀
                break  # 只替换第一个匹配

        return key  # 返回转换后的键

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """将映射应用到权重迭代器"""
        return (
            (out_name, data)  # 输出名称和数据
            for name, data in weights  # 遍历权重
            if (out_name := self._map_name(name)) is not None  # 过滤被忽略的权重
        )

    def apply_list(self, values: list[str]) -> list[str]:
        """将映射应用到字符串列表"""
        return [
            out_name  # 输出名称
            for name in values  # 遍历值
            if (out_name := self._map_name(name)) is not None  # 过滤被忽略的
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        """将映射应用到字典"""
        return {
            out_name: value  # 输出名称和值
            for name, value in values.items()  # 遍历字典
            if (out_name := self._map_name(name)) is not None  # 过滤被忽略的
        }


class AutoWeightsLoader:
    """自动权重加载器，递归地将权重加载到模型参数中"""  # 旋转位置编码未使用的权重

    ROTARY_EMBEDS_UNUSED_WEIGHTS = [  # 旋转位置编码未使用的权重
        "rotary_pos_emb.inv_freq",  # 逆频率
        "rotary_emb.inv_freq",  # 逆频率
        "rotary_emb.cos_cached",  # 余弦缓存
        "rotary_emb.sin_cached",  # 正弦缓存
    ]

    def __init__(
        self,
        module: torch.nn.Module,  # 目标模块
        *,  # 强制关键字参数
        skip_prefixes: list[str] | None = None,  # 跳过的前缀
        skip_substrs: list[str] | None = None,  # 跳过的子串
        ignore_unexpected_prefixes: list[str] | None = None,  # 忽略的意外前缀
        ignore_unexpected_suffixes: list[str] | None = None,  # 忽略的意外后缀
    ) -> None:
        self.module = module  # 目标模块
        self.skip_prefixes = list(skip_prefixes or [])  # 跳过前缀列表
        self.skip_substrs = [  # 跳过子串列表
            *(skip_substrs or []),  # 用户指定的
            *self.ROTARY_EMBEDS_UNUSED_WEIGHTS,  # 旋转位置编码
        ]
        self.ignore_unexpected_prefixes = list(ignore_unexpected_prefixes or [])  # 忽略前缀
        self.ignore_unexpected_suffixes = list(ignore_unexpected_suffixes or [])  # 忽略后缀

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        """按前缀分组权重"""
        weights_by_parts = (  # 分割权重名
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )
        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):  # 按前缀分组
            yield prefix, (  # 输出前缀和组
                ("" if len(parts) == 1 else parts[1], weight_data)  # 剩余部分
                for parts, weight_data in group
            )

    @staticmethod
    def _get_qualname(prefix: str, rest: str) -> str:
        """获取限定名称"""
        if prefix == "":  # 前缀为空
            return rest  # 返回rest
        if rest == "":  # rest为空
            return prefix  # 返回前缀
        return f"{prefix}.{rest}"  # 拼接

    def _can_skip(self, qualname: str) -> bool:
        """检查是否应跳过该限定名"""
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(  # 前缀匹配
            sub in qualname for sub in self.skip_substrs  # 子串匹配
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        """检查是否应忽略意外的限定名"""
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes  # 前缀匹配
        ) or any(qualname.endswith(s) for s in self.ignore_unexpected_suffixes)  # 后缀匹配

    def _load_param(
        self,
        base_prefix: str,  # 基础前缀
        param: torch.nn.Parameter,  # 目标参数
        weights: Iterable[tuple[str, torch.Tensor]],  # 权重
    ) -> Iterable[str]:
        """加载参数权重"""
        for weight_name, weight_data in weights:  # 遍历权重
            weight_qualname = self._get_qualname(base_prefix, weight_name)  # 获取限定名
            if self._can_skip(weight_qualname):  # 跳过
                continue
            if weight_name != "":  # 还有嵌套
                if self._can_ignore_unexpected(weight_qualname):  # 忽略
                    continue
                raise ValueError(  # 不允许嵌套到参数
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into parameter {base_prefix!r}"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取加载器
            weight_loader(param, weight_data)  # 加载权重
            yield weight_qualname  # 返回限定名

    def _load_module(
        self,
        base_prefix: str,  # 基础前缀
        module: torch.nn.Module,  # 目标模块
        weights: Iterable[tuple[str, torch.Tensor]],  # 权重
    ) -> Iterable[str]:
        """递归加载模块权重"""
        if module.__class__.__name__ == "PPMissingLayer":  # PP缺失层
            return  # 跳过

        if module is not self.module:  # 非根模块
            module_load_weights = getattr(module, "load_weights", None)  # 获取load_weights方法
            if callable(module_load_weights):  # 方法可调用
                loaded = module_load_weights(weights)  # 调用
                if loaded is not None:  # 返回了加载名称
                    yield from (  # 输出限定名
                        self._get_qualname(base_prefix, loaded_name)
                        for loaded_name in loaded
                    )
                return  # 返回

        child_modules = dict(module.named_children())  # 子模块字典
        child_params = dict(module.named_parameters(recurse=False))  # 子参数字典
        child_buffers = dict(module.named_buffers(recurse=False))  # 子缓冲区字典
        for child_prefix, child_weights in self._groupby_prefix(weights):  # 按前缀分组
            prefix = self._get_qualname(base_prefix, child_prefix)  # 限定名
            if child_prefix in child_modules:  # 是子模块
                if self._can_skip(prefix + "."):  # 跳过
                    continue
                yield from self._load_module(  # 递归加载
                    prefix,
                    child_modules[child_prefix],
                    child_weights,
                )
                continue

            if child_prefix in child_params:  # 是子参数
                if self._can_skip(prefix):  # 跳过
                    continue
                yield from self._load_param(  # 加载参数
                    prefix, child_params[child_prefix], child_weights
                )
                continue

            if child_prefix in child_buffers:  # 是子缓冲区
                if self._can_skip(prefix):  # 跳过
                    continue
                yield from self._load_param(  # 加载缓冲区参数
                    prefix, child_buffers[child_prefix], child_weights
                )
                continue

            if self._can_skip(prefix) or self._can_skip(prefix + "."):  # 跳过
                continue
            if self._can_ignore_unexpected(prefix) or self._can_ignore_unexpected(  # 忽略
                prefix + "."
            ):
                continue
            raise ValueError(  # 未找到匹配的模块或参数
                f"No module or parameter named {prefix!r} in {self.module._get_name()}."
            )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],  # 权重迭代器
        *,  # 强制关键字参数
        mapper: WeightsMapper | None = None,  # 权重映射器
    ) -> set[str]:
        """加载所有权重"""
        if mapper is not None:  # 如果有映射器
            weights = mapper.apply(weights)  # 应用映射
        weights = (  # 过滤跳过的权重
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )
        return set(self._load_module("", self.module, weights))  # 递归加载并返回已加载名称集合


def enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""  # 仅在CUDA+bfloat16 KV缓存时启用融合KV缓冲区设置
    pool = get_token_to_kv_pool()  # 获取KV池
    return (
        _is_cuda  # 是CUDA
        and pool.dtype == torch.bfloat16  # bfloat16
        and not isinstance(pool, SWAKVPool)  # 不是SWA池
        and not is_prefill_context_parallel_enabled()  # 未启用上下文并行
    ) or (_is_hip and not is_prefill_context_parallel_enabled())  # 或HIP且无上下文并行


def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,  # 值张量
    layer: RadixAttention,  # 注意力层
    forward_batch: ForwardBatch,  # 前向批次
):
    """创建融合KV缓冲区设置参数"""
    from sglang.jit_kernel.rope import FusedSetKVBufferArg  # 导入融合参数类

    layer_id = layer.layer_id  # 获取层ID
    token_to_kv_pool = get_token_to_kv_pool()  # 获取KV池

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)  # 获取键缓冲区
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)  # 获取值缓冲区

    if not _is_hip:  # 非HIP（CUDA）
        assert layer.k_scale is None and layer.v_scale is None, "scale not supported"  # 不支持缩放
        return FusedSetKVBufferArg(  # 创建融合参数
            value=value,
            k_buffer=k_buffer.view(k_buffer.shape[0], -1),  # 展平键缓冲区
            v_buffer=v_buffer.view(v_buffer.shape[0], -1),  # 展平值缓冲区
            cache_loc=forward_batch.out_cache_loc,  # 缓存位置
        )
    else:  # HIP
        page_size = token_to_kv_pool.page_size  # 页面大小
        slot_mapping_swa = (  # SWA槽映射
            token_to_kv_pool.full_to_swa_index_mapping.long()
            if layer.sliding_window_size > 0  # 有滑动窗口
            else None  # 无滑动窗口
        )
        return {  # 返回字典参数
            "v": value.view(-1, layer.tp_v_head_num, layer.v_head_dim),  # 值视图
            "k_scale": layer.k_scale,  # 键缩放
            "v_scale": layer.v_scale,  # 值缩放
            "key_cache": k_buffer.view(  # 键缓存视图
                -1, page_size, layer.tp_k_head_num, layer.qk_head_dim
            ),
            "value_cache": v_buffer.view(  # 值缓存视图
                -1, page_size, layer.tp_v_head_num, layer.v_head_dim
            ),
            "slot_mapping": forward_batch.out_cache_loc,  # 槽映射
            "swa_slot_mapping": slot_mapping_swa,  # SWA槽映射
        }


def permute_inv(perm: torch.Tensor) -> torch.Tensor:
    """计算排列的逆排列"""
    inv_perm = torch.empty_like(perm)  # 创建空张量
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)  # 设置逆排列
    return inv_perm  # 返回逆排列


def compute_cu_seqlens_from_grid_numpy(grid_thw: torch.Tensor) -> torch.Tensor:
    """使用NumPy从grid_thw计算cu_seqlens
    Compute cu_seqlens from grid_thw using NumPy.

    grid_thw: [T, 3] int tensor on CPU.
              columns: [repeat_count, H, W]
    Returns:
        cu_seqlens: 1D int32 tensor on CPU, shape [N + 1]
    """
    assert (
        grid_thw.device.type == "cpu"
    ), "compute_cu_seqlens_from_grid_numpy expects a CPU tensor"  # 断言在CPU上
    arr = grid_thw.numpy()  # 转为NumPy数组

    cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(  # 重复并累加
        axis=0, dtype=np.int32
    )
    cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])  # 前面添加0
    cu_seqlens = torch.from_numpy(cu_seqlens)  # 转回PyTorch张量
    return cu_seqlens  # 返回cu_seqlens


class RotaryPosMixin:
    """旋转位置编码混入类，提供2D位置ID计算"""

    @staticmethod
    @lru_cache(maxsize=1024)  # LRU缓存
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        """计算2D旋转位置ID"""
        if isinstance(h, torch.Tensor):  # 张量转整数
            h = int(h.item())
        if isinstance(w, torch.Tensor):  # 张量转整数
            w = int(w.item())
        if isinstance(spatial_merge_size, torch.Tensor):  # 张量转整数
            spatial_merge_size = int(spatial_merge_size.item())
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))  # 高度位置ID
        h_div = h // spatial_merge_size  # 高度除以合并大小
        w_div = w // spatial_merge_size  # 宽度除以合并大小
        hpos_ids = hpos_ids.reshape(  # 重塑高度位置ID
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)  # 转置
        hpos_ids = hpos_ids.flatten()  # 展平

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))  # 宽度位置ID
        wpos_ids = wpos_ids.reshape(  # 重塑宽度位置ID
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)  # 转置
        wpos_ids = wpos_ids.flatten()  # 展平

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))  # 堆叠并返回


def _reshape_for_qk_norm(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Reshape a (..., H*D) tensor into (..., H, D) ahead of QK RMSNorm.
    
    On CUDA with the inductor piecewise-cuda-graph compiler, return a
    stride-preserving view so inductor can fuse this reshape with the
    subsequent RMSNorm (and any upstream/downstream FP8 quant) into a
    single triton kernel -- the original motivation of #21734.
    
    Everywhere else (ROCm, or CUDA with the eager PCG fallback), use the
    flat 2D reshape that forces a copy when the input is a non-contiguous
    QKV-split stride-trick view. ROCm's RMSNorm kernels assume contiguous
    inputs and fault on strided tensors (root cause of the #21734 revert
    in #23159).
    """  # 为QK归一化重塑张量形状
    if (
        _is_cuda  # CUDA
        and get_global_server_args().piecewise_cuda_graph_compiler == "inductor"  # inductor编译器
    ):
        return x.view(*x.shape[:-1], -1, head_dim)  # 保持步幅的视图
    return x.reshape(-1, head_dim)  # 强制复制的2D重塑


def apply_qk_norm(
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    q_norm: RMSNorm,  # Q归一化层
    k_norm: RMSNorm,  # K归一化层
    head_dim: int,  # 头维度
    alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    allow_inplace: bool = True,  # 是否允许原地操作
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用QK归一化，如果条件满足则使用JIT融合原地归一化
    Apply QK normalization for query and key tensors.
    If eligible, we will use JIT fused inplace QK normalization for better performance.

    Args:
        q: Query tensor of shape [batch_size, ...]
        k: Key tensor of shape [batch_size, ...]
        q_norm: RMSNorm layer for query normalization
        k_norm: RMSNorm layer for key normalization
        head_dim: Dimension of each attention head
        alt_stream: Optional alternative CUDA stream for overlapping computation
        allow_inplace: Whether to allow inplace normalization. (True for better performance)

    Returns:
        Tuple of normalized query and key tensors
    """

    batch_size = q.size(0)  # 批次大小
    q_eps = q_norm.variance_epsilon  # Q归一化eps
    k_eps = k_norm.variance_epsilon  # K归一化eps
    if (
        _is_cuda  # TODO(dark): have not tested on ROCm or other backends
        and allow_inplace  # TODO(dark): this can be relaxed if needed
        and (q_eps == k_eps)  # TODO(dark): this can also be relaxed
        and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()  # 非确定性推理
        and get_global_server_args().piecewise_cuda_graph_compiler
        != "inductor"  # let inductor fuse QK norm
        and can_use_fused_inplace_qknorm(head_dim, q.dtype)  # 可以使用融合归一化
    ):
        fused_inplace_qknorm(  # 融合原地QK归一化
            q=q.view(batch_size, -1, head_dim),
            k=k.view(batch_size, -1, head_dim),
            q_weight=q_norm.weight,
            k_weight=k_norm.weight,
            head_dim=head_dim,
            eps=q_eps,
        )
        return q, k  # 返回归一化后的q和k

    if alt_stream is not None and get_is_capture_mode():  # 有备用流且在捕获模式
        current_stream = get_current_device_stream_fast()  # 获取当前流
        alt_stream.wait_stream(current_stream)  # 等待当前流
        q_by_head = _reshape_for_qk_norm(q, head_dim)  # 重塑Q
        q_by_head = q_norm(q_by_head)  # Q归一化
        with torch.cuda.stream(alt_stream):  # 在备用流上
            k_by_head = _reshape_for_qk_norm(k, head_dim)  # 重塑K
            k_by_head = k_norm(k_by_head)  # K归一化
        current_stream.wait_stream(alt_stream)  # 等待备用流
    else:  # 无备用流
        q_by_head = _reshape_for_qk_norm(q, head_dim)  # 重塑Q
        q_by_head = q_norm(q_by_head)  # Q归一化
        k_by_head = _reshape_for_qk_norm(k, head_dim)  # 重塑K
        k_by_head = k_norm(k_by_head)  # K归一化
    q = q_by_head.view(q.shape)  # 恢复Q形状
    k = k_by_head.view(k.shape)  # 恢复K形状
    return q, k  # 返回归一化后的q和k


# ---------------------------------------------------------------------------
# Fused QK GemmaRMSNorm Triton kernel
# grid = q_rows (the larger dimension in GQA).  Every block computes Q norm
# for its row; the first k_rows blocks also compute K norm.  No torch.cat,
# no tl.where for weight selection, no output slice.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_qk_gemma_rmsnorm_kernel(
    Q_ptr,  # Q指针
    K_ptr,  # K指针
    Q_out_ptr,  # Q输出指针
    K_out_ptr,  # K输出指针
    QW_ptr,  # Q权重指针
    KW_ptr,  # K权重指针
    q_stride,  # Q步幅
    k_stride,  # K步幅
    k_rows,  # K行数
    HEAD_DIM: tl.constexpr,  # 头维度（编译时常量）
    BLOCK_HD: tl.constexpr,  # 块大小（编译时常量）
    EPS: tl.constexpr,  # eps值（编译时常量）
    FP16: tl.constexpr,  # 是否FP16（编译时常量）
):
    """融合QK GemmaRMSNorm Triton内核"""
    pid = tl.program_id(0)  # 获取程序ID
    cols = tl.arange(0, BLOCK_HD)  # 列索引
    mask = cols < HEAD_DIM  # 掩码
    out_dtype = tl.float16 if FP16 else tl.bfloat16  # 输出数据类型

    # Q norm (every block) — use q_stride to handle non-contiguous input
    q_off = pid * q_stride + cols  # Q偏移
    q = tl.load(Q_ptr + q_off, mask=mask, other=0.0).to(tl.float32)  # 加载Q
    w_q = tl.load(QW_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载Q权重
    q_var = tl.sum(q * q, axis=0) / HEAD_DIM  # 计算方差
    q_normed = (q * tl.rsqrt(q_var + EPS) * (w_q + 1.0)).to(out_dtype)  # 归一化Q
    # output is always contiguous
    q_out_off = pid * HEAD_DIM + cols  # Q输出偏移
    tl.store(Q_out_ptr + q_out_off, q_normed, mask=mask)  # 存储Q

    # K norm (first k_rows blocks only) — use k_stride for input
    if pid < k_rows:  # 只在前k_rows个块计算K
        k_off = pid * k_stride + cols  # K偏移
        k = tl.load(K_ptr + k_off, mask=mask, other=0.0).to(tl.float32)  # 加载K
        w_k = tl.load(KW_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载K权重
        k_var = tl.sum(k * k, axis=0) / HEAD_DIM  # 计算方差
        k_normed = (k * tl.rsqrt(k_var + EPS) * (w_k + 1.0)).to(out_dtype)  # 归一化K
        k_out_off = pid * HEAD_DIM + cols  # K输出偏移
        tl.store(K_out_ptr + k_out_off, k_normed, mask=mask)  # 存储K


def fused_qk_gemma_rmsnorm(
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    q_weight: torch.Tensor,  # Q权重
    k_weight: torch.Tensor,  # K权重
    eps: float,  # eps值
    head_dim: int,  # 头维度
) -> Tuple[torch.Tensor, torch.Tensor]:
    """融合QK GemmaRMSNorm — 单个Triton内核同时计算q_norm和k_norm
    Fused QK GemmaRMSNorm — single Triton kernel for both q_norm and k_norm.

    grid = q_rows; every block processes its Q row, and the first k_rows
    blocks also process K.  No torch.cat, no slice, no tl.where.
    Passes input strides to the kernel so non-contiguous tensors (e.g. from
    qkv.split()) are read correctly without an extra .contiguous() copy.
    """
    q_flat = q.reshape(-1, head_dim)  # 展平Q
    k_flat = k.reshape(-1, head_dim)  # 展平K

    q_rows = q_flat.shape[0]  # Q行数
    k_rows = k_flat.shape[0]  # K行数

    q_out = torch.empty(q_rows, head_dim, dtype=q.dtype, device=q.device)  # Q输出
    k_out = torch.empty(k_rows, head_dim, dtype=k.dtype, device=k.device)  # K输出

    BLOCK_HD = triton.next_power_of_2(head_dim)  # 块大小（2的幂）

    _fused_qk_gemma_rmsnorm_kernel[(q_rows,)](  # 启动内核
        q_flat,
        k_flat,
        q_out,
        k_out,
        q_weight,
        k_weight,
        q_flat.stride(0),  # Q步幅
        k_flat.stride(0),  # K步幅
        k_rows,
        HEAD_DIM=head_dim,
        BLOCK_HD=BLOCK_HD,
        EPS=eps,
        FP16=(q.dtype == torch.float16),  # 是否FP16
    )

    return q_out, k_out  # 返回归一化后的Q和K


# Register the inplace op
fused_inplace_qknorm = register_custom_op(fused_inplace_qknorm, mutates_args=["q", "k"])  # 注册原地操作
