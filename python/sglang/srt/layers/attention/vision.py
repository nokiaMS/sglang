# 视觉注意力层模块
# 提供多种视觉Transformer注意力后端实现，包括SDPA、Triton、FlashAttention 3/4、FlashInfer/cuDNN、
# AMD Aiter、华为Ascend NPU和Intel AMX等后端，用于多模态模型中视觉编码器的注意力计算

from __future__ import annotations  # 启用延迟注解评估

import dataclasses  # 数据类装饰器
import functools  # 函数工具模块
import math  # 数学运算模块
import warnings  # 警告模块
from functools import lru_cache, partial  # LRU缓存和偏函数
from typing import Any, Callable, Optional, Tuple  # 类型提示

import torch  # PyTorch核心库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
from einops import rearrange  # 张量重排工具

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm as can_use_jit_qk_norm  # JIT QK归一化检查
from sglang.srt.environ import envs  # 环境变量
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size  # 张量并行相关
from sglang.srt.models.utils import apply_qk_norm  # QK归一化应用
from sglang.srt.utils import (  # 工具函数导入
    cpu_has_amx_support,  # CPU AMX支持检测
    get_bool_env_var,  # 布尔环境变量获取
    get_device_capability,  # 设备计算能力获取
    is_blackwell_supported,  # Blackwell架构支持检测
    is_cpu,  # 是否CPU
    is_cuda,  # 是否CUDA
    is_hip,  # 是否HIP(AMD)
    is_musa,  # 是否摩尔线程GPU
    is_npu,  # 是否NPU(华为昇腾)
    is_xpu,  # 是否XPU(Intel)
    print_info_once,  # 单次打印信息
)
from sglang.srt.utils.multi_stream_utils import (  # 多流工具
    maybe_execute_in_parallel,  # 可能并行执行
    with_multi_stream,  # 多流上下文
)

_is_cpu = is_cpu()  # 是否为CPU平台
_is_cuda = is_cuda()  # 是否为CUDA平台
_is_musa = is_musa()  # 是否为摩尔线程平台
_is_npu = is_npu()  # 是否为昇腾NPU平台
_is_hip = is_hip()  # 是否为AMD HIP平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU是否支持AMX指令集
_is_xpu = is_xpu()  # 是否为Intel XPU平台

if _is_cuda:  # 如果是CUDA平台
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache  # cuDNN批量预填充注意力

    from sglang.jit_kernel.flash_attention import (  # JIT Flash Attention
        flash_attn_varlen_func,  # 变长Flash Attention函数
    )

if _is_cpu and _is_cpu_amx_available:  # 如果是CPU且支持AMX
    flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func  # 使用sgl_kernel的flash attn

if _is_musa:  # 如果是摩尔线程平台
    from flash_attn_interface import flash_attn_varlen_func  # 摩尔线程Flash Attention

if _is_npu:  # 如果是昇腾NPU平台
    import torch_npu  # 昇腾NPU扩展

from sglang.srt.distributed import (  # 分布式通信
    split_tensor_along_last_dim,  # 沿最后一维切分张量
    tensor_model_parallel_all_gather,  # 张量并行全收集
)
from sglang.srt.distributed import utils as dist_utils  # 分布式工具
from sglang.srt.layers.attention.triton_ops.prefill_attention import (  # Triton预填充注意力
    context_attention_fwd,  # 上下文注意力前向
)
from sglang.srt.layers.layernorm import RMSNorm  # RMS归一化层
from sglang.srt.layers.linear import (  # 线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.quantization import QuantizationConfig  # 量化配置
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb  # 旋转位置编码应用
from sglang.srt.server_args import get_global_server_args  # 全局服务器参数
from sglang.srt.utils import add_prefix, get_bool_env_var  # 前缀添加和布尔环境变量

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AMD Aiter后端

ROTARY_EMBED_CLASSES = {  # 旋转嵌入类别映射
    "normal": apply_rotary_pos_emb,  # 标准旋转位置编码
}

# === Vision Encoder === #  # === 视觉编码器 === #
FLASHINFER_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024  # FlashInfer工作空间大小，128MB

# Batch buckets for cuDNN graph caching - graphs are cached per bucket size
# This avoids creating a new graph for each unique batch size at runtime
# cuDNN图缓存的批量桶 - 图按桶大小缓存，避免运行时为每个唯一批量大小创建新图
BATCH_BUCKETS = [8, 16, 32, 64]  # 批量桶大小列表

# Bucketized max seqlens to reduce cuDNN recompilation frequency while
# preserving a tighter upper bound than a single fixed max seqlen.
# 分桶的最大序列长度，减少cuDNN重编译频率，同时保持比单一固定最大序列长度更紧的上界。
FLASHINFER_MAX_SEQLEN_BUCKETS = [  # FlashInfer最大序列长度桶
    4 * 1024,  # 4K
    8 * 1024,  # 8K
    16 * 1024,  # 16K
    32 * 1024,  # 32K
    64 * 1024,  # 64K
    128 * 1024,  # 128K
]


@dataclasses.dataclass  # 单例缓存数据类，用于缓存单一数据对象
class SingletonCache:
    data: Any = None  # 缓存的数据

    def set_data(self, value: Any) -> None:  # 设置缓存数据
        self.data = value  # 赋值数据

    def get_data(self) -> Optional[Any]:  # 获取缓存数据
        return self.data  # 返回数据

    def empty(self) -> bool:  # 判断缓存是否为空
        return self.get_data() is None  # 数据为None则空


# TODO: requires real seqlens from images  # TODO: 需要来自图像的真实序列长度
@functools.lru_cache(maxsize=128)  # LRU缓存，最大128项
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    # 为给定batch_size、seqlen和device生成累积序列长度(cu_seqlens)，基于这些参数缓存结果
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    生成给定batch_size、seqlen和device的累积序列长度(cu_seqlens)，基于这些参数缓存结果。
    """
    cu_seqlens = torch.arange(  # 生成等差序列
        0,  # 起始值
        (batch_size + 1) * seqlen,  # 结束值
        step=seqlen,  # 步长
        dtype=torch.int32,  # 32位整型
        device=device,  # 目标设备
    )
    return cu_seqlens  # 返回累积序列长度


def resolve_seqlens(  # 解析序列长度，将不同来源的cu_seqlens统一为torch.Tensor
    cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度，可以是张量、单例缓存或None
    bsz: int,  # 批量大小
    seq_len: int,  # 序列长度
    *,
    device: torch.device,  # 目标设备
) -> torch.Tensor:  # 返回解析后的累积序列长度张量
    if cu_seqlens is None:  # 如果为None
        resolved_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=device)  # 根据形状生成
    elif isinstance(cu_seqlens, SingletonCache):  # 如果是单例缓存
        if cu_seqlens.empty():  # 如果缓存为空
            cu_seqlens.set_data(_get_cu_seqlens_for_shape(bsz, seq_len, device=device))  # 设置缓存
        resolved_seqlens = cu_seqlens.get_data()  # 获取缓存数据
    else:  # 其他情况
        resolved_seqlens = cu_seqlens  # 直接使用
    assert isinstance(  # 断言类型
        resolved_seqlens, torch.Tensor
    ), "cu_seqlens must be a torch.Tensor"  # cu_seqlens必须是torch.Tensor
    return resolved_seqlens  # 返回解析后的序列长度


class VisionSdpaAttention(nn.Module):  # 视觉SDPA注意力模块，使用PyTorch缩放点积注意力
    r"""
    Scaled Dot Product Attention inner product
    缩放点积注意力内积

    """

    def __init__(  # 初始化方法
        self,
        head_dim: int,  # 头维度
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        dropout: float = 0.0,  # dropout比率
        flatten_batch: bool = False,  # 是否展平批量维度
        softmax_in_single_precision: bool = False,  # 是否使用单精度softmax
        softmax_scale: float | None = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ):
        super().__init__()  # 调用父类初始化
        self.head_size = head_dim  # 头大小
        self.num_heads = num_heads  # 头数
        self.num_kv_heads = num_kv_heads  # KV头数
        self.flatten_batch = flatten_batch  # 是否展平批量
        self.softmax_in_single_precision = softmax_in_single_precision  # 单精度softmax标志
        self.dropout = dropout  # dropout率
        self.scale = (  # 缩放因子
            softmax_scale  # 使用指定缩放
            if softmax_scale is not None  # 如果指定了
            else 1.0 / math.sqrt(self.head_size)  # 否则使用1/sqrt(head_size)
        )

    @staticmethod  # 静态方法
    @lru_cache(maxsize=128)  # LRU缓存，最大128项
    def _generate_mask_cache(  # 生成带缓存的布尔注意力掩码
        s: int, flatten_batch: bool, cu_seqlens: tuple  # 序列长度、是否展平批量、累积序列长度元组
    ) -> torch.BoolTensor:  # 返回布尔掩码张量
        """
        Generate a boolean attention mask with caching mechanism.
        生成带缓存机制的布尔注意力掩码。
        Args:
            s: sequence length  # 序列长度
            flatten_batch: whether to flatten batch dimension  # 是否展平批量维度
            cu_seqlens: tuple of cumulative sequence lengths  # 累积序列长度的元组
        Returns:
            attention mask tensor of shape [b, 1, s, s] or [1, s, s]
            注意力掩码张量，形状为[b, 1, s, s]或[1, s, s]
        """
        if flatten_batch:  # 如果展平批量
            mask = torch.zeros([1, s, s], dtype=torch.bool)  # 初始化全零掩码
            for i in range(1, len(cu_seqlens)):  # 遍历每个序列
                start = cu_seqlens[i - 1]  # 起始位置
                end = cu_seqlens[i]  # 结束位置
                mask[..., start:end, start:end] = True  # 设置对角块为True
        else:  # 不展平批量
            # [1, 1, 1, s]  # 行索引
            row_indices = torch.arange(s).view(1, 1, 1, s)  # 行索引张量
            # [1, 1, s, 1]  # 列索引
            col_indices = torch.arange(s).view(1, 1, s, 1)  # 列索引张量
            # [b, 1, 1, 1]  # 序列长度
            seq_lens = torch.tensor(  # 构造序列长度张量
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],  # 计算每个序列长度
            ).view(-1, 1, 1, 1)  # 重塑形状

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)  # 行列都在序列长度内则为True

        return mask  # 返回掩码

    def generate_patch_attention_mask(  # 生成patch注意力掩码
        self,
        s: int,  # 序列长度
        cu_seqlens: Optional[torch.Tensor],  # 累积序列长度
        flatten_batch: bool = False,  # 是否展平批量
    ) -> Optional[torch.Tensor]:  # 返回掩码张量或None
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        创建形状为(b, 1, s, s)或(1, 1, s, s)的非因果4D掩码。
        Args:
            s: sequence length  # 序列长度
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            # cu_seqlens: 累积序列长度张量。如果未提供，返回空掩码
            flatten_batch: whether to flatten batch dimension  # 是否展平批量维度
        Returns:
            attention mask tensor or None  # 注意力掩码张量或None
        """
        if cu_seqlens is None:  # 如果没有提供cu_seqlens
            return None  # 返回None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())  # 转为元组用于缓存

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)  # 调用带缓存的方法

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        bsz: int,  # 批量大小
        cu_seqlens: Optional[torch.Tensor] = None,  # 累积序列长度
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
        Returns:
             [b * s, h, head_size]  # 返回形状为[b*s, h, head_size]的输出
        """
        if self.flatten_batch:  # 如果展平批量
            assert bsz == 1, "flatten_batch is True, bsz must be 1"  # 展平批量时bsz必须为1

        assert q.dim() == 3, q.shape  # q必须是3维

        s = q.shape[0] // bsz  # 计算序列长度

        # [b, 1, s, s]  # 注意力掩码形状
        if attention_mask is None:  # 如果没有提供掩码
            attention_mask = self.generate_patch_attention_mask(  # 生成patch注意力掩码
                s, cu_seqlens, flatten_batch=self.flatten_batch  # 传入参数
            )

        if attention_mask is None:  # 如果掩码仍为None
            if self.softmax_in_single_precision:  # 如果需要单精度softmax
                raise RuntimeError("Empty attention mask")  # 单精度模式下不允许空掩码
        else:  # 有掩码
            attention_mask = attention_mask.to(device=q.device)  # 将掩码移到与q相同的设备

        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]  # 重排为[b, h, s, d]

        if self.softmax_in_single_precision:  # 如果使用单精度softmax
            k = rearrange(k, "b h s d -> b h d s")  # 转置k用于矩阵乘法
            attn_weights = torch.matmul(q, k) * self.scale  # 计算注意力权重并缩放
            del k  # 释放k
            # masking  # 掩码处理
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min  # 反转掩码并设为最小值
            attn_weights = attn_weights + attention_mask  # 应用掩码
            del attention_mask  # 释放掩码
            # full-precision  # 全精度softmax
            attn_weights = nn.functional.softmax(  # softmax
                attn_weights, dim=-1, dtype=torch.float32  # 沿最后一维，float32精度
            ).to(q.dtype)  # 转回原始精度
            attn_weights = nn.functional.dropout(  # dropout
                attn_weights, p=self.dropout, training=False  # 推理模式不dropout
            )
            output = torch.matmul(attn_weights, v)  # 加权求和
            del attn_weights, v  # 释放内存
        else:  # 使用SDPA
            # SDPA  # 缩放点积注意力
            # [b, h, s, head_size]  # 输出形状
            output = F.scaled_dot_product_attention(  # PyTorch原生SDPA
                q,  # 查询
                k,  # 键
                v,  # 值
                attn_mask=attention_mask,  # 注意力掩码
                dropout_p=self.dropout,  # dropout率
                is_causal=False,  # 非因果注意力
                scale=self.scale,  # 缩放因子
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]  # 从[b,h,s,d]变为[b*s,h,d]
        output = rearrange(output, "b h s d -> (b s) h d")  # 重排输出

        return output  # 返回输出


class VisionTritonAttention(nn.Module):  # 视觉Triton注意力模块，使用Triton实现的非因果注意力
    """
    Triton-implemented attention without a causal mask
    使用Triton实现的非因果注意力
    """

    def __init__(  # 初始化方法
        self,
        **kwargs,  # 其他参数
    ):
        super().__init__()  # 调用父类初始化
        use_data_parallel = (  # 是否使用数据并行
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False  # 默认不使用
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()  # 张量并行大小

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度
        bsz: int,  # 批量大小
        seq_len: int,  # 序列长度
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
            softmax_scale: override softmax scale (default 1/sqrt(head_dim))
            # softmax_scale: 覆盖softmax缩放（默认1/sqrt(head_dim)）
        Returns:
             [b * s, h, head_size]  # 返回形状为[b*s, h, head_size]的输出
        """
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():  # 如果启用了CUDA图
            if "output_ws" not in kwargs:  # 如果没有输出工作空间
                raise RuntimeError("output_ws should be prepared for cuda-graph mode")  # 需要准备输出工作空间

            if not isinstance(cu_seqlens, list):  # cu_seqlens必须是列表
                raise RuntimeError("cuda-graph mode cu_seqlens should be a list")  # CUDA图模式下cu_seqlens应为列表

            output = kwargs["output_ws"]  # 使用预分配的输出工作空间
            context_attention_fwd(  # 调用Triton上下文注意力
                q,  # 查询
                k,  # 键
                v,  # 值
                output,  # 输出
                cu_seqlens[0],  # cu_seqlens_q
                cu_seqlens[1],  # seq_lens
                cu_seqlens[2],  # max_seqlen
                is_causal=False,  # 非因果
                sm_scale=softmax_scale,  # softmax缩放
            )
        else:  # 非CUDA图模式
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)  # 解析序列长度

            # [b * s, head, head_size]  # 输出形状
            output = torch.empty_like(q)  # 预分配输出

            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算各序列长度
            max_seqlen = seq_lens.max().item()  # 获取最大序列长度
            context_attention_fwd(  # 调用Triton上下文注意力
                q,  # 查询
                k,  # 键
                v,  # 值
                output,  # 输出
                cu_seqlens.to(q.device),  # 累积序列长度
                seq_lens.to(q.device),  # 各序列长度
                max_seqlen,  # 最大序列长度
                is_causal=False,  # 非因果
                sm_scale=softmax_scale,  # softmax缩放
            )

        return output  # 返回输出


class VisionFlash3Attention(nn.Module):  # 视觉Flash Attention 3注意力模块
    def __init__(  # 初始化方法
        self,
        **kwargs,  # 其他参数
    ):
        if not (_is_cuda or _is_musa):  # 仅CUDA或摩尔线程可用
            raise Exception("VisionFlash3Attention is only available for cuda or musa")  # FA3仅支持cuda或musa
        super().__init__()  # 调用父类初始化
        use_data_parallel = (  # 是否使用数据并行
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False  # 默认不使用
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()  # 张量并行大小

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度
        bsz: int,  # 批量大小
        seq_len: int,  # 序列长度
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
        Returns:
             [b * s, h, head_size]  # 返回形状为[b*s, h, head_size]的输出
        """
        window_size = kwargs.get("window_size", (-1, -1))  # 获取窗口大小
        s_aux = kwargs.get("s_aux", None)  # 获取辅助sink参数

        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():  # 如果启用了CUDA图
            max_seqlen = cu_seqlens[1]  # 从cu_seqlens获取最大序列长度
            fa_kwargs = dict(  # Flash Attention参数字典
                cu_seqlens_q=cu_seqlens[0],  # 查询的累积序列长度
                cu_seqlens_k=cu_seqlens[0],  # 键的累积序列长度
                max_seqlen_q=max_seqlen,  # 查询最大序列长度
                max_seqlen_k=max_seqlen,  # 键最大序列长度
                softmax_scale=softmax_scale,  # softmax缩放
                window_size=window_size,  # 窗口大小
            )
            if s_aux is not None:  # 如果有sink参数
                fa_kwargs["sinks"] = s_aux  # 添加sink
            output = flash_attn_varlen_func(q, k, v, **fa_kwargs)  # 调用变长Flash Attention
        else:  # 非CUDA图模式
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)  # 解析序列长度
            cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)  # 转为int32并移到q设备
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算各序列长度
            max_seqlen = seq_lens.max().item()  # 获取最大序列长度

            fa_kwargs = dict(  # Flash Attention参数字典
                cu_seqlens_q=cu_seqlens,  # 查询的累积序列长度
                cu_seqlens_k=cu_seqlens,  # 键的累积序列长度
                max_seqlen_q=max_seqlen,  # 查询最大序列长度
                max_seqlen_k=max_seqlen,  # 键最大序列长度
                softmax_scale=softmax_scale,  # softmax缩放
                window_size=window_size,  # 窗口大小
            )
            if s_aux is not None:  # 如果有sink参数
                fa_kwargs["sinks"] = s_aux  # 添加sink
            output = flash_attn_varlen_func(q, k, v, **fa_kwargs)  # 调用变长Flash Attention

        return output  # 返回输出


class VisionFlash4Attention(nn.Module):  # 视觉Flash Attention 4注意力模块，用于Blackwell架构
    def __init__(  # 初始化方法
        self,
        **kwargs,  # 其他参数
    ):
        if not _is_cuda:  # 仅CUDA可用
            raise Exception("VisionFlash4Attention is only available for cuda")  # FA4仅支持cuda
        super().__init__()  # 调用父类初始化

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度
        bsz: int,  # 批量大小
        seq_len: int,  # 序列长度
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
        Returns:
             [b * s, h, head_size]  # 返回形状为[b*s, h, head_size]的输出
        """
        if cu_seqlens is None:  # 如果cu_seqlens为None
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)  # 根据形状生成
        elif isinstance(cu_seqlens, SingletonCache):  # 如果是单例缓存
            if cu_seqlens.empty():  # 如果缓存为空
                cu_seqlens.set_data(  # 设置缓存
                    _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)  # 根据形状生成
                )
            cu_seqlens = cu_seqlens.get_data()  # 获取缓存数据

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)  # 转为int32并移到q设备
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算各序列长度
        max_seqlen = seq_lens.max().item()  # 获取最大序列长度

        output = flash_attn_varlen_func(  # 调用变长Flash Attention
            q,  # 查询
            k,  # 键
            v,  # 值
            cu_seqlens_q=cu_seqlens,  # 查询累积序列长度
            cu_seqlens_k=cu_seqlens,  # 键累积序列长度
            max_seqlen_q=max_seqlen,  # 查询最大序列长度
            max_seqlen_k=max_seqlen,  # 键最大序列长度
            softmax_scale=softmax_scale,  # softmax缩放
            ver=4,  # 使用FA4版本
        )

        return output  # 返回输出


class VisionFlashInferAttention(nn.Module):  # 视觉FlashInfer/cuDNN注意力模块
    def __init__(  # 初始化方法
        self,
        **kwargs,  # 其他参数
    ):
        if not _is_cuda:  # 仅CUDA可用
            raise Exception("VisionFlashInferAttention is only available for cuda")  # FlashInfer仅支持cuda
        super().__init__()  # 调用父类初始化
        self.workspace_buffer = (  # 工作空间缓冲区
            kwargs["workspace_buffer"] if "workspace_buffer" in kwargs else None  # 默认为None
        )

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度
        bsz: int,  # 批量大小
        seq_len: int,  # 序列长度
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
        Returns:
             [b * s, h, head_size]  # 返回形状为[b*s, h, head_size]的输出
        """
        if "sequence_lengths" not in kwargs:  # 检查sequence_lengths参数
            raise RuntimeError(
                "sequence_lengths should be prepared for vision flashinfer_cudnn attention backend"
            )  # sequence_lengths应已准备好
        if "max_seqlen" not in kwargs:  # 检查max_seqlen参数
            raise RuntimeError(
                "max_seqlen should be prepared for vision flashinfer_cudnn attention backend"
            )  # max_seqlen应已准备好

        sequence_lengths = kwargs["sequence_lengths"]  # (B_padded,) or (B_padded,1,1,1)  # 序列长度
        max_seqlen = kwargs["max_seqlen"]  # 最大序列长度

        # max_seqlen must be python int  # max_seqlen必须是Python整数
        if isinstance(max_seqlen, torch.Tensor):  # 如果是张量
            if max_seqlen.is_cuda:  # 如果在GPU上
                max_seqlen = int(max_seqlen.detach().cpu().item())  # 从GPU取值转为int
            else:  # 在CPU上
                max_seqlen = int(max_seqlen.item())  # 直接取值转为int
        else:  # 不是张量
            max_seqlen = int(max_seqlen)  # 直接转为int

        # flatten if caller gives (b, s, h, d)  # 如果调用者给出4D输入则展平
        is_reshaped = q.dim() == 4  # 检查是否为4D
        if is_reshaped:  # 如果是4D
            reshape_batch_size = q.shape[0]  # 保存批量大小
            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])  # 展平为3D

        if not isinstance(cu_seqlens, torch.Tensor):  # cu_seqlens必须是张量
            raise RuntimeError(
                "flashinfer_cudnn expects packed indptrs as a torch.Tensor"
            )  # flashinfer_cudnn需要张量形式的打包indptr

        # sequence_lengths -> (B,)  # 序列长度转为1D
        if not isinstance(sequence_lengths, torch.Tensor):  # sequence_lengths必须是张量
            raise RuntimeError("sequence_lengths must be a torch.Tensor")  # sequence_lengths必须是torch.Tensor
        seq_lens_1d = sequence_lengths.view(-1).to(device=q.device, dtype=torch.int32)  # 展平并转int32
        B = int(seq_lens_1d.numel())  # 批量大小

        # cu_seqlens contains packed *element indptrs*:
        # [qk_indptr(B+1), v_indptr(B+1), o_indptr(B+1)] => total 3*(B+1)
        # cu_seqlens包含打包的*元素索引指针*：
        # [qk_indptr(B+1), v_indptr(B+1), o_indptr(B+1)] => 总共 3*(B+1)
        cu_seqlens_1d = cu_seqlens.view(-1).to(device=q.device, dtype=torch.int32)  # 展平cu_seqlens
        expected = 3 * (B + 1)  # 期望的元素数量
        if int(cu_seqlens_1d.numel()) != expected:  # 检查元素数量
            raise RuntimeError(
                f"packed indptr numel mismatch: got {cu_seqlens_1d.numel()}, expected {expected} (= 3*(B+1))"
            )  # 打包indptr元素数量不匹配

        split = B + 1  # 分割点
        indptr_qk = cu_seqlens_1d[:split].view(split, 1, 1, 1)  # QK索引指针
        indptr_v = cu_seqlens_1d[split : 2 * split].view(split, 1, 1, 1)  # V索引指针
        indptr_o = cu_seqlens_1d[2 * split :].view(split, 1, 1, 1)  # 输出索引指针

        # cuDNN style: (B,1,1,1)  # cuDNN风格：4D序列长度
        seq_lens_4d = seq_lens_1d.view(B, 1, 1, 1)  # 4D序列长度

        # indptr are in ELEMENT offsets (not token offsets)  # indptr是元素偏移（非token偏移）
        token_width_q = int(q.shape[1] * q.shape[2])  # heads * head_dim on this rank  # 每个token的查询元素宽度
        total_elems_q = int(q.numel())  # 查询总元素数

        # check each real sequence fits
        # (skip padded tail where seq_len==0)
        # 检查每个真实序列是否越界（跳过seq_len==0的填充尾部）
        start_elems = indptr_qk.view(-1)[:-1]  # (B,)  # 起始元素
        end_elems = start_elems + seq_lens_1d * token_width_q  # 结束元素
        if (end_elems > total_elems_q).any():  # 检查是否越界
            raise RuntimeError("offset + len out of bounds; packed indptr is wrong")  # 偏移+长度越界，indptr有误

        _, _, head_size = q.shape  # 获取头大小
        scale = softmax_scale if softmax_scale is not None else head_size**-0.5  # 缩放因子

        output, _ = cudnn_batch_prefill_with_kv_cache(  # 调用cuDNN批量预填充注意力
            q,  # 查询
            k,  # 键
            v,  # 值
            scale,  # 缩放因子
            self.workspace_buffer,  # 工作空间
            max_token_per_sequence=max_seqlen,  # 每序列最大token数
            max_sequence_kv=max_seqlen,  # KV最大序列长度
            actual_seq_lens_q=seq_lens_4d,  # 实际查询序列长度
            actual_seq_lens_kv=seq_lens_4d,  # 实际KV序列长度
            causal=False,  # 非因果
            return_lse=True,  # 返回log-softmax值
            batch_offsets_q=indptr_qk,  # 查询批量偏移
            batch_offsets_k=indptr_qk,  # 键批量偏移
            batch_offsets_v=indptr_v,  # 值批量偏移
            batch_offsets_o=indptr_o,  # 输出批量偏移
            is_cuda_graph_compatible=True,  # CUDA图兼容
        )

        if is_reshaped:  # 如果原始输入是4D
            output = rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)  # 重塑回4D

        return output  # 返回输出


class VisionAiterAttention(nn.Module):  # 视觉AMD Aiter注意力模块，用于AMD GPU
    def __init__(  # 初始化方法
        self,
        **kwargs,  # 其他参数
    ):
        if not _is_hip:  # 仅AMD HIP平台可用
            raise Exception("aiter_attn is only available for AMD")  # aiter_attn仅支持AMD
        try:  # 尝试导入aiter
            from aiter import flash_attn_varlen_func as aiter_flash_attn_varlen_func  # AMD Aiter Flash Attention
        except ImportError as e:  # 导入失败
            raise ImportError(
                "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
            ) from e  # aiter是AMD专用内核库，请确保在AMD设备上安装了aiter

        self.flash_attn_varlen_func = aiter_flash_attn_varlen_func  # 保存aiter的flash attn函数
        super().__init__()  # 调用父类初始化

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度
        bsz: int,  # 批量大小
        seq_len: int,  # 序列长度
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)  # 解析序列长度

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)  # 转为int32并移到q设备
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算各序列长度
        max_seqlen = seq_lens.max().item()  # 获取最大序列长度

        return self.flash_attn_varlen_func(  # 调用aiter变长Flash Attention
            q=q,  # 查询
            k=k,  # 键
            v=v,  # 值
            cu_seqlens_q=cu_seqlens,  # 查询累积序列长度
            cu_seqlens_k=cu_seqlens,  # 键累积序列长度
            max_seqlen_q=max_seqlen,  # 查询最大序列长度
            max_seqlen_k=max_seqlen,  # 键最大序列长度
            softmax_scale=softmax_scale,  # softmax缩放
        )


class VisionAscendAttention(nn.Module):  # 视觉华为昇腾注意力模块，使用NPU Flash Attention

    def __init__(  # 初始化方法
        self,
        **kwargs,  # 其他参数
    ):
        if not _is_npu:  # 仅昇腾NPU平台可用
            raise Exception("VisionAscendAttention is only available for ascend npu")  # VisionAscendAttention仅支持昇腾NPU
        super().__init__()  # 调用父类初始化

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度
        bsz: int,  # 批量大小
        seq_len: int,  # 序列长度
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
        Returns:
             [b * s, h, head_size]  # 返回形状为[b*s, h, head_size]的输出
        """
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():  # 如果启用了NPU图模式
            if "output_ws" not in kwargs:  # 如果没有输出工作空间
                raise RuntimeError("output_ws should be prepared for npu-graph mode")  # 需要准备输出工作空间
            output = kwargs["output_ws"]  # 使用预分配的输出工作空间
            seq_len_arg = cu_seqlens  # 直接使用cu_seqlens作为序列长度参数
        else:  # 非图模式
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")  # 在CPU上解析序列长度
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算各序列长度
            if seq_lens.is_npu:  # 如果在NPU上
                seq_lens = seq_lens.to("cpu")  # 移到CPU
            output = torch.empty_like(q)  # 预分配输出
            seq_len_arg = seq_lens.to(torch.int32)  # 转为int32

        _, num_heads, head_size = q.shape  # 获取头数和头大小
        num_kv_heads = k.shape[1]  # 获取KV头数

        scale_value = softmax_scale if softmax_scale is not None else head_size**-0.5  # 缩放因子

        torch_npu._npu_flash_attention_unpad(  # 调用昇腾NPU Flash Attention
            query=q,  # 查询
            key=k,  # 键
            value=v,  # 值
            seq_len=seq_len_arg,  # 序列长度
            scale_value=scale_value,  # 缩放因子
            num_heads=num_heads,  # 头数
            num_kv_heads=num_kv_heads,  # KV头数
            out=output,  # 输出
        )
        return output  # 返回输出


class VisionAMXAttention(nn.Module):  # 视觉Intel AMX注意力模块，用于支持AMX指令集的CPU
    def __init__(  # 初始化方法
        self,
        **kwargs,  # 其他参数
    ):
        if not _is_cpu or not _is_cpu_amx_available:  # 仅支持AMX的CPU可用
            raise Exception(
                "VisionAMXAttention is only available for cpu with amx support"
            )  # VisionAMXAttention仅支持带AMX的CPU
        super().__init__()  # 调用父类初始化

    def forward(  # 前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        cu_seqlens: torch.Tensor | SingletonCache | None,  # 累积序列长度
        bsz: int,  # 批量大小
        seq_len: int,  # 序列长度
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
        Returns:
             [b * s, h, head_size]  # 返回形状为[b*s, h, head_size]的输出
        """
        if cu_seqlens is None:  # 如果cu_seqlens为None
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)  # 根据形状生成
        elif isinstance(cu_seqlens, SingletonCache):  # 如果是单例缓存
            if cu_seqlens.empty():  # 如果缓存为空
                cu_seqlens.set_data(  # 设置缓存
                    _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)  # 根据形状生成
                )
            cu_seqlens = cu_seqlens.get_data()  # 获取缓存数据

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)  # 转为int32并移到q设备
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # 计算各序列长度
        max_seqlen = seq_lens.max().item()  # 获取最大序列长度

        output = flash_attn_varlen_func(  # 调用变长Flash Attention
            q,  # 查询
            k,  # 键
            v,  # 值
            cu_seqlens_q=cu_seqlens,  # 查询累积序列长度
            cu_seqlens_k=cu_seqlens,  # 键累积序列长度
            max_seqlen_q=max_seqlen,  # 查询最大序列长度
            max_seqlen_k=max_seqlen,  # 键最大序列长度
            causal=False,  # 非因果注意力
        )

        return output  # 返回输出


QKV_BACKEND_IMPL = {  # QKV后端实现映射表
    "triton_attn": VisionTritonAttention,  # Triton注意力
    "sdpa": VisionSdpaAttention,  # SDPA注意力
    "fa3": VisionFlash3Attention,  # Flash Attention 3
    "fa4": VisionFlash4Attention,  # Flash Attention 4
    "flashinfer_cudnn": VisionFlashInferAttention,  # FlashInfer cuDNN
    "ascend_attn": VisionAscendAttention,  # 昇腾注意力
    "aiter_attn": VisionAiterAttention,  # AMD Aiter注意力
    "amx_attn": VisionAMXAttention,  # Intel AMX注意力
}


class VisionAttention(nn.Module):  # 视觉注意力主模块，多模态Transformer中无缓存的多头注意力
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.
        无缓存的多头注意力，主要用于多模态Transformer。


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        # use_qkv_parallel (bool, 可选): 如果为True，使用QKV并行注意力。
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision
            # 如果为True，softmax将以单精度执行；否则以半精度执行

    """

    def __init__(  # 初始化方法
        self,
        embed_dim: int,  # 嵌入维度
        num_heads: int,  # 注意力头数
        projection_size: int,  # 投影大小
        use_qkv_parallel: bool,  # 是否使用QKV并行
        num_kv_heads: Optional[int] = None,  # KV头数
        head_dim: Optional[int] = None,  # 头维度
        qkv_backend: Optional[str] = None,  # QKV后端名称
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        dropout: float = 0.0,  # dropout比率
        softmax_in_single_precision: bool = False,  # 是否使用单精度softmax
        softmax_scale: Optional[float] = None,  # softmax缩放因子
        flatten_batch: bool = False,  # 是否展平批量
        prefix: str = "",  # 前缀
        proj_bias: bool = True,  # 投影是否使用偏置
        num_dummy_heads: int = 0,  # 虚拟头数（用于TP对齐）
        qkv_bias: bool = True,  # QKV是否使用偏置
        qk_normalization: bool = False,  # 是否进行QK归一化
        qk_normalization_by_head_size: bool = False,  # 是否按头大小进行QK归一化
        layer_norm_eps: float = 1e-06,  # LayerNorm epsilon
        customized_position_embedding_applier: Callable[  # 自定义位置编码应用器
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        use_data_parallel: bool = False,  # 是否使用数据并行
        use_dp_attention_reduce: bool = False,  # 是否使用DP注意力归约
        aux_stream: Optional[torch.cuda.Stream] = None,  # 辅助CUDA流
        workspace_buffer: Optional[torch.Tensor] = None,  # 工作空间缓冲区
        use_sink: bool = False,  # 是否使用sink注意力
        window_size: Tuple[int, int] = (-1, -1),  # 窗口大小
        **kwargs,  # 其他参数
    ):
        super().__init__()  # 调用父类初始化
        if head_dim is None and "head_size" in kwargs:  # 兼容旧参数名
            head_dim = kwargs.pop("head_size")  # 弹出head_size
            warnings.warn(  # 发出弃用警告
                "VisionAttention(head_size=...) is deprecated; use head_dim=...",
                DeprecationWarning,  # 弃用警告类型
                stacklevel=2,  # 堆栈层级
            )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()  # 张量并行大小
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()  # 张量并行秩
        self.dropout = dropout  # dropout率
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads  # KV头数默认与Q头数相同
        self.head_size = head_dim if head_dim is not None else embed_dim // num_heads  # 头大小
        self.hidden_size_per_attention_head = dist_utils.divide(  # 每个注意力头的隐藏大小
            projection_size, num_heads  # 投影大小/头数
        )
        self.num_attention_heads_per_partition = dist_utils.divide(  # 每个分区的注意力头数
            num_dummy_heads + num_heads, self.tp_size  # (虚拟头+真实头)/TP大小
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(  # 每个分区的KV头数
            num_dummy_heads + num_kv_heads, self.tp_size  # (虚拟头+KV头)/TP大小
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size  # 查询维度
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size  # KV维度

        self.qk_normalization = qk_normalization  # QK归一化标志
        self.qk_normalization_by_head_size = qk_normalization_by_head_size  # 按头大小归一化标志

        # Additional dummy heads are used to enable TP for common GPU counts.
        # 附加虚拟头用于在常见GPU数量上启用张量并行。
        self.dummy_dim = (num_dummy_heads + num_heads) * self.head_size  # 虚拟维度

        if self.qk_normalization:  # 如果启用QK归一化
            self.q_norm, self.k_norm = self._init_qk_norm(  # 初始化QK归一化
                self.dummy_dim, layer_norm_eps, embed_dim  # 维度、eps、隐藏大小
            )

        elif self.qk_normalization_by_head_size:  # 如果按头大小归一化
            self.q_norm, self.k_norm = self._init_qk_norm(  # 初始化QK归一化
                self.head_size, layer_norm_eps  # 头维度、eps
            )

        # Select attention backend via a unified method  # 通过统一方法选择注意力后端
        _passed_backend = qkv_backend  # 传入的后端
        qkv_backend = self._determine_attention_backend(_passed_backend)  # 确定注意力后端
        if (  # 如果没有设置多模态注意力后端且没有传入后端
            get_global_server_args().mm_attention_backend is None
            and _passed_backend is None
        ):
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")  # 打印默认后端
        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")  # 打印使用的后端

        self.customized_position_embedding_applier = (  # 自定义位置编码应用器
            customized_position_embedding_applier
        )
        self.softmax_scale = softmax_scale  # softmax缩放因子
        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](  # 实例化选定的后端
            head_dim=self.head_size,  # 头维度
            num_heads=self.num_attention_heads_per_partition,  # 头数
            num_kv_heads=self.num_attention_kv_heads_per_partition,  # KV头数
            dropout=dropout,  # dropout率
            flatten_batch=flatten_batch,  # 是否展平批量
            softmax_in_single_precision=softmax_in_single_precision,  # 单精度softmax
            softmax_scale=softmax_scale,  # softmax缩放
            use_data_parallel=use_data_parallel,  # 数据并行
            workspace_buffer=workspace_buffer,  # 工作空间缓冲区
        )

        self.use_qkv_parallel = use_qkv_parallel  # QKV并行标志
        if use_qkv_parallel:  # 如果使用QKV并行
            self.qkv_proj = QKVParallelLinear(  # QKV并行线性层
                hidden_size=embed_dim,  # 隐藏大小
                head_size=self.head_size,  # 头大小
                total_num_heads=num_dummy_heads + num_heads,  # 总头数
                total_num_kv_heads=num_dummy_heads + num_kv_heads,  # 总KV头数
                bias=qkv_bias,  # 偏置
                quant_config=quant_config,  # 量化配置
                tp_rank=self.tp_rank,  # TP秩
                tp_size=self.tp_size,  # TP大小
                prefix=add_prefix("qkv_proj", prefix),  # 前缀
            )
        else:  # 不使用QKV并行
            self.qkv_proj = ColumnParallelLinear(  # 列并行线性层
                input_size=embed_dim,  # 输入大小
                output_size=3 * self.dummy_dim,  # 输出大小（3倍虚拟维度）
                bias=qkv_bias,  # 偏置
                quant_config=quant_config,  # 量化配置
                tp_rank=self.tp_rank,  # TP秩
                tp_size=self.tp_size,  # TP大小
                prefix=add_prefix("qkv_proj", prefix),  # 前缀
            )
        self.proj = RowParallelLinear(  # 输出投影，行并行线性层
            input_size=self.dummy_dim,  # 输入大小
            output_size=embed_dim,  # 输出大小
            bias=proj_bias,  # 偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=self.tp_rank,  # TP秩
            tp_size=self.tp_size,  # TP大小
            prefix=add_prefix("proj", prefix),  # 前缀
            use_dp_attention_reduce=use_dp_attention_reduce,  # DP注意力归约
        )

        self.workspace_buffer = workspace_buffer  # 工作空间缓冲区
        self.aux_stream = aux_stream  # 辅助CUDA流
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()] if aux_stream else []  # CUDA事件

        self.window_size = window_size  # 窗口大小
        if use_sink:  # 如果使用sink
            # Allocate the full (unsharded) sink tensor for weight loading;
            # only the local TP slice is used in forward.
            # 分配完整的（未分片的）sink张量用于权重加载；前向传播中仅使用本地TP切片。
            self.sinks = nn.Parameter(  # sink参数
                torch.empty(  # 空张量
                    self.num_attention_heads_per_partition * self.tp_size,  # 全量头数
                    dtype=torch.bfloat16,  # bfloat16精度
                ),
                requires_grad=False,  # 不需要梯度
            )
        else:  # 不使用sink
            self.sinks = None  # sinks为None

    def _init_qk_norm(  # 初始化QK归一化层
        self, norm_dim: int, eps: float, var_hidden_size: Optional[int] = None  # 归一化维度、eps、可变隐藏大小
    ):
        norm_kwargs = (  # 归一化额外参数
            dict(
                weight_dtype=torch.float32,  # 权重使用float32
                cast_x_before_out_mul=True,  # 在输出乘法前转换x
            )
            if get_global_server_args().rl_on_policy_target is not None  # 如果有RL目标
            else {}  # 否则无额外参数
        )
        q_norm = RMSNorm(  # 查询归一化层
            norm_dim,  # 归一化维度
            eps=eps,  # epsilon
            var_hidden_size=var_hidden_size,  # 可变隐藏大小
            **norm_kwargs,  # 额外参数
        )
        k_norm = RMSNorm(  # 键归一化层
            norm_dim,  # 归一化维度
            eps=eps,  # epsilon
            var_hidden_size=var_hidden_size,  # 可变隐藏大小
            **norm_kwargs,  # 额外参数
        )
        return q_norm, k_norm  # 返回Q和K的归一化层

    def _determine_attention_backend(self, passed_backend: Optional[str]) -> str:  # 确定多模态注意力后端字符串
        """Decide the multimodal attention backend string.
        确定多模态注意力后端字符串。

        Priority: server args override > constructor arg > platform default.
        优先级：服务器参数覆盖 > 构造函数参数 > 平台默认。

        Platform defaults:
        平台默认值：
        - CUDA (Hopper SM90): "fa3"
        - CUDA (Blackwell SM100): "fa4"
        - CUDA (other): "triton_attn"
        - Non-CUDA: "sdpa"
        """
        override_backend = get_global_server_args().mm_attention_backend  # 获取服务器参数覆盖
        if override_backend is not None:  # 如果有覆盖
            backend = override_backend  # 使用覆盖值
        elif passed_backend is not None:  # 如果有传入后端
            backend = passed_backend  # 使用传入值
        elif is_cuda():  # CUDA平台
            major, minor = get_device_capability()  # 获取计算能力
            if major == 9:  # Hopper架构
                backend = "fa3"  # 使用FA3
            elif major == 10 and minor != 3:  # Blackwell架构（非SM103）
                backend = "fa4"  # 使用FA4
            else:  # 其他CUDA
                backend = "triton_attn"  # 使用Triton
        elif _is_musa:  # 摩尔线程平台
            if get_device_capability() >= (3, 1):  # 计算能力>=3.1
                backend = "fa3"  # 使用FA3
            else:  # 较低计算能力
                backend = "triton_attn"  # 使用Triton
        elif _is_hip:  # AMD HIP平台
            if get_device_capability() >= (9, 4) and _use_aiter:  # 计算能力>=9.4且启用aiter
                backend = "aiter_attn"  # 使用Aiter
            else:  # 其他AMD
                backend = "triton_attn"  # 使用Triton
        elif _is_cpu and _is_cpu_amx_available:  # CPU且支持AMX
            backend = "amx_attn"  # 使用AMX
        elif _is_xpu:  # Intel XPU平台
            backend = "triton_attn"  # 使用Triton
        else:  # 其他平台
            backend = "sdpa"  # 使用SDPA
        if backend == "fa3" and is_blackwell_supported():  # FA3不支持Blackwell
            raise ValueError("The 'fa3' backend is not supported on Blackwell GPUs")  # FA3后端不支持Blackwell GPU

        return backend  # 返回后端名称

    def _apply_qk_norm_head_size(self, q: torch.Tensor, k: torch.Tensor):  # 按头大小应用QK归一化（用于GLM-OCR视觉注意力）
        """apply qk norm for GLM-OCR vit attn
        为GLM-OCR视觉注意力应用QK归一化"""
        q_by_head = q.reshape(-1, self.head_size)  # 按头重塑Q
        q_by_head = self.q_norm(q_by_head)  # Q归一化
        k_by_head = k.reshape(-1, self.head_size)  # 按头重塑K
        k_by_head = self.k_norm(k_by_head)  # K归一化
        q = q_by_head.view(q.shape)  # 恢复Q原始形状
        k = k_by_head.view(k.shape)  # 恢复K原始形状
        return q, k  # 返回归一化后的Q和K

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):  # 应用QK归一化（用于InternVL视觉注意力）
        """apply qk norm for internvl vit attn
        为InternVL视觉注意力应用QK归一化"""

        def q_l2norm():  # Q的L2归一化
            q_ = q.flatten(1, 2)  # 展平头和头维度
            if self.tp_size > 1:  # 如果张量并行
                q_ = tensor_model_parallel_all_gather(q_.contiguous())  # 全收集
            q_ = self.q_norm(q_)  # Q归一化
            if self.tp_size > 1:  # 如果张量并行
                splitter = partial(  # 分割器
                    split_tensor_along_last_dim, num_partitions=self.tp_size  # 沿最后一维分割
                )
                q_ = splitter(q_)[self.tp_rank]  # 取当前分片
            q_ = q_.unflatten(-1, (-1, self.head_size))  # 反展平头维度
            return q_  # 返回归一化后的Q

        def k_l2norm():  # K的L2归一化
            k_ = k.flatten(1, 2)  # 展平头和头维度
            if self.tp_size > 1:  # 如果张量并行
                k_ = tensor_model_parallel_all_gather(k_.contiguous())  # 全收集
            k_ = self.k_norm(k_)  # K归一化
            if self.tp_size > 1:  # 如果张量并行
                splitter = partial(  # 分割器
                    split_tensor_along_last_dim, num_partitions=self.tp_size  # 沿最后一维分割
                )
                k_ = splitter(k_)[self.tp_rank]  # 取当前分片
            k_ = k_.unflatten(-1, (-1, self.head_size))  # 反展平头维度
            return k_  # 返回归一化后的K

        with with_multi_stream(True):  # 启用多流
            q, k = maybe_execute_in_parallel(  # 可能并行执行Q和K归一化
                q_l2norm,  # Q归一化函数
                k_l2norm,  # K归一化函数
                self.ln_events,  # CUDA事件
                self.aux_stream,  # 辅助流
            )
        return q, k  # 返回归一化后的Q和K

    def forward(  # 前向传播
        self,
        x: torch.Tensor,  # 输入张量
        cu_seqlens: Optional[torch.Tensor] = None,  # 累积序列长度
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 位置嵌入(cos, sin)
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,  # 旋转位置编码余弦
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,  # 旋转位置编码正弦
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        full_attn: bool = True,  # 是否使用全注意力（非滑动窗口）
        **kwargs,  # 其他参数
    ) -> torch.Tensor:  # 返回输出张量
        r"""
        Args:
            x: [b, s, embed_dim]  # 输入张量，形状为[b, s, embed_dim]
            cu_seqlens: [b]  # 累积序列长度，形状为[b]
        Returns:
             [s, b, head * head_size]  # 返回形状为[s, b, head*head_size]的输出
        """
        if x.dim() == 2:  # 如果输入是2D
            x = x.unsqueeze(0)  # 增加批量维度
        assert x.dim() == 3, x.shape  # 输入必须是3D
        if (  # 如果有RL目标且有位置嵌入
            get_global_server_args().rl_on_policy_target is not None
            and position_embeddings is not None
        ):
            assert isinstance(position_embeddings, tuple), (  # 位置嵌入必须是元组
                "expected position_embeddings to be a tuple of two tensors,\n"
                f"but got {type(position_embeddings)}, change if needed"
            )  # 期望位置嵌入为两个张量的元组
            position_embeddings = tuple(p.to(x.dtype) for p in position_embeddings)  # 转换精度
        x_shape = x.shape  # 保存形状
        bsz, s, _ = x_shape  # 批量大小、序列长度、嵌入维度
        head = self.num_attention_heads_per_partition  # 注意力头数
        kv_head = self.num_attention_kv_heads_per_partition  # KV头数

        attn_output_ws = kwargs["output_ws"] if "output_ws" in kwargs else None  # 输出工作空间
        max_seqlen = kwargs["max_seqlen"] if "max_seqlen" in kwargs else None  # 最大序列长度
        sequence_lengths = (  # 序列长度
            kwargs["sequence_lengths"] if "sequence_lengths" in kwargs else None  # 从kwargs获取
        )
        if self.use_qkv_parallel:  # 如果使用QKV并行
            # [b, s, embed_dim] --> [b, s, embed_dim]  # 投影后形状不变
            qkv, _ = self.qkv_proj(x)  # QKV投影
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV

            # [b, s, embed_dim] --> [b * s, head, head_size]  # 重塑为3D
            q = q.reshape(bsz * s, head, -1)  # 重塑Q
            k = k.reshape(bsz * s, kv_head, -1)  # 重塑K
            v = v.reshape(bsz * s, kv_head, -1)  # 重塑V
        else:  # 不使用QKV并行
            # [b, s, embed_dim] --> [s, b, embed_dim]  # 转换为[s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")  # 重排x
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]  # QKV投影
            qkv, _ = self.qkv_proj(x)  # QKV投影

            # [s, b, head, head_dim_sum]  # 重塑QKV
            new_x_shape = qkv.size()[:-1] + (  # 新形状
                head,  # 头数
                self.q_size + 2 * self.kv_size,  # Q + 2*KV维度
            )
            qkv = qkv.view(*new_x_shape)  # 应用新形状

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]  # 拆分为Q、K、V
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 按维度拆分

            # [s, b, head, head_size] --> [b, s, head, head_size]  # 转换批量维度
            q, k, v = [rearrange(x, "s b ... -> b s ...") for x in (q, k, v)]  # 重排QKV

        if not (_is_cpu and _is_cpu_amx_available):  # 非CPU AMX平台
            q = q.contiguous()  # 确保Q连续
            k = k.contiguous()  # 确保K连续
            v = v.contiguous()  # 确保V连续
        if self.qk_normalization_by_head_size:  # 如果按头大小归一化
            q, k = self._apply_qk_norm_head_size(q, k)  # 应用QK归一化

        cos = None  # 旋转位置编码余弦
        sin = None  # 旋转位置编码正弦

        if position_embeddings is not None:  # 如果有位置嵌入
            if self.customized_position_embedding_applier is not None:  # 如果有自定义应用器
                q, k = self.customized_position_embedding_applier(  # 应用自定义位置编码
                    q, k, position_embeddings, x_shape  # 传入Q、K、位置嵌入和形状
                )
            else:  # 使用标准位置编码
                cos, sin = position_embeddings  # 解包cos和sin
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:  # 如果有旋转位置编码
            cos = rotary_pos_emb_cos  # 使用传入的cos
            sin = rotary_pos_emb_sin  # 使用传入的sin

        if cos is not None and sin is not None:  # 如果有旋转位置编码
            original_q_shape = q.shape  # 保存Q原始形状
            original_k_shape = k.shape  # 保存K原始形状

            # [total_tokens, head, head_size] for q / [total_tokens, kv_head, head_size] for k
            # Q的形状为[total_tokens, head, head_size]，K为[total_tokens, kv_head, head_size]
            q = q.view(-1, head, self.head_size)  # 重塑Q
            k = k.view(-1, kv_head, self.head_size)  # 重塑K

            if cos.size(-1) * 2 == self.head_size:  # 如果cos维度是头大小的一半
                cos = torch.cat([cos, cos], dim=-1)  # 复制cos以匹配头大小
                sin = torch.cat([sin, sin], dim=-1)  # 复制sin以匹配头大小

            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # 应用旋转位置编码
            q = q.view(original_q_shape)  # 恢复Q原始形状
            k = k.view(original_k_shape)  # 恢复K原始形状

        if q.dim() == 4:  # 如果Q是4D
            # [b, s, head, head_size] --> [b * s, head, head_size]  # 展平批量维度
            q = rearrange(q, "b s ... -> (b s) ...")  # 重排Q
        if k.dim() == 4:  # 如果K是4D
            # [b, s, head, head_size] --> [b * s, head, head_size]  # 展平批量维度
            k = rearrange(k, "b s ... -> (b s) ...")  # 重排K
        if v.dim() == 4:  # 如果V是4D
            # [b, s, head, head_size] --> [b * s, head, head_size]  # 展平批量维度
            v = rearrange(v, "b s ... -> (b s) ...")  # 重排V

        assert q.dim() == 3, q.dim()  # Q必须是3D
        assert k.dim() == 3, k.dim()  # K必须是3D
        assert v.dim() == 3, v.dim()  # V必须是3D

        # internvl  # InternVL模型
        if self.qk_normalization and not self.qk_normalization_by_head_size:  # 如果启用QK归一化（非按头大小）
            # jit kernel  # JIT内核
            if can_use_jit_qk_norm(self.head_size, q.dtype):  # 如果可以使用JIT QK归一化

                # q: [tokens, head, head_size]  ->  [tokens, embed_dim]
                # q: [tokens, head, head_size] -> [tokens, embed_dim]
                head_dim_for_norm = head * self.head_size  # 归一化维度

                q, k = apply_qk_norm(  # 应用JIT QK归一化
                    q=q,  # 查询
                    k=k,  # 键
                    q_norm=self.q_norm,  # Q归一化层
                    k_norm=self.k_norm,  # K归一化层
                    head_dim=head_dim_for_norm,  # 归一化维度
                    alt_stream=self.aux_stream,  # 辅助流
                )

            else:  # 不能使用JIT
                q, k = self._apply_qk_norm(q, k)  # 使用Python实现

        if full_attn or self.sinks is None:  # 全注意力或无sink
            effective_window_size = (-1, -1)  # 无窗口限制
            s_aux = None  # 无辅助参数
        else:  # 使用sink注意力
            effective_window_size = self.window_size  # 使用配置的窗口大小
            q_head_start = self.tp_rank * self.num_attention_heads_per_partition  # 当前TP分片的头起始
            q_head_end = (self.tp_rank + 1) * self.num_attention_heads_per_partition  # 当前TP分片的头结束
            s_aux = self.sinks[q_head_start:q_head_end]  # 取当前分片的sink

        output = self.qkv_backend.forward(  # 调用后端前向
            q=q,  # 查询
            k=k,  # 键
            v=v,  # 值
            bsz=bsz,  # 批量大小
            seq_len=s,  # 序列长度
            cu_seqlens=cu_seqlens,  # 累积序列长度
            attention_mask=attention_mask,  # 注意力掩码
            sequence_lengths=sequence_lengths,  # 序列长度
            max_seqlen=max_seqlen,  # 最大序列长度
            output_ws=attn_output_ws,  # 输出工作空间
            softmax_scale=self.softmax_scale,  # softmax缩放
            window_size=effective_window_size,  # 窗口大小
            s_aux=s_aux,  # sink辅助参数
        )

        assert output.dim() == 3, output.shape  # 输出必须是3D

        if self.use_qkv_parallel:  # 如果使用QKV并行
            # [b * s, h, head_size] --> [b, s, h * head_size]  # 重塑输出
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)  # 重排

            # [b, s, h * head_size] --> [b, s, h * head_size]  # 输出投影
            output, _ = self.proj(output)  # 行并行投影
        else:  # 不使用QKV并行
            # [b * s, h, head_size] --> [s, b, h * head_size]  # 重塑输出
            context_layer = rearrange(  # 重排
                output, "(b s) h d -> s b (h d)", b=bsz, s=s  # 转为[s, b, h*d]
            ).contiguous()  # 确保连续

            # [s, b, h * head_size] --> [s, b, h * head_size]  # 输出投影
            output, _ = self.proj(context_layer)  # 行并行投影

            # [s, b, h * head_size] --> [b, s, h * head_size]  # 转换批量维度
            output = output.view(bsz, s, -1)  # 重塑为[b, s, h*d]

        return output  # 返回输出
