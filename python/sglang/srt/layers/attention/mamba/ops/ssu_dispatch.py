# SSU（选择性状态更新）调度模块 - 实现Mamba2模型中选择性状态更新的后端调度
# 提供Triton和FlashInfer两种后端实现的统一接口，支持运行时后端切换

from __future__ import annotations  # 启用延迟注解评估

import logging  # 导入日志模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch深度学习框架

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.server_args import ServerArgs  # 导入服务器参数类型

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class MambaSSUBackend(ABC):  # Mamba选择性状态更新后端抽象基类
    @property  # 属性装饰器
    @abstractmethod  # 抽象方法装饰器
    def name(self) -> str:  # 后端名称属性
        """Human-readable name used for logging."""
        """用于日志记录的人类可读名称。"""

    @abstractmethod  # 抽象方法装饰器
    def __call__(  # 调用接口，执行选择性状态更新
        self,
        state: torch.Tensor,  # SSM状态张量 (batch, nheads, dim, dstate)
        x: torch.Tensor,  # 输入张量
        dt: torch.Tensor,  # delta时间张量
        A: torch.Tensor,  # A矩阵
        B: torch.Tensor,  # B矩阵
        C: torch.Tensor,  # C矩阵
        D: torch.Tensor | None = None,  # D向量，可选
        z: torch.Tensor | None = None,  # z门控张量，可选
        dt_bias: torch.Tensor | None = None,  # dt偏置，可选
        dt_softplus: bool = False,  # 是否对dt应用softplus
        state_batch_indices: torch.Tensor | None = None,  # 状态批次索引，可选
        pad_slot_id: int = -1,  # 填充槽ID，默认-1
        out: torch.Tensor | None = None,  # 输出张量，可选
        disable_state_update: bool = False,  # 是否禁用状态更新
        intermediate_states_buffer: torch.Tensor | None = None,  # 中间状态缓冲区，可选
        cache_steps: int | None = None,  # 缓存步数，可选
        retrieve_parent_token: torch.Tensor | None = None,  # 父token索引张量（EAGLE树注意力），可选
        intermediate_state_indices: torch.Tensor | None = None,  # 中间状态缓冲区索引，可选
    ) -> None: ...  # 无返回值


class TritonSSUBackend(MambaSSUBackend):  # 基于Triton的选择性状态更新后端
    """Triton-based selective-state-update backend."""
    """基于Triton的选择性状态更新后端。"""

    def __init__(self) -> None:  # 初始化方法
        from sglang.srt.layers.attention.mamba.ops.mamba_ssm import (  # 延迟导入Triton内核
            selective_state_update,  # 选择性状态更新函数
        )

        self._kernel = selective_state_update  # 保存内核函数引用

    @property  # 属性装饰器
    def name(self) -> str:  # 后端名称
        return "triton"  # 返回"triton"

    def __call__(  # 执行选择性状态更新
        self,
        state: torch.Tensor,  # SSM状态张量
        x: torch.Tensor,  # 输入张量
        dt: torch.Tensor,  # delta时间张量
        A: torch.Tensor,  # A矩阵
        B: torch.Tensor,  # B矩阵
        C: torch.Tensor,  # C矩阵
        D: torch.Tensor | None = None,  # D向量，可选
        z: torch.Tensor | None = None,  # z门控张量，可选
        dt_bias: torch.Tensor | None = None,  # dt偏置，可选
        dt_softplus: bool = False,  # 是否对dt应用softplus
        state_batch_indices: torch.Tensor | None = None,  # 状态批次索引，可选
        pad_slot_id: int = -1,  # 填充槽ID
        out: torch.Tensor | None = None,  # 输出张量，可选
        disable_state_update: bool = False,  # 是否禁用状态更新
        intermediate_states_buffer: torch.Tensor | None = None,  # 中间状态缓冲区，可选
        cache_steps: int | None = None,  # 缓存步数，可选
        retrieve_parent_token: torch.Tensor | None = None,  # 父token索引，可选
        intermediate_state_indices: torch.Tensor | None = None,  # 中间状态索引，可选
    ) -> None:
        self._kernel(  # 调用Triton内核
            state,  # SSM状态
            x,  # 输入
            dt,  # delta时间
            A,  # A矩阵
            B,  # B矩阵
            C,  # C矩阵
            D=D,  # D向量
            z=z,  # z门控
            dt_bias=dt_bias,  # dt偏置
            dt_softplus=dt_softplus,  # softplus标志
            state_batch_indices=state_batch_indices,  # 状态批次索引
            pad_slot_id=pad_slot_id,  # 填充槽ID
            out=out,  # 输出
            disable_state_update=disable_state_update,  # 禁用状态更新标志
            intermediate_states_buffer=intermediate_states_buffer,  # 中间状态缓冲区
            cache_steps=cache_steps,  # 缓存步数
            retrieve_parent_token=retrieve_parent_token,  # 父token索引
            intermediate_state_indices=intermediate_state_indices,  # 中间状态索引
        )


class FlashInferSSUBackend(MambaSSUBackend):  # 基于FlashInfer的选择性状态更新后端
    """FlashInfer-based selective-state-update backend."""
    """基于FlashInfer的选择性状态更新后端。"""

    def __init__(self) -> None:  # 初始化方法
        from flashinfer.mamba import selective_state_update  # 延迟导入FlashInfer内核

        self._kernel = selective_state_update  # 保存内核函数引用

    @property  # 属性装饰器
    def name(self) -> str:  # 后端名称
        return "flashinfer"  # 返回"flashinfer"

    def __call__(  # 执行选择性状态更新
        self,
        state: torch.Tensor,  # SSM状态张量
        x: torch.Tensor,  # 输入张量
        dt: torch.Tensor,  # delta时间张量
        A: torch.Tensor,  # A矩阵
        B: torch.Tensor,  # B矩阵
        C: torch.Tensor,  # C矩阵
        D: torch.Tensor | None = None,  # D向量，可选
        z: torch.Tensor | None = None,  # z门控张量，可选
        dt_bias: torch.Tensor | None = None,  # dt偏置，可选
        dt_softplus: bool = False,  # 是否对dt应用softplus
        state_batch_indices: torch.Tensor | None = None,  # 状态批次索引，可选
        pad_slot_id: int = -1,  # 填充槽ID
        out: torch.Tensor | None = None,  # 输出张量，可选
        disable_state_update: bool = False,  # 是否禁用状态更新
        intermediate_states_buffer: torch.Tensor | None = None,  # 中间状态缓冲区，可选
        cache_steps: int | None = None,  # 缓存步数，可选
        retrieve_parent_token: torch.Tensor | None = None,  # 父token索引，可选
        intermediate_state_indices: torch.Tensor | None = None,  # 中间状态索引，可选
    ) -> None:
        if retrieve_parent_token is not None:  # 如果请求了父token检索
            raise ValueError(  # 抛出值错误
                "FlashInfer backend does not support retrieve_parent_token. "
                "Use --mamba-backend triton for EAGLE tree attention."
                # "FlashInfer后端不支持retrieve_parent_token。"
                # "EAGLE树注意力请使用 --mamba-backend triton。"
            )
        # FlashInfer expects cache_steps as an int (0 when unused).
        # FlashInfer期望cache_steps为整数（未使用时为0）。
        self._kernel(  # 调用FlashInfer内核
            state,  # SSM状态
            x,  # 输入
            dt,  # delta时间
            A,  # A矩阵
            B,  # B矩阵
            C,  # C矩阵
            D=D,  # D向量
            z=z,  # z门控
            dt_bias=dt_bias,  # dt偏置
            dt_softplus=dt_softplus,  # softplus标志
            state_batch_indices=state_batch_indices,  # 状态批次索引
            pad_slot_id=pad_slot_id,  # 填充槽ID
            out=out,  # 输出
            disable_state_update=disable_state_update,  # 禁用状态更新标志
            intermediate_states_buffer=intermediate_states_buffer,  # 中间状态缓冲区
            cache_steps=0 if cache_steps is None else cache_steps,  # 未使用时为0，否则使用指定值
            intermediate_state_indices=intermediate_state_indices,  # 中间状态索引
        )


_BACKEND_REGISTRY: dict[str, type[MambaSSUBackend]] = {  # 后端注册表
    "triton": TritonSSUBackend,  # Triton后端
    "flashinfer": FlashInferSSUBackend,  # FlashInfer后端
}

_mamba_ssu_backend: MambaSSUBackend | None = None  # 全局后端实例，初始化为None


def initialize_mamba_selective_state_update_backend(server_args: ServerArgs) -> None:  # 初始化选择性状态更新后端
    """Instantiate the selective-state-update backend from server config.
    """根据服务器配置实例化选择性状态更新后端。

    This should be called once during scheduler initialization.
    此函数应在调度器初始化期间调用一次。

    Args:
    参数:
        server_args: Server arguments containing ``mamba_backend`` setting.
        server_args: 包含 ``mamba_backend`` 设置的服务器参数。

    Raises:
    异常:
        ValueError: If the requested backend is unavailable or cannot be imported.
        ValueError: 如果请求的后端不可用或无法导入。
    """
    global _mamba_ssu_backend  # 声明使用全局变量

    requested = server_args.mamba_backend or "triton"  # 获取请求的后端名称，默认为"triton"

    backend_cls = _BACKEND_REGISTRY.get(requested)  # 从注册表查找后端类
    if backend_cls is None:  # 如果后端不存在
        raise ValueError(  # 抛出值错误
            f"Unknown mamba backend '{requested}'. "  # 未知后端
            f"Available backends: {list(_BACKEND_REGISTRY.keys())}"  # 列出可用后端
        )

    try:  # 尝试初始化后端
        _mamba_ssu_backend = backend_cls()  # 创建后端实例
    except ImportError:  # 如果导入失败
        raise ValueError(  # 抛出值错误
            f"Mamba backend '{requested}' requested but its dependencies are not "  # 请求的后端依赖不可用
            f"available. Install the required package or use a different "  # 请安装所需包或使用其他
            f"--mamba-backend value."  # --mamba-backend 值
        )

    logger.debug(  # 记录调试信息
        "Mamba selective_state_update backend initialized: %s",
        _mamba_ssu_backend.name,  # 已初始化的后端名称
    )


def selective_state_update(  # 选择性状态更新的统一调度函数
    state: torch.Tensor,  # SSM状态张量 (batch, nheads, dim, dstate)
    x: torch.Tensor,  # 输入张量
    dt: torch.Tensor,  # delta时间张量
    A: torch.Tensor,  # A矩阵
    B: torch.Tensor,  # B矩阵
    C: torch.Tensor,  # C矩阵
    D: torch.Tensor | None = None,  # D向量，可选
    z: torch.Tensor | None = None,  # z门控张量，可选
    dt_bias: torch.Tensor | None = None,  # dt偏置，可选
    dt_softplus: bool = False,  # 是否对dt应用softplus
    state_batch_indices: torch.Tensor | None = None,  # 状态批次索引，可选
    pad_slot_id: int = -1,  # 填充槽ID，默认-1
    out: torch.Tensor | None = None,  # 输出张量，可选
    disable_state_update: bool = False,  # 是否禁用状态更新（用于投机验证）
    intermediate_states_buffer: torch.Tensor | None = None,  # 中间状态缓冲区，可选
    cache_steps: int | None = None,  # 缓冲区中的总步数，可选
    retrieve_parent_token: torch.Tensor | None = None,  # (batch, T)父token索引张量，用于EAGLE树注意力
    intermediate_state_indices: torch.Tensor | None = None,  # (batch,)中间状态缓冲区操作索引，可选
) -> None:
    """Dispatch selective-state-update to the configured backend.
    """将选择性状态更新调度到已配置的后端。

    This function provides a unified interface regardless of the underlying
    backend. Backend-specific argument adaptation is handled inside each
    :class:`MambaSSUBackend` subclass.
    此函数提供统一接口，无论底层使用何种后端。后端特定的参数适配在各
    :class:`MambaSSUBackend` 子类内部处理。

    Args:
    参数:
        state: SSM state tensor (batch, nheads, dim, dstate)
        state: SSM状态张量 (批次, 头数, 维度, 状态维度)
        x: Input tensor
        x: 输入张量
        dt: Delta time tensor
        dt: delta时间张量
        A: A matrix
        A: A矩阵
        B: B matrix
        B: B矩阵
        C: C matrix
        C: C矩阵
        D: Optional D vector
        D: 可选D向量
        z: Optional z tensor for gating
        z: 可选z门控张量
        dt_bias: Optional dt bias
        dt_bias: 可选dt偏置
        dt_softplus: Whether to apply softplus to dt
        dt_softplus: 是否对dt应用softplus
        state_batch_indices: Optional batch indices for state
        state_batch_indices: 可选状态批次索引
        out: Preallocated output tensor (in-place updated)
        out: 预分配输出张量（原地更新）
        disable_state_update: If True, don't write back to state (for speculative verify)
        disable_state_update: 为True时不写回状态（用于投机验证）
        intermediate_states_buffer: Buffer to cache intermediate states
        intermediate_states_buffer: 缓存中间状态的缓冲区
        cache_steps: Total number of steps in the buffer
        cache_steps: 缓冲区中的总步数
        retrieve_parent_token: (batch, T) tensor of parent token indices for EAGLE tree attention
        retrieve_parent_token: (batch, T)父token索引张量，用于EAGLE树注意力
        intermediate_state_indices: (batch,) tensor of indices for intermediate_states_buffer operations.
            If provided, uses these indices instead of state_batch_indices for the buffer.
        intermediate_state_indices: (batch,)中间状态缓冲区操作的索引张量。
            如果提供，使用这些索引代替state_batch_indices进行缓冲区操作。
    """
    assert _mamba_ssu_backend is not None, (  # 断言后端已初始化
        "Mamba selective_state_update backend not initialized. "
        "Call initialize_mamba_selective_state_update_backend() first."
        # "Mamba selective_state_update后端未初始化。"
        # "请先调用 initialize_mamba_selective_state_update_backend()。"
    )

    _mamba_ssu_backend(  # 调用已配置的后端
        state,  # SSM状态
        x,  # 输入
        dt,  # delta时间
        A,  # A矩阵
        B,  # B矩阵
        C,  # C矩阵
        D=D,  # D向量
        z=z,  # z门控
        dt_bias=dt_bias,  # dt偏置
        dt_softplus=dt_softplus,  # softplus标志
        state_batch_indices=state_batch_indices,  # 状态批次索引
        pad_slot_id=pad_slot_id,  # 填充槽ID
        out=out,  # 输出
        disable_state_update=disable_state_update,  # 禁用状态更新标志
        intermediate_states_buffer=intermediate_states_buffer,  # 中间状态缓冲区
        cache_steps=cache_steps,  # 缓存步数
        retrieve_parent_token=retrieve_parent_token,  # 父token索引
        intermediate_state_indices=intermediate_state_indices,  # 中间状态索引
    )
