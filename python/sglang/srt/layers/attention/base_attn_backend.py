# 注意力后端基类模块
# 定义了所有注意力后端的抽象基类AttentionBackend，提供前向传播、CUDA图、
# 解码、扩展、投机解码验证等接口的统一抽象。所有具体的注意力后端
# （如FlashInfer、Triton等）均需继承此类并实现相应方法。
from __future__ import annotations  # 启用延迟类型注解评估

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型

import torch  # 导入PyTorch框架

from sglang.kernel_api_logging import debug_kernel_api  # 导入内核API调试装饰器
from sglang.srt.utils.common import is_npu  # 导入NPU设备检测函数

if TYPE_CHECKING:  # 类型检查阶段才导入，避免运行时循环依赖
    from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata  # 导入索引器元数据基类
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层类
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 导入前向批次信息和前向模式
    from sglang.srt.speculative.spec_info import SpecInput  # 导入投机解码输入规格类


class AttentionBackend(ABC):  # 注意力后端抽象基类
    """The base class of attention backends"""  # 注意力后端的基类

    # Opt out only when this backend never reads seq_lens_cpu / seq_lens_sum.
    # 仅当此后端从不读取seq_lens_cpu / seq_lens_sum时才选择退出。
    needs_cpu_seq_lens: bool = True  # 是否需要CPU序列长度，默认为True

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向传播元数据
        """Init the metadata for a forward pass."""  # 初始化前向传播的元数据
        raise NotImplementedError()  # 抛出未实现异常

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):  # 初始化CUDA图全局共享状态
        """Init the global shared states for cuda graph."""  # 初始化CUDA图的全局共享状态
        raise NotImplementedError()  # 抛出未实现异常

    def init_forward_metadata_capture_cuda_graph(  # 初始化CUDA图捕获时的前向元数据
        self,
        bs: int,  # 批次大小
        num_tokens: int,  # token数量
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度张量
        encoder_lens: Optional[torch.Tensor],  # 编码器长度（可选）
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息（可选）
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""  # 初始化捕获CUDA图时前向传播的元数据
        raise NotImplementedError()  # 抛出未实现异常

    def init_forward_metadata_replay_cuda_graph(  # 初始化CUDA图重放时的前向元数据
        self,
        bs: int,  # 批次大小
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度张量
        seq_lens_sum: int,  # 序列长度总和
        encoder_lens: Optional[torch.Tensor],  # 编码器长度（可选）
        forward_mode: ForwardMode,  # 前向模式
        spec_info: Optional[SpecInput],  # 投机解码信息（可选）
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度（可选）
    ):
        """Init the metadata for a forward pass for replaying a cuda graph."""  # 初始化重放CUDA图时前向传播的元数据
        raise NotImplementedError()  # 抛出未实现异常

    def get_cuda_graph_seq_len_fill_value(self):  # 获取CUDA图序列长度填充值
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""  # 获取填充序列长度的填充值，通常为0或1
        raise NotImplementedError()  # 抛出未实现异常

    def on_after_cuda_graph_warmup(self):  # CUDA图预热后的钩子函数
        """Hook between cuda graph warmup pass and the actual capture.
        # CUDA图预热过程和实际捕获之间的钩子

        Override to undo state that warmup mutated or eagerly advanced
        (e.g. dirty metadata buffers, raw->full upgrades) before capture
        freezes the kernel pointers.
        # 重写以撤销预热修改或提前推进的状态（如脏元数据缓冲区、raw->full升级），
        # 在捕获冻结内核指针之前。
        """
        pass  # 默认无操作

    def get_verify_buffers_to_fill_after_draft(self):  # 获取草稿后需要填充的验证缓冲区
        """
        Return buffers of verify attention kernels that needs to be filled after draft.
        # 返回草稿后需要填充的验证注意力内核缓冲区

        Typically, these are tree mask and position buffers.
        # 通常这些是树掩码和位置缓冲区
        """
        return [None, None]  # 默认返回两个None

    def update_verify_buffers_to_fill_after_draft(  # 更新草稿后需要填充的验证缓冲区
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]  # 投机解码信息和CUDA图批次大小
    ):
        """
        Update the buffers returned by get_verify_fill_after_draft_buffers if needed.
        # 如有需要，更新get_verify_fill_after_draft_buffers返回的缓冲区

        Here, we need to redo the computation of all metadata of the attention backend
        that depends on tree mask and position buffers.
        # 此处需要重新计算注意力后端中依赖于树掩码和位置缓冲区的所有元数据
        """
        raise NotImplementedError()  # 抛出未实现异常

    @debug_kernel_api  # 内核API调试装饰器
    def forward(  # 注意力层前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 基数注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache: bool = True,  # 是否保存KV缓存，默认为True
        **kwargs,  # 其他关键字参数
    ):
        """Run forward on an attention layer."""  # 在注意力层上执行前向传播
        if forward_batch.forward_mode.is_idle():  # 如果前向模式为空闲
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)  # 返回空张量
        elif forward_batch.forward_mode.is_decode():  # 如果前向模式为解码
            return self.forward_decode(  # 调用解码前向传播
                q,  # 查询张量
                k,  # 键张量
                v,  # 值张量
                layer,  # 注意力层
                forward_batch,  # 前向批次
                save_kv_cache=save_kv_cache,  # 是否保存KV缓存
                **kwargs,  # 其他关键字参数
            )
        elif forward_batch.forward_mode.is_mixed() and is_npu():  # 如果前向模式为混合且是NPU设备
            return self.forward_mixed(  # 调用混合前向传播
                q,  # 查询张量
                k,  # 键张量
                v,  # 值张量
                layer,  # 注意力层
                forward_batch,  # 前向批次
                save_kv_cache=save_kv_cache,  # 是否保存KV缓存
                **kwargs,  # 其他关键字参数
            )
        else:  # 否则为扩展模式
            return self.forward_extend(  # 调用扩展前向传播
                q,  # 查询张量
                k,  # 键张量
                v,  # 值张量
                layer,  # 注意力层
                forward_batch,  # 前向批次
                save_kv_cache=save_kv_cache,  # 是否保存KV缓存
                **kwargs,  # 其他关键字参数
            )

    def forward_decode(  # 解码前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 基数注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache: bool = True,  # 是否保存KV缓存，默认为True
        **kwargs,  # 其他关键字参数
    ):
        """Run a forward for decode."""  # 执行解码前向传播
        raise NotImplementedError()  # 抛出未实现异常

    def forward_extend(  # 扩展（预填充）前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 基数注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache: bool = True,  # 是否保存KV缓存，默认为True
        **kwargs,  # 其他关键字参数
    ):
        """Run a forward for extend."""  # 执行扩展前向传播
        raise NotImplementedError()  # 抛出未实现异常

    def forward_mixed(  # 混合前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 基数注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache: bool = True,  # 是否保存KV缓存，默认为True
    ):
        """Run a forward for mix."""  # 执行混合前向传播
        raise NotImplementedError()  # 抛出未实现异常

    def support_triton(self):  # 检查当前后端是否支持Triton
        """Check if the current backend supports triton."""  # 检查当前后端是否支持Triton
        return True  # 默认支持

    def get_indexer_metadata(  # 获取索引器元数据
        self,
        layer_id: int,  # 层ID
        forward_batch: ForwardBatch,  # 前向批次
    ) -> Optional[BaseIndexerMetadata]:  # 返回索引器元数据或None
        """Get the indexer metadata. None means don't support indexer."""  # 获取索引器元数据，None表示不支持索引器
        return None  # 默认返回None
