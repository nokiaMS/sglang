# 稀疏注意力后端适配器模块，定义了注意力后端的抽象基类及FlashAttention和DSA后端的具体适配器，
# 用于将稀疏检索结果转换为特定后端所需的注意力元数据格式
import logging  # 导入日志模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import TYPE_CHECKING, Any, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class BackendAdaptor(ABC):  # 后端适配器抽象基类
    """Base class for attention backend adaptors."""  # 注意力后端适配器的基类 / 注意力后端适配器的基类

    def __init__(self, device: torch.device):  # 初始化后端适配器
        self.device = device  # 保存设备信息
        self._original_metadata = None  # 保存原始元数据的备份，初始为None

    def save_original_metadata(self, metadata: Any) -> None:  # 保存原始元数据的方法
        """Save original metadata in the beginning of the forward pass."""  # 在前向传播开始时保存原始元数据 / 在前向传播开始时保存原始元数据
        pass  # 基类中为空实现

    @abstractmethod  # 标记为抽象方法，子类必须实现
    def adapt_for_attn_metadata(  # 适配注意力元数据的抽象方法
        self,
        selected_indices: torch.Tensor,  # 选中的逻辑索引张量
        valid_lengths: torch.Tensor,  # 每个请求的有效长度张量
        sparse_mask: torch.Tensor,  # 稀疏掩码，标记哪些请求需要稀疏处理
        current_metadata: Any,  # 当前的注意力元数据
        forward_batch: "ForwardBatch",  # 前向批次信息
        req_to_token: torch.Tensor,  # 请求到token的映射张量
        page_size: int,  # 页面大小
        layer_id: int,  # 层ID
        **kwargs,  # 其他关键字参数
    ) -> Any:  # 返回适配后的注意力元数据
        """
        Adapt attention metadata for sparse KVCache access.  # 适配注意力元数据以支持稀疏KV缓存访问

        Transforms sparse retrieval results (logical indices of important KV pages/tokens)
        into backend-specific attention metadata format.  # 将稀疏检索结果（重要KV页/token的逻辑索引）转换为后端特定的注意力元数据格式

        Returns:
            Modified attention metadata compatible with the backend  # 返回与后端兼容的修改后的注意力元数据
        """
        pass  # 抽象方法，子类必须实现


class DSABackendAdaptor(BackendAdaptor):  # DSA（DeepSeek稀疏注意力）后端适配器类
    """Adaptor for DSA (DeepSeek Sparse Attention) backend."""  # DSA（DeepSeek稀疏注意力）后端的适配器 / DSA（DeepSeek稀疏注意力）后端的适配器

    def __init__(  # DSA后端适配器初始化
        self,
        device: torch.device,  # 设备信息
        req_to_token_pool,  # 请求到token的映射池
    ):
        super().__init__(device)  # 调用父类初始化
        self.req_to_token_pool = req_to_token_pool  # 保存请求到token映射池

    def adapt_for_attn_metadata(  # 适配注意力元数据的具体实现
        self,
        selected_indices: torch.Tensor,  # 选中的逻辑索引张量
        valid_lengths: torch.Tensor,  # 每个请求的有效长度张量
        sparse_mask: torch.Tensor,  # 稀疏掩码
        current_metadata: Any,  # 当前的注意力元数据
        forward_batch: "ForwardBatch",  # 前向批次信息
        req_to_token: torch.Tensor,  # 请求到token的映射张量
        page_size: int,  # 页面大小
        layer_id: int,  # 层ID
        **kwargs,  # 其他关键字参数
    ) -> Optional[torch.Tensor]:  # 返回适配后的张量或None
        """
        Transform logical page indices to physical device indices for DSA backend.  # 将逻辑页面索引转换为DSA后端的物理设备索引
        """
        # TODO: Implement DSA backend adaptor logic  # 待实现：DSA后端适配器逻辑
        pass  # 尚未实现


class FlashAttentionAdaptor(BackendAdaptor):  # FlashAttention后端适配器类
    """Adaptor for FlashAttention backend."""  # FlashAttention后端的适配器 / FlashAttention后端的适配器

    def save_original_metadata(self, metadata: Any) -> None:  # 保存原始元数据的具体实现
        self._original_metadata = {  # 将原始元数据备份到字典中
            "page_table": metadata.page_table.clone(),  # 克隆页表
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),  # 克隆缓存序列长度
            "cu_seqlens_k": metadata.cu_seqlens_k.clone(),  # 克隆KV累积序列长度
            "max_seq_len_k": metadata.max_seq_len_k,  # 保存最大KV序列长度（标量无需克隆）
        }

    def adapt_for_attn_metadata(  # 适配FlashAttention元数据的具体实现
        self,
        selected_indices: torch.Tensor,  # 选中的逻辑索引张量
        valid_lengths: torch.Tensor,  # 每个请求的有效长度张量
        sparse_mask: torch.Tensor,  # 稀疏掩码
        current_metadata: Any,  # 当前的注意力元数据
        forward_batch: "ForwardBatch",  # 前向批次信息
        req_to_token: torch.Tensor,  # 请求到token的映射张量
        page_size: int,  # 页面大小
        layer_id: int,  # 层ID
        **kwargs,  # 其他关键字参数
    ) -> Any:  # 返回适配后的注意力元数据
        """
        Adapt FlashAttention metadata for sparse KVCache access.  # 适配FlashAttention元数据以支持稀疏KV缓存访问

        Modifies page_table, cache_seqlens, and related metadata to redirect
        FlashAttention to only process selected sparse pages.  # 修改页表、缓存序列长度及相关元数据，使FlashAttention仅处理选中的稀疏页面

        # TODO: Optimize performance  # 待优化：性能优化
        """
        if self._original_metadata is None:  # 如果没有保存原始元数据
            return current_metadata  # 直接返回当前元数据，不做修改

        if not sparse_mask.any():  # 如果没有任何请求需要稀疏处理
            return current_metadata  # 直接返回当前元数据

        current_metadata.page_table.copy_(self._original_metadata["page_table"])  # 恢复原始页表
        current_metadata.cache_seqlens_int32.copy_(  # 恢复原始缓存序列长度
            self._original_metadata["cache_seqlens_int32"]
        )

        physical_pages = self._logical_to_physical_pages_batch(  # 将逻辑页索引批量转换为物理页索引
            selected_indices,  # 逻辑页索引
            forward_batch.req_pool_indices,  # 请求池索引
            req_to_token,  # 请求到token的映射
            page_size,  # 页面大小
        )

        max_selected = physical_pages.shape[1]  # 获取最大选中页数
        valid_mask = torch.arange(max_selected, device=physical_pages.device).unsqueeze(  # 创建有效位置掩码
            0  # 在第0维增加维度
        ) < valid_lengths.unsqueeze(1)  # 与有效长度比较，标记有效位置
        update_mask = sparse_mask.unsqueeze(1) & valid_mask  # 结合稀疏掩码和有效掩码得到更新掩码

        current_metadata.page_table[:, :max_selected] = torch.where(  # 条件更新页表：仅在更新掩码处替换为物理页
            update_mask, physical_pages, current_metadata.page_table[:, :max_selected]
        )

        seq_lens = forward_batch.seq_lens  # 获取序列长度
        positions_in_page = (seq_lens - 1) % page_size  # 计算每个请求在当前页中的位置
        diff = page_size - positions_in_page - 1  # 计算当前页中剩余未填充的token数
        sparse_seq_lens = (valid_lengths * page_size - diff).to(torch.int32)  # 计算稀疏序列长度

        current_metadata.cache_seqlens_int32 = torch.where(  # 条件更新缓存序列长度：稀疏请求用稀疏长度，其他保留原始值
            sparse_mask, sparse_seq_lens, self._original_metadata["cache_seqlens_int32"]
        )

        current_metadata.cu_seqlens_k = torch.nn.functional.pad(  # 根据更新后的序列长度重新计算累积序列长度
            torch.cumsum(  # 计算累积和
                current_metadata.cache_seqlens_int32, dim=0, dtype=torch.int32  # 沿批次维度累加
            ),
            (1, 0),  # 在前面填充一个0
        )
        current_metadata.max_seq_len_k = int(current_metadata.cache_seqlens_int32.max())  # 更新最大KV序列长度
        return current_metadata  # 返回修改后的元数据

    def _logical_to_physical_pages_batch(  # 将逻辑页索引批量转换为物理页索引的私有方法
        self,
        logical_pages: torch.Tensor,  # 逻辑页索引张量，形状为[bs, max_pages]
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        req_to_token: torch.Tensor,  # 请求到token的映射张量
        page_size: int,  # 页面大小
    ) -> torch.Tensor:  # 返回物理页索引张量
        bs, max_pages = logical_pages.shape  # 获取批次大小和最大页数

        page_starts = logical_pages * page_size  # 计算每个逻辑页的起始token索引
        page_starts_clamped = page_starts.clamp(min=0)  # 将负值裁剪为0，防止索引越界

        req_indices_expanded = req_pool_indices.unsqueeze(1).expand(-1, max_pages)  # 扩展请求索引以匹配逻辑页维度
        first_tokens = req_to_token[req_indices_expanded, page_starts_clamped]  # 查找每个页起始位置的物理token ID

        physical_pages = first_tokens // page_size  # 通过整除页面大小得到物理页索引
        physical_pages = torch.where(  # 处理无效逻辑页（负值），将其物理页索引置为0
            logical_pages >= 0, physical_pages, torch.zeros_like(physical_pages)
        )

        return physical_pages.to(torch.int32)  # 转换为int32类型并返回
