# 稀疏注意力协调器模块，定义了请求状态追踪器、稀疏配置和协调器，
# 负责管理稀疏注意力的完整生命周期，包括表示构建、稀疏检索和token卸载
import logging  # 导入日志模块
from dataclasses import dataclass, field  # 导入数据类装饰器和字段工具
from typing import TYPE_CHECKING, Any, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch

from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool  # 导入KV缓存池和请求到token映射池
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm  # 导入稀疏算法基类
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor  # 导入后端适配器基类

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
    from sglang.srt.managers.schedule_batch import Req  # 导入请求类
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class RequestTrackers:  # 请求状态追踪器类，用于跟踪稀疏注意力请求的状态
    """State tracker for sparse attention requests."""  # 稀疏注意力请求的状态追踪器 / 稀疏注意力请求的状态追踪器

    def __init__(  # 初始化请求状态追踪器
        self,
        max_pool_size: int,  # 池的最大大小
        device: torch.device,  # 设备信息
        num_layers: int,  # 层数量
        min_sparse_prompt_len: int,  # 最小稀疏提示长度阈值
        max_context_len: int,  # 最大上下文长度
    ):
        self.device = device  # 保存设备信息
        self.num_layers = num_layers  # 保存层数量

        self.repr_constructed = torch.zeros(  # 表示是否已构建稀疏表示的布尔张量
            max_pool_size, dtype=torch.bool, device=device
        )
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)  # 每个请求的提示长度
        self.last_constructed_page = torch.zeros(  # 上次构建表示时的页索引
            max_pool_size, dtype=torch.int64, device=device
        )

        # TODO: Add more trackers for hierarchical KVCache management  # 待添加：更多用于分层KV缓存管理的追踪器

    def register(self, idx: int, prompt_len: int) -> None:  # 注册新请求，初始化其追踪状态
        self.repr_constructed[idx] = False  # 标记表示尚未构建
        self.prompt_lens[idx] = prompt_len  # 记录提示长度
        self.last_constructed_page[idx] = 0  # 上次构建页索引初始化为0

    def clear(self, idx: int) -> None:  # 清除指定请求的追踪状态
        self.repr_constructed[idx] = False  # 重置表示构建标记
        self.prompt_lens[idx] = 0  # 清零提示长度
        self.last_constructed_page[idx] = 0  # 清零上次构建页索引


@dataclass  # 数据类装饰器
class SparseConfig:  # 稀疏注意力配置类
    """Configuration for sparse attention."""  # 稀疏注意力的配置 / 稀疏注意力的配置

    top_k: int = 2048  # 检索的Top-K数量，默认2048
    device_buffer_size: int = 4096  # 设备缓冲区大小，默认4096
    host_to_device_ratio: int = 2  # 主机到设备的比率，默认2
    algorithm: Optional[str] = None  # 稀疏算法名称，默认为None
    backend: Optional[str] = None  # 注意力后端名称，默认为None
    page_size: Optional[int] = None  # 页面大小，默认为None
    min_sparse_prompt_len: Optional[int] = None  # 最小稀疏提示长度阈值，默认为None
    sparse_extra_config: dict = field(  # 算法特定的额外配置字典
        default_factory=dict
    )  # Algorithm-specific config, parsed by each algorithm  # 算法特定配置，由各算法自行解析


class SparseCoordinator:  # 稀疏注意力协调器类
    """
    Coordinator for sparse attention with retrievable KV cache compression.  # 可检索KV缓存压缩的稀疏注意力协调器

    This coordinator framework is designed for decode-phase retrievable algorithms
    (e.g., Quest, PQCache, SnapKV) that dynamically select important KV cache entries
    based on current queries. It manages the lifecycle of sparse attention including
    representation construction, sparse retrieval, and token offloading.  # 该协调器框架专为解码阶段可检索算法设计，
    # 根据当前查询动态选择重要的KV缓存条目。管理稀疏注意力的完整生命周期，包括表示构建、稀疏检索和token卸载。

    Request Lifecycle and API Calls:  # 请求生命周期和API调用
        1. Request Start:  # 1. 请求开始
           - on_request_begin(req) -> Register request and initialize state  # 注册请求并初始化状态

        2. Prefill Phase:  # 2. 预填充阶段
           - attention_end(...)    -> Construct representations  # 构建表示

        3. Decode Phase:  # 3. 解码阶段
           - forward_begin(batch)  -> Wait for pending KVCache offloading  # 等待挂起的KV缓存卸载完成
           - attention_begin(...)  -> Identify important KV, load offloaded KVCache, adapt attention metadata  # 识别重要KV，加载卸载的KV缓存，适配注意力元数据
           - attention_end(...)    -> Construct/update representations  # 构建/更新表示
           - forward_end(batch)    -> Trigger KVCache offloading  # 触发KV缓存卸载

        4. Request End:  # 4. 请求结束
           - on_request_end(req) -> Clean up state and resources  # 清理状态和资源
    """

    def __init__(  # 初始化稀疏注意力协调器
        self,
        config: SparseConfig,  # 稀疏配置
        algorithm: BaseSparseAlgorithm,  # 稀疏算法实例
        backend_adaptor: Optional[BackendAdaptor],  # 后端适配器实例
        req_to_token_pool: ReqToTokenPool,  # 请求到token映射池
        token_to_kv_pool: KVCache,  # token到KV缓存池
        start_layer: int,  # 起始层ID
        end_layer: int,  # 结束层ID
        device: torch.device,  # 设备信息
    ):
        self.config = config  # 保存稀疏配置
        self.algorithm = algorithm  # 保存稀疏算法实例
        self.backend_adaptor = backend_adaptor  # 保存后端适配器
        self.req_to_token_pool = req_to_token_pool  # 保存请求到token映射池
        self.token_to_kv_pool = token_to_kv_pool  # 保存token到KV缓存池
        self.start_layer = start_layer  # 保存起始层ID
        self.end_layer = end_layer  # 保存结束层ID
        self.device = device  # 保存设备信息
        self.page_size = config.page_size  # 保存页面大小

        self.states = RequestTrackers(  # 创建请求状态追踪器
            req_to_token_pool.req_to_token.shape[0],  # 池大小
            device,  # 设备
            end_layer - start_layer + 1,  # 层数
            self.config.min_sparse_prompt_len,  # 最小稀疏提示长度
            self.req_to_token_pool.max_context_len,  # 最大上下文长度
        )

        # Initialize algorithm representation pool and context  # 初始化算法表示池和上下文
        self.algorithm.initialize_representation_pool(
            start_layer,  # 起始层
            end_layer,  # 结束层
            self.token_to_kv_pool,  # token到KV缓存池
            self.req_to_token_pool,  # 请求到token映射池
            self.states,  # 请求状态追踪器
        )

        logger.info(  # 记录初始化信息
            f"SparseCoordinator initialized with sparse algorithm={type(algorithm).__name__}"
        )

    def on_request_begin(self, req: "Req") -> None:  # 处理请求开始事件
        """
        Handle request begin event. Called when a new request is created.  # 处理请求开始事件，在新请求创建时调用

        Registers the request in the state tracker to enable sparse attention processing.  # 在状态追踪器中注册请求以启用稀疏注意力处理
        """
        if req.req_pool_idx is not None:  # 如果请求已分配池索引
            self.states.register(req.req_pool_idx, len(req.origin_input_ids))  # 注册请求及其提示长度

    def on_request_end(self, req: "Req") -> None:  # 处理请求结束事件
        """
        Handle request end event. Called when a request is completed or aborted.  # 处理请求结束事件，在请求完成或中止时调用
        Cleans up request-specific state and releases resources.  # 清理请求特定的状态并释放资源
        """
        if req.req_pool_idx is None:  # 如果请求没有池索引，直接返回
            return

        self.states.clear(req.req_pool_idx)  # 清除该请求的状态追踪

        # TODO: Implement request end handling  # 待实现：请求结束处理
        # - Release host indices if any were allocated for offloading  # - 释放为卸载分配的主机索引

    def forward_begin(self, forward_batch: "ForwardBatch") -> None:  # 处理前向传播开始事件
        """
        Handle forward pass begin event. Called before each forward pass starts.  # 处理前向传播开始事件，在每次前向传播开始前调用

        Wait for pending KVCache offloading operations to complete before forward pass.  # 在前向传播前等待挂起的KV缓存卸载操作完成
        Ensures memory consistency for subsequent sparse attention operations.  # 确保后续稀疏注意力操作的内存一致性
        """
        # TODO: Implement forward begin handling  # 待实现：前向传播开始处理
        # - Check if there are pending offloading operations  # - 检查是否有挂起的卸载操作
        pass  # 当前为空实现

    def forward_end(self, forward_batch: "ForwardBatch") -> None:  # 处理前向传播结束事件
        """
        Handle forward pass end event. Called after each forward pass completes.  # 处理前向传播结束事件，在每次前向传播完成后调用

        Trigger async KVCache offloading operations.  # 触发异步KV缓存卸载操作
        """
        # TODO: Implement forward end handling  # 待实现：前向传播结束处理
        # - Identify tokens to offload  # - 识别需要卸载的token
        # - Trigger async offloading operations  # - 触发异步卸载操作
        pass  # 当前为空实现

    def attention_begin(  # 处理注意力计算开始事件
        self,
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        value: torch.Tensor,  # 值张量
        layer: "RadixAttention",  # 注意力层
        forward_batch: "ForwardBatch",  # 前向批次信息
        attn_metadata: Optional[Any],  # 注意力元数据
        **kwargs,  # 其他关键字参数
    ) -> Optional[Any]:  # 返回适配后的注意力元数据或None
        """
        Handle attention begin event. Called before each attention pass starts.  # 处理注意力计算开始事件，在每次注意力计算开始前调用

        Identify important KV entries via sparse algorithm, load offloaded KVCache if needed,
        and adapt attention metadata for the attention backend.  # 通过稀疏算法识别重要的KV条目，按需加载卸载的KV缓存，
        # 并为注意力后端适配注意力元数据
        """
        if layer.layer_id == self.start_layer:  # 如果是起始层
            self.backend_adaptor.save_original_metadata(attn_metadata)  # 在起始层保存原始注意力元数据

        return self._handle_sparse_retrieve(  # 执行稀疏检索并返回适配后的元数据
            query, layer, forward_batch, attn_metadata, **kwargs
        )

    def attention_end(  # 处理注意力计算结束事件
        self,
        output: torch.Tensor,  # 注意力输出张量
        layer: "RadixAttention",  # 注意力层
        forward_batch: "ForwardBatch",  # 前向批次信息
    ) -> None:  # 无返回值
        """
        Handle attention end event. Called after each attention pass completes.  # 处理注意力计算结束事件，在每次注意力计算完成后调用

        Maybe construct and update sparse representations.  # 可能构建和更新稀疏表示
        """
        layer_id = layer.layer_id  # 获取当前层ID

        # Maybe construct representations  # 可能构建表示
        self.algorithm.construct_representations(
            layer_id=layer_id,  # 层ID
            req_pool_indices=forward_batch.req_pool_indices,  # 请求池索引
            seq_lens=forward_batch.seq_lens,  # 序列长度
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),  # 获取该层的键缓存缓冲区
            forward_batch=forward_batch,  # 前向批次信息
        )

        # Maybe update representations  # 可能更新表示
        self.algorithm.update_representations(
            layer_id=layer_id,  # 层ID
            req_pool_indices=forward_batch.req_pool_indices,  # 请求池索引
            seq_lens=forward_batch.seq_lens,  # 序列长度
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),  # 获取该层的键缓存缓冲区
            forward_batch=forward_batch,  # 前向批次信息
        )

    def _handle_sparse_retrieve(  # 处理稀疏检索的私有方法
        self,
        query: torch.Tensor,  # 查询张量
        layer: "RadixAttention",  # 注意力层
        forward_batch: "ForwardBatch",  # 前向批次信息
        attn_metadata: Optional[Any],  # 注意力元数据
        **kwargs,  # 其他关键字参数
    ) -> Optional[torch.Tensor]:  # 返回适配后的注意力元数据或None
        req_pool_indices = forward_batch.req_pool_indices  # 获取请求池索引
        # Compute Topk  # 计算Top-K
        sparse_mask = self._compute_sparse_mask(req_pool_indices)  # 计算稀疏掩码
        selected_indices, valid_lengths = self.algorithm.retrieve_topk(  # 调用算法检索Top-K重要KV条目
            queries=query,  # 查询张量
            layer_id=layer.layer_id,  # 层ID
            req_pool_indices=req_pool_indices,  # 请求池索引
            sparse_mask=sparse_mask,  # 稀疏掩码
            forward_batch=forward_batch,  # 前向批次信息
            attn_metadata=attn_metadata,  # 注意力元数据
            **kwargs,  # 其他关键字参数
        )

        # Adapt Attention Metadata  # 适配注意力元数据
        return self.backend_adaptor.adapt_for_attn_metadata(  # 调用后端适配器适配元数据
            selected_indices=selected_indices,  # 选中的索引
            valid_lengths=valid_lengths,  # 有效长度
            sparse_mask=sparse_mask,  # 稀疏掩码
            current_metadata=attn_metadata,  # 当前注意力元数据
            forward_batch=forward_batch,  # 前向批次信息
            req_to_token=self.req_to_token_pool.req_to_token,  # 请求到token映射
            page_size=self.page_size,  # 页面大小
            layer_id=layer.layer_id,  # 层ID
        )

    def _compute_sparse_mask(self, req_pool_indices):  # 计算稀疏掩码的私有方法
        mask = (  # 生成掩码：提示长度大于等于最小稀疏提示长度的请求标记为True
            self.states.prompt_lens[req_pool_indices]  # 获取各请求的提示长度
            >= self.config.min_sparse_prompt_len  # 与最小稀疏提示长度比较
        )

        return mask  # 返回稀疏掩码
