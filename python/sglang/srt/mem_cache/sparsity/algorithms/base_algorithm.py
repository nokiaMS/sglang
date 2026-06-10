# 稀疏注意力算法的基类定义文件，提供抽象接口和通用实现框架
# 包含BaseSparseAlgorithm（抽象基类）和BaseSparseAlgorithmImpl（通用实现基类）
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import TYPE_CHECKING  # 导入类型检查常量，用于条件导入类型提示

import torch  # 导入PyTorch张量库

if TYPE_CHECKING:  # 仅在类型检查时导入，运行时不导入
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类


class BaseSparseAlgorithm(ABC):  # 稀疏注意力算法的抽象基类，继承自ABC
    """
    Abstract base class for sparse attention algorithms.
    稀疏注意力算法的抽象基类。

    This class provides a unified interface for implementing various retrievable KVCache
    compression algorithms. Token-wise sparsity is treated as page-wise with page_size=1.
    该类为实现各种可检索KVCache压缩算法提供统一接口。逐token稀疏被视为page_size=1的逐页稀疏。

    References:
        - ChunkKV: https://arxiv.org/abs/2502.00299
        - Quest: https://arxiv.org/pdf/2406.10774
        - PQCache: https://arxiv.org/abs/2407.12820
        - SnapKV: https://arxiv.org/pdf/2404.14469
        - Look-ahead QCache: https://arxiv.org/pdf/2505.20334
        - and more... 以及更多...
    """

    def __init__(self, config, device: torch.device, **kwargs):  # 初始化稀疏注意力算法基类
        self.config = config  # 保存稀疏注意力配置对象
        self.device = device  # 保存计算设备（CPU或CUDA设备）
        self.req_to_token_pool = None  # 请求到token的映射池，初始化为空
        self.states = None  # 算法状态对象，初始化为空

    def initialize_representation_pool(  # 初始化算法特定的表示池和上下文
        self,
        start_layer: int,  # 起始层索引
        end_layer: int,  # 结束层索引
        token_to_kv_pool,  # token到KV缓存池的映射
        req_to_token_pool,  # 请求到token的映射池
        states,  # 算法状态对象
    ):
        """
        Initialize algorithm-specific representation pool and set context.
        初始化算法特定的表示池并设置上下文。

        Called once during SparseCoordinator initialization. Algorithms allocate
        their own representation tensors and store references to context.
        在SparseCoordinator初始化期间调用一次。算法分配自己的表示张量并存储上下文引用。

        Algorithm-specific implementations:
            - ChunkKV: Allocate chunk scores [num_chunks, 1] for tracking semantic chunk importance
            - Quest: Allocate page representations [num_pages, repr_dim] via key pooling
            - PQCache: Allocate centroids [n_subvec, n_centroids, subvec_dim] and token codes [num_tokens, n_subvec]
            - SnapKV: Allocate voting scores [num_tokens] and selected positions mask for retention strategy
            - Look-ahead QCache: Allocate importance scores [num_tokens], eviction mask, and optional pseudo query cache [cache_size, hidden_dim]
        算法特定实现：
            - ChunkKV: 分配块分数[num_chunks, 1]用于跟踪语义块重要性
            - Quest: 通过键池化分配页表示[num_pages, repr_dim]
            - PQCache: 分配质心[n_subvec, n_centroids, subvec_dim]和token编码[num_tokens, n_subvec]
            - SnapKV: 分配投票分数[num_tokens]和保留策略的选定位置掩码
            - Look-ahead QCache: 分配重要性分数[num_tokens]、驱逐掩码和可选的伪查询缓存[cache_size, hidden_dim]
        """
        pass  # 基类中为空实现，由子类重写

    def construct_representations(  # 在预填充阶段构建初始表示
        self,
        layer_id: int,  # 当前层索引
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度张量
        k_buffer: torch.Tensor,  # 键缓存缓冲区
        forward_batch: "ForwardBatch",  # 前向批次信息
    ):
        """
        Construct initial representations during prefill phase.
        在预填充阶段构建初始表示。

        Called at every layer during forward pass. Algorithm internally decides
        whether to perform construction.
        Typically only constructs once per request during prefill/extend phase.
        在前向传播的每一层调用。算法内部决定是否执行构建。
        通常在每个请求的预填充/扩展阶段只构建一次。

        Algorithm-specific implementations:
            - ChunkKV: Compute chunk importance scores via aggregated key L2 norms within semantic chunks
            - Quest: Compute page representations via mean pooling of keys within each page
            - PQCache: Run K-means clustering to generate centroids and assign each token to nearest centroid
            - SnapKV: Select observation window (recent tokens), compute attention weights, aggregate via voting to identify important prefix positions, apply 1D pooling to preserve context
            - Look-ahead QCache: Generate pseudo lookahead query (e.g., mean of last k queries), compute KV importance scores, mark low-importance KVs for eviction
        算法特定实现：
            - ChunkKV: 通过语义块内聚合键L2范数计算块重要性分数
            - Quest: 通过每页内键的均值池化计算页表示
            - PQCache: 运行K-means聚类生成质心并将每个token分配到最近质心
            - SnapKV: 选择观察窗口（最近token），计算注意力权重，通过投票聚合识别重要前缀位置，应用1D池化保留上下文
            - Look-ahead QCache: 生成伪前瞻查询（如最后k个查询的均值），计算KV重要性分数，标记低重要性KV以驱逐
        """
        pass  # 基类中为空实现，由子类重写

    def update_representations(  # 在解码阶段增量更新表示
        self,
        layer_id: int,  # 当前层索引
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度张量
        k_buffer: torch.Tensor,  # 键缓存缓冲区
        forward_batch: "ForwardBatch",  # 前向批次信息
    ):
        """
        Incrementally update representations during decode phase.
        在解码阶段增量更新表示。

        Called at every layer during forward pass. Algorithm internally decides
        whether to update based on:
        在前向传播的每一层调用。算法内部根据以下条件决定是否更新：
        - self.states.repr_constructed[req_id]: Whether initial construction done 是否已完成初始构建
        - self.states.last_constructed_page[req_id]: Last constructed page index 上次构建的页索引
        - Current seq_lens: To detect new tokens/pages 当前序列长度：检测新token/页

        Algorithm-specific implementations:
            - ChunkKV: Incrementally compute importance scores for newly generated chunks during decode
            - Quest: Incrementally compute representations for newly generated pages during decode
            - PQCache: Assign new tokens to existing centroids (no centroid update during decode)
            - SnapKV: Optional: periodically re-run voting with sliding observation window (typically static after prefill)
            - Look-ahead QCache: Periodically regenerate pseudo queries and re-evaluate importance scores to adapt to generation dynamics
        算法特定实现：
            - ChunkKV: 在解码期间增量计算新生成块的重要性分数
            - Quest: 在解码期间增量计算新生成页的表示
            - PQCache: 将新token分配到现有质心（解码期间不更新质心）
            - SnapKV: 可选：使用滑动观察窗口周期性重新运行投票（通常预填充后保持静态）
            - Look-ahead QCache: 周期性重新生成伪查询并重新评估重要性分数以适应生成动态
        """
        pass  # 基类中为空实现，由子类重写

    @abstractmethod  # 标记为抽象方法，子类必须实现
    def retrieve_topk(  # 检索Top-K重要的KV索引用于稀疏注意力计算
        self,
        queries: torch.Tensor,  # 当前查询向量 [bs, num_heads, head_dim]
        layer_id: int,  # 当前层索引
        req_pool_indices: torch.Tensor,  # 请求池索引 [bs]
        sparse_mask: torch.Tensor,  # 稀疏注意力掩码 [bs]，标记哪些请求需要稀疏注意力
        **kwargs,  # 算法特定的额外参数
    ) -> tuple:  # 返回选中的索引和有效长度的元组
        """
        Retrieve top-k important KV indices for sparse attention.
        检索Top-K重要的KV索引用于稀疏注意力计算。

        Called before attention computation at each layer. Uses current query
        and pre-computed representations to select the most important subset
        of KV cache for attention computation.
        在每层注意力计算之前调用。使用当前查询和预计算的表示来选择KV缓存中最重要的子集进行注意力计算。

        Args:
            queries: [bs, num_heads, head_dim] Current query vectors 当前查询向量
            layer_id: Current layer index 当前层索引
            req_pool_indices: [bs] Request pool indices 请求池索引
            sparse_mask: [bs] bool, which requests need sparse attention 哪些请求需要稀疏注意力
            attn_metadata: Attention metadata (contains seq_lens, etc.) 注意力元数据（包含序列长度等）
            **kwargs: Algorithm-specific arguments 算法特定的参数

        Returns:
            selected_indices: [bs, max_selected] Selected page/token indices, padded with -1 选中的页/token索引，用-1填充
            valid_lengths: [bs] Actual number of selected indices per request 每个请求实际选中的索引数量

        Note:
            - Indices are logical positions that will be mapped to physical KV cache by BackendAdaptor
            - 索引是逻辑位置，将由BackendAdaptor映射到物理KV缓存

        Algorithm-specific implementations:
            - ChunkKV: Select top-k chunks based on pre-computed importance scores with layer-wise index reuse
            - Quest: Compute query-page similarity using current query and stored page representations, select top-k pages
            - PQCache: Calculate query-centroid similarity, use centroid scores to rank tokens, select top-k tokens
            - SnapKV: Return union of voted important prefix positions (with clustered neighbors) and observation window tokens
            - Look-ahead QCache: Return KVs not marked for eviction (eviction based on pseudo query importance evaluation)
        算法特定实现：
            - ChunkKV: 基于预计算的重要性分数选择top-k块，支持逐层索引复用
            - Quest: 使用当前查询和存储的页表示计算查询-页相似度，选择top-k页
            - PQCache: 计算查询-质心相似度，使用质心分数对token排序，选择top-k token
            - SnapKV: 返回投票的重要前缀位置（含聚类邻居）和观察窗口token的并集
            - Look-ahead QCache: 返回未标记驱逐的KV（驱逐基于伪查询重要性评估）
        """
        pass  # 抽象方法，子类必须实现


class BaseSparseAlgorithmImpl(BaseSparseAlgorithm):  # 稀疏注意力算法的通用实现基类，继承自BaseSparseAlgorithm
    """
    Implementation base class for sparse attention algorithms.
    稀疏注意力算法的实现基类。

    Provides common infrastructure for algorithms that operate at page/chunk granularity
    (token-wise is simply page_size=1):
    为以页/块粒度操作的算法提供通用基础设施（逐token只是page_size=1的情况）：
    - Generic construct/update flow with state tracking 通用构建/更新流程与状态跟踪
    - TopK retrieval with recent page retention (can be overridden) 带最近页保留的TopK检索（可重写）

    Subclasses need to implement:
    子类需要实现：
    - _initialize_representation_pools(): Initialize algorithm-specific representation pools 初始化算法特定的表示池
    - _compute_page_representations(): Compute page scores/representations 计算页分数/表示
    - _retrieve_page_scores(): Retrieve page scores for TopK selection 检索页分数用于TopK选择

    Subclasses can also override any method for specialized behavior
    子类也可以重写任何方法以实现特殊行为
    """

    def __init__(self, config, device: torch.device, **kwargs):  # 初始化稀疏注意力算法实现基类
        super().__init__(config, device, **kwargs)  # 调用父类初始化
        self.sparsity_ratio = config.sparse_extra_config.get("sparsity_ratio", 0.7)  # 获取稀疏比率，默认0.7
        self.num_recent_pages = config.sparse_extra_config.get("num_recent_pages", 4)  # 获取保留的最近页数，默认4
        self.page_size = config.page_size  # 获取页大小

    def initialize_representation_pool(  # 初始化算法特定的表示池和设置上下文
        self,
        start_layer: int,  # 起始层索引
        end_layer: int,  # 结束层索引
        token_to_kv_pool,  # token到KV缓存池的映射
        req_to_token_pool,  # 请求到token的映射池
        states,  # 算法状态对象
    ):
        self.req_to_token_pool = req_to_token_pool  # 保存请求到token的映射池
        self.token_to_kv_pool = token_to_kv_pool  # 保存token到KV缓存池的映射
        self.start_layer = start_layer  # 保存起始层索引
        self.end_layer = end_layer  # 保存结束层索引
        self.states = states  # 保存算法状态对象

        total_num_tokens = token_to_kv_pool.get_key_buffer(start_layer).shape[0]  # 获取KV池中总token数
        total_num_pages = (total_num_tokens + self.page_size - 1) // self.page_size  # 计算总页数（向上取整）

        # Initialize algorithm-specific representation pools 初始化算法特定的表示池
        self._initialize_representation_pools(start_layer, end_layer, total_num_pages)  # 调用子类实现的初始化方法

    def construct_representations(  # 在预填充阶段构建页表示
        self,
        layer_id,  # 当前层索引
        req_pool_indices,  # 请求池索引
        seq_lens,  # 序列长度
        k_buffer,  # 键缓存缓冲区
        forward_batch,  # 前向批次信息
    ) -> torch.Tensor:

        if not forward_batch.forward_mode.is_extend():  # 如果不是扩展（预填充）模式，直接返回
            return

        num_pages = seq_lens // self.page_size  # 计算每个请求的页数
        valid_mask = (  # 构建有效掩码：需要构建表示的请求
            ~self.states.repr_constructed[req_pool_indices]  # 尚未构建过表示
            & (seq_lens >= self.states.prompt_lens[req_pool_indices])  # 序列长度达到提示长度
            & (num_pages > 0)  # 至少有一页
        )

        if not valid_mask.any():  # 如果没有有效请求，直接返回
            return

        # Compute page representations by subclass 由子类计算页表示
        self._compute_page_representations(
            layer_id,  # 当前层索引
            req_pool_indices[valid_mask],  # 筛选有效请求的池索引
            seq_lens[valid_mask],  # 筛选有效请求的序列长度
            0,  # 起始页为0（从第一页开始）
            num_pages[valid_mask],  # 结束页为各请求的页数
            k_buffer,  # 键缓存缓冲区
        )

        # Update tracking states 更新跟踪状态
        if layer_id == self.end_layer - 1:  # 在最后一层完成后更新状态
            success_indices = req_pool_indices[valid_mask]  # 获取成功构建的请求索引
            self.states.repr_constructed[success_indices] = True  # 标记这些请求已完成表示构建
            self.states.last_constructed_page[success_indices] = num_pages[valid_mask]  # 记录最后构建的页索引

    def update_representations(  # 在解码阶段增量更新页表示
        self,
        layer_id,  # 当前层索引
        req_pool_indices,  # 请求池索引
        seq_lens,  # 序列长度
        k_buffer,  # 键缓存缓冲区
        forward_batch,  # 前向批次信息
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_decode_or_idle():  # 如果不是解码或空闲模式，直接返回
            return

        start_page = self.states.last_constructed_page[req_pool_indices]  # 获取上次构建的页索引
        end_page = seq_lens // self.page_size  # 计算当前序列长度对应的页数
        valid_mask = self.states.repr_constructed[req_pool_indices] & (  # 构建有效掩码
            start_page < end_page  # 有新增页需要更新
        )

        if not valid_mask.any():  # 如果没有有效请求，直接返回
            return

        # Compute page representations by subclass 由子类计算页表示
        self._compute_page_representations(
            layer_id,  # 当前层索引
            req_pool_indices[valid_mask],  # 筛选有效请求的池索引
            seq_lens[valid_mask],  # 筛选有效请求的序列长度
            start_page[valid_mask],  # 起始页为上次构建的位置
            end_page[valid_mask],  # 结束页为当前页数
            k_buffer,  # 键缓存缓冲区
        )

        # Update tracking states 更新跟踪状态
        if layer_id == self.end_layer - 1:  # 在最后一层完成后更新状态
            success_indices = req_pool_indices[valid_mask]  # 获取成功更新的请求索引
            self.states.last_constructed_page[success_indices] = end_page[valid_mask]  # 更新最后构建的页索引

    def retrieve_topk(  # 默认的TopK检索方法：基于分数的选择 + 最近页保留
        self,
        queries: torch.Tensor,  # 查询向量
        layer_id: int,  # 当前层索引
        req_pool_indices: torch.Tensor,  # 请求池索引
        sparse_mask: torch.Tensor,  # 稀疏注意力掩码
        **kwargs,  # 额外参数
    ) -> tuple:  # 返回选中的索引和有效长度的元组
        """
        Default TopK retrieval: score-based selection + recent pages.
        默认TopK检索：基于分数的选择 + 最近页保留。
        Subclasses can override for query-dependent retrieval.
        子类可以重写以实现依赖查询的检索。

        TODO:
            1. Using triton kernel to speed up this function 使用triton内核加速此函数
            2. Support CUDA Graph 支持CUDA图
        """
        bs, device = queries.shape[0], queries.device  # 获取批次大小和设备

        seq_lens_source = kwargs.get("forward_batch", None)  # 从额外参数中获取forward_batch
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):  # 如果没有forward_batch或不含seq_lens
            raise ValueError(  # 抛出值错误异常
                "forward_batch with seq_lens is required for TopK retrieval"  # TopK检索需要包含seq_lens的forward_batch
            )
        seq_lens = seq_lens_source.seq_lens.to(device)  # 将序列长度转移到对应设备

        req_to_token = self.req_to_token_pool.req_to_token  # 获取请求到token的映射矩阵
        max_req_tokens = req_to_token.shape[1]  # 获取每个请求的最大token数

        per_request_indices = []  # 存储每个请求选中的页索引
        per_request_lengths = []  # 存储每个请求选中的页数量

        for i in range(bs):  # 遍历每个请求
            if not sparse_mask[i]:  # 如果该请求不需要稀疏注意力
                per_request_indices.append(  # 添加空索引张量
                    torch.empty(0, device=device, dtype=torch.int32)  # 创建空的int32张量
                )
                per_request_lengths.append(0)  # 记录长度为0
                continue  # 跳过当前请求

            num_pages = int((seq_lens[i].item() + self.page_size - 1) // self.page_size)  # 计算当前请求的页数
            if num_pages <= self.num_recent_pages:  # 如果页数不超过保留的最近页数
                per_request_indices.append(  # 添加空索引张量
                    torch.empty(0, device=device, dtype=torch.int32)  # 创建空的int32张量
                )
                per_request_lengths.append(0)  # 记录长度为0
                continue  # 跳过当前请求（页数太少无需稀疏）

            page_idx = torch.arange(num_pages, device=device)  # 创建页索引序列[0, 1, ..., num_pages-1]
            page_start_token = req_to_token[  # 获取每页起始token的物理索引
                req_pool_indices[i],  # 当前请求的池索引
                (page_idx * self.page_size).clamp(0, max_req_tokens - 1),  # 每页起始位置，限制在有效范围内
            ]
            phys_pages = (page_start_token // self.page_size).unsqueeze(0)  # 计算每页的物理页索引，增加batch维度

            scores = self._retrieve_page_scores(  # 调用子类方法检索页分数
                layer_id,  # 当前层索引
                phys_pages,  # 物理页索引
                req_pool_indices[i : i + 1],  # 当前请求的池索引（保持batch维度）
                queries[i : i + 1],  # 当前请求的查询向量（保持batch维度）
            )

            recent_start = max(num_pages - self.num_recent_pages, 0)  # 计算最近页的起始索引
            scores = scores.clone()  # 克隆分数以避免修改原始数据
            scores[:, recent_start:] = float("-inf")  # 将最近页的分数设为负无穷（不参与TopK选择）

            history_pages = max(recent_start, 1)  # 历史页数（排除最近页）
            k = max(int(history_pages * self.sparsity_ratio), 1)  # 计算TopK的k值，至少为1
            k = min(k, history_pages)  # k值不超过历史页数
            topk_idx = torch.topk(scores, k=k, dim=1, sorted=False)[1].squeeze(0)  # 获取TopK索引，不排序，移除batch维度

            recent_idx = torch.arange(  # 创建最近页的索引序列
                recent_start, recent_start + self.num_recent_pages, device=device  # 从recent_start开始的num_recent_pages个索引
            )
            recent_idx = recent_idx[recent_idx < num_pages]  # 过滤掉超出页数范围的索引

            combined = (  # 合并TopK索引和最近页索引
                torch.cat([topk_idx, recent_idx], dim=0).sort()[0].to(torch.int32)  # 拼接后排序并转为int32
            )

            per_request_indices.append(combined)  # 添加当前请求的合并索引
            per_request_lengths.append(int(combined.numel()))  # 添加当前请求选中的页数

        max_len = max(max(per_request_lengths, default=0), 1)  # 计算所有请求中的最大选中页数，至少为1
        out_indices = torch.full((bs, max_len), -1, dtype=torch.int32, device=device)  # 创建输出索引张量，用-1填充
        out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)  # 创建输出长度张量，初始化为0

        for i, selected in enumerate(per_request_indices):  # 遍历每个请求的选中索引
            length = per_request_lengths[i]  # 获取当前请求选中的页数
            if length == 0:  # 如果没有选中页
                continue  # 跳过
            out_indices[i, :length] = selected  # 将选中索引填入输出张量
            out_lengths[i] = length  # 记录有效长度

        return out_indices, out_lengths  # 返回选中的索引和有效长度

    def _initialize_representation_pools(  # 初始化算法特定的表示池（由子类实现）
        self, start_layer: int, end_layer: int, total_num_pages: int  # 起始层、结束层、总页数
    ):
        """Initialize algorithm-specific representation pools for all layers. 初始化所有层的算法特定表示池。"""
        raise NotImplementedError  # 未实现异常，子类必须重写

    def _compute_page_representations(  # 计算给定页范围的页表示（由子类实现）
        self,
        layer_id: int,  # 当前层索引
        reqs: torch.Tensor,  # 请求索引张量
        seq_lens: torch.Tensor,  # 序列长度张量
        start_page,  # 起始页索引
        end_page: torch.Tensor,  # 结束页索引张量
        k_buffer: torch.Tensor,  # 键缓存缓冲区
    ):
        """Compute and store page representations for given page range. 计算并存储给定页范围的页表示。"""
        raise NotImplementedError  # 未实现异常，子类必须重写

    def _retrieve_page_scores(  # 检索页分数用于TopK选择（由子类实现）
        self,
        layer_id: int,  # 当前层索引
        phys_pages: torch.Tensor,  # 物理页索引张量
        req_pool_indices: torch.Tensor,  # 请求池索引张量
        queries: torch.Tensor,  # 查询向量张量
    ) -> torch.Tensor:  # 返回页分数张量
        """Retrieve page scores for TopK selection. 检索页分数用于TopK选择。"""
        raise NotImplementedError  # 未实现异常，子类必须重写
