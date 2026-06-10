# HiSparse协调器模块
# 本模块实现了HiSparse（高性能稀疏注意力）的协调器，用于管理
# 设备端KV缓存与主机端KV缓存之间的数据传输、缓冲区分配、
# Token到设备缓冲区的映射以及Top-K页面的换入操作。
# 支持DeepSeek V4和非V4两种HiSparse模式，实现高效的稀疏注意力计算。

# to be combined with the sparse coordinator class and sparse algorithm family
# 将与稀疏协调器类和稀疏算法族合并

import logging  # 导入日志模块 # 导入日志模块
from typing import List, NamedTuple, Union  # 导入类型注解 # 导入类型注解

import torch  # 导入PyTorch库 # 导入PyTorch库

from sglang.srt.managers.schedule_batch import Req  # 导入请求类 # 导入请求类
from sglang.srt.mem_cache.hisparse_memory_pool import (  # 导入HiSparse内存池相关类 # 导入HiSparse内存池相关类
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    DeepSeekV4SingleKVPoolHost,
    HiSparseDSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost  # 导入MLA主机内存池 # 导入MLA主机内存池
from sglang.srt.utils import get_device_module  # 导入设备模块获取工具 # 导入设备模块获取工具

device_module = get_device_module()  # 获取当前设备模块 # 获取当前设备模块

from sglang.jit_kernel.hisparse import (  # 导入HiSparse JIT内核 # 导入HiSparse JIT内核
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # 导入请求到Token池 # 导入请求到Token池

logger = logging.getLogger(__name__)  # 创建日志记录器 # 创建日志记录器


class HiSparseAct(NamedTuple):  # HiSparse活动记录 # HiSparse活动记录
    start_event: device_module.Event  # 开始事件 # 开始事件
    finish_event: device_module.Event  # 完成事件 # 完成事件
    req: Req  # 请求对象 # 请求对象


class HiSparseTokenStats(NamedTuple):  # HiSparse Token统计信息 # HiSparse Token统计信息
    device_tokens: int  # 设备端Token数 # 设备端Token数
    device_token_usage: float  # 设备端Token使用率 # 设备端Token使用率
    host_tokens: int  # 主机端Token数 # 主机端Token数
    host_token_usage: float  # 主机端Token使用率 # 主机端Token使用率


class HiSparseCoordinator:  # HiSparse协调器类 # HiSparse协调器类
    def __init__(  # 初始化方法 # 初始化方法
        self,
        req_to_token_pool: ReqToTokenPool,  # 请求到Token的映射池 # 请求到Token的映射池
        token_to_kv_pool_allocator: Union[  # Token到KV池分配器 # Token到KV池分配器
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,  # top-k值 # top-k值
        device_buffer_size: int,  # 设备缓冲区大小 # 设备缓冲区大小
        device: str,  # 设备名称 # 设备名称
        tp_group,  # 张量并行组 # 张量并行组
        host_to_device_ratio: int = 2,  # 主机到设备比例，默认2 # 主机到设备比例，默认2
    ):
        self.req_to_token_pool = req_to_token_pool  # 保存请求到Token池 # 保存请求到Token池
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator  # 保存KV池分配器 # 保存KV池分配器
        self.top_k = top_k  # 保存top-k值 # 保存top-k值
        self.device_buffer_size = device_buffer_size  # 保存设备缓冲区大小 # 保存设备缓冲区大小
        self.device = device  # 保存设备名称 # 保存设备名称
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio  # 获取压缩比 # 获取压缩比

        self.is_dsv4_hisparse = isinstance(  # 判断是否为DeepSeek V4 HiSparse # 判断是否为DeepSeek V4 HiSparse
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:  # 如果是DSV4 HiSparse # 如果是DSV4 HiSparse
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache  # 获取设备端KV缓存 # 获取设备端KV缓存
            host_size = self.token_to_kv_pool_allocator.size_full // self.compress_ratio  # 计算主机端大小 # 计算主机端大小
            self.mem_pool_host = DeepSeekV4SingleKVPoolHost(  # 创建DSV4主机端KV池 # 创建DSV4主机端KV池
                self.mem_pool_device,
                host_size,
                page_size=self.mem_pool_device.page_size,
            )
            self.item_size_bytes = (  # 计算每个条目的字节数 # 计算每个条目的字节数
                self.mem_pool_host.kv_cache_total_dim
                * self.mem_pool_host.dtype.itemsize
            )
        else:  # 否则 # 否则
            assert isinstance(  # 断言为HiSparseTokenToKVPoolAllocator # 断言为HiSparseTokenToKVPoolAllocator
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (  # 获取HiSparse DSA设备端KV缓存 # 获取HiSparse DSA设备端KV缓存
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(  # 创建MLA主机端KV池 # 创建MLA主机端KV池
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size  # 获取每个条目的字节数 # 获取每个条目的字节数
        self.page_size = self.mem_pool_device.page_size  # 获取页面大小 # 获取页面大小

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]  # 获取最大请求数 # 获取最大请求数
        max_context_len = req_to_token_pool.max_context_len  # 获取最大上下文长度 # 获取最大上下文长度
        max_compressed_context_len = (  # 计算最大压缩上下文长度 # 计算最大压缩上下文长度
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        # 为新Token保留一个额外的页面
        self.padded_buffer_size = (  # 计算填充后的缓冲区大小 # 计算填充后的缓冲区大小
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(  # 创建请求到设备缓冲区的映射 # 创建请求到设备缓冲区的映射
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(  # 创建请求设备缓冲区大小张量 # 创建请求设备缓冲区大小张量
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(  # 创建请求到主机池的映射 # 创建请求到主机池的映射
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(  # 创建请求主机池已分配长度张量 # 创建请求主机池已分配长度张量
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()  # 创建写入暂存流 # 创建写入暂存流
        self.decode_backup_stream = device_module.Stream()  # 创建解码备份流 # 创建解码备份流
        self.ack_staging_queue: List[HiSparseAct] = []  # 确认暂存队列 # 确认暂存队列
        self.decode_producer_stream = None  # 解码生产者流 # 解码生产者流
        self._backup_done_event = device_module.Event()  # 备份完成事件 # 备份完成事件
        self._has_pending_backup = False  # 是否有待完成的备份 # 是否有待完成的备份

        self.tp_group = tp_group  # 保存张量并行组 # 保存张量并行组
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)  # 获取张量并行世界大小 # 获取张量并行世界大小

        # initialize data structures for swap-in kernel
        # 初始化换入内核的数据结构
        layer_num = self.mem_pool_device.layer_num  # 获取层数 # 获取层数
        self.req_device_buffer_tokens = torch.full(  # 创建请求设备缓冲区Token张量 # 创建请求设备缓冲区Token张量
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(  # 创建请求设备缓冲区Token位置张量 # 创建请求设备缓冲区Token位置张量
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(  # 创建LRU初始化张量 # 创建LRU初始化张量
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (  # 创建LRU槽位张量 # 创建LRU槽位张量
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(  # 创建设备缓冲区范围张量(int32) # 创建设备缓冲区范围张量(int32)
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        # 为swap_in_selected_pages预分配的输出缓冲区（CUDA图安全）
        self.top_k_device_locs_buffer = torch.full(  # top-k设备位置缓冲区 # top-k设备位置缓冲区
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(  # 原始索引缓冲区 # 原始索引缓冲区
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        # 标量张量：批次中真实（非填充）请求的数量。
        # 在每次图重放前更新，使填充块提前返回。
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)  # 真实请求数 # 真实请求数

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        # CPU标志：True表示"在下一个解码步骤跳过备份"，因为暂存已经备份了
        # 所有预填充Token。在一个步骤后清除。
        self._skip_first_backup = [False] * max_num_req_slots  # 跳过首次备份标志列表 # 跳过首次备份标志列表

    def set_decode_producer_stream(self, stream) -> None:  # 设置解码生产者流 # 设置解码生产者流
        self.decode_producer_stream = stream  # 保存解码生产者流 # 保存解码生产者流

    def get_token_stats(self) -> HiSparseTokenStats:  # 获取Token统计信息 # 获取Token统计信息
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator  # 获取设备端分配器 # 获取设备端分配器
        device_capacity = device_allocator.size  # 获取设备端容量 # 获取设备端容量
        device_tokens = device_capacity - device_allocator.available_size()  # 计算已使用设备Token数 # 计算已使用设备Token数
        host_capacity = self.mem_pool_host.size  # 获取主机端容量 # 获取主机端容量
        host_tokens = host_capacity - self.mem_pool_host.available_size()  # 计算已使用主机Token数 # 计算已使用主机Token数
        return HiSparseTokenStats(  # 返回统计信息 # 返回统计信息
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:  # 将请求接纳到暂存区 # 将请求接纳到暂存区
        req.hisparse_staging = True  # 标记请求正在暂存中 # 标记请求正在暂存中

        full_kv_indices = self.req_to_token_pool.req_to_token[  # 获取完整的KV索引 # 获取完整的KV索引
            req.req_pool_idx, : len(req.fill_ids)
        ].to(dtype=torch.int64, copy=True)
        device_indices = (  # 将完整索引转换为HiSparse设备索引 # 将完整索引转换为HiSparse设备索引
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)  # 获取预填充长度 # 获取预填充长度
        host_indices = self.mem_pool_host.alloc_paged_token_slots(  # 在主机端分配分页Token槽位 # 在主机端分配分页Token槽位
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()  # 创建开始事件 # 创建开始事件
        finish_event = device_module.Event()  # 创建完成事件 # 创建完成事件
        start_event.record()  # 记录开始事件 # 记录开始事件
        with device_module.stream(self.write_staging_stream):  # 在写入暂存流中执行 # 在写入暂存流中执行
            start_event.wait(self.write_staging_stream)  # 等待开始事件 # 等待开始事件
            self.mem_pool_host.backup_from_device_all_layer(  # 从设备备份所有层到主机 # 从设备备份所有层到主机
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()  # 记录完成事件 # 记录完成事件
            if host_indices.is_cuda:  # 如果主机索引在CUDA上 # 如果主机索引在CUDA上
                host_indices.record_stream(self.write_staging_stream)  # 记录流 # 记录流
            if device_indices.is_cuda:  # 如果设备索引在CUDA上 # 如果设备索引在CUDA上
                device_indices.record_stream(self.write_staging_stream)  # 记录流 # 记录流

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))  # 添加到确认队列 # 添加到确认队列

    def admit_request_direct(self, req: Req) -> None:  # 直接接纳请求（不经暂存） # 直接接纳请求（不经暂存）
        """Direct-to-host path: KV data already resides in host pool via RDMA.
        直接到主机路径：KV数据已通过RDMA驻留在主机池中。

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.
        完全跳过暂存DMA。仅分配小型设备缓冲区（4KB）用于解码时换入，
        然后标记请求为就绪。主机索引已写入req_to_host_pool。

        Metadata fixups after alloc_device_buffer():
        在alloc_device_buffer()后的元数据修复：
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        - alloc_device_buffer()设置device_buffer_tokens = [0, 1, ..., buf_size-1]，
          这告诉换入内核这些Token已缓存在设备缓冲区中。在暂存路径中这是正确的
          （预填充已填充缓冲区），但这里缓冲区是空的。
        """
        if self.is_dsv4_hisparse:  # 如果是DSV4 HiSparse # 如果是DSV4 HiSparse
            # TODO(dsv4): wire PD direct-to-host. Needs (a) load_to_device_per_layer
            # TODO(dsv4): 连接PD直接到主机。需要(a) load_to_device_per_layer
            raise NotImplementedError(
                "PD direct-to-host admission is not supported for dsv4 hisparse yet."
            )  # 尚不支持DSV4 HiSparse的PD直接到主机接纳 # 尚不支持DSV4 HiSparse的PD直接到主机接纳

        self.alloc_device_buffer(req)  # 分配设备缓冲区 # 分配设备缓冲区

        if req.kv_allocated_len <= self.device_buffer_size:  # 如果KV分配长度<=设备缓冲区大小 # 如果KV分配长度<=设备缓冲区大小
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # 短序列（seq_len <= device_buffer_size）：内核快速路径直接返回
            # device_buffer_locs而不进行任何主机加载，因此我们必须将所有Token
            # 从主机池预加载到设备缓冲区
            # TODO(hzh0425): Optimize this.
            # TODO(hzh0425): 优化此处。
            self._preload_to_device_buffer(req)  # 预加载到设备缓冲区 # 预加载到设备缓冲区
        else:  # 否则 # 否则
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            # 长序列：将device_buffer_tokens重置为-1，使内核看到所有槽位为空
            # -> 每次top-k查找都是未命中 -> 主机加载。
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1  # 重置设备缓冲区Token为-1 # 重置设备缓冲区Token为-1

        req.hisparse_staging = False  # 标记暂存完成 # 标记暂存完成
        self._skip_first_backup[req.req_pool_idx] = True  # 设置跳过首次备份标志 # 设置跳过首次备份标志
        logger.debug("HiSparse: admitting request %s directly", req.rid)  # 记录调试日志 # 记录调试日志

    def _preload_to_device_buffer(self, req: Req) -> None:  # 预加载Token到设备缓冲区 # 预加载Token到设备缓冲区
        """Preload all tokens from host pool into the device buffer.
        从主机池预加载所有Token到设备缓冲区。"""
        n = req.kv_allocated_len  # 获取KV分配长度 # 获取KV分配长度
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]  # 获取主机索引 # 获取主机索引
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]  # 获取设备位置 # 获取设备位置

        for layer_id in range(self.mem_pool_device.layer_num):  # 遍历每一层 # 遍历每一层
            self.mem_pool_host.load_to_device_per_layer(  # 按层从主机加载到设备 # 按层从主机加载到设备
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:  # 分配设备缓冲区 # 分配设备缓冲区
        if self.is_dsv4_hisparse:  # 如果是DSV4 HiSparse # 如果是DSV4 HiSparse
            allocated_len = len(req.fill_ids)  # 使用填充ID长度 # 使用填充ID长度
            alloc_size = self.padded_buffer_size  # 使用填充后的缓冲区大小 # 使用填充后的缓冲区大小
        else:  # 否则 # 否则
            allocated_len = req.kv_allocated_len  # 使用KV分配长度 # 使用KV分配长度
            page_size = self.mem_pool_device.page_size  # 获取页面大小 # 获取页面大小
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            # 仅分配当前Token所需的空间（页面对齐）。
            # 当预填充已填满device_buffer_size时，包含保留页面。
            alloc_size = min(  # 计算分配大小 # 计算分配大小
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:  # 如果达到设备缓冲区大小 # 如果达到设备缓冲区大小
                alloc_size = self.padded_buffer_size  # 使用填充后的缓冲区大小 # 使用填充后的缓冲区大小

        compressed_logical_indices = (  # 计算压缩逻辑索引 # 计算压缩逻辑索引
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)  # 获取压缩长度 # 获取压缩长度

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(  # 分配设备缓冲区索引 # 分配设备缓冲区索引
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:  # 如果分配失败 # 如果分配失败
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )  # 记录错误日志 # 记录错误日志
            raise RuntimeError("HiSparse alloc_device_buffer returned None")  # 抛出运行时错误 # 抛出运行时错误

        buffer_indices = buffer_indices.to(torch.int32)  # 转换为int32 # 转换为int32
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices  # 保存设备缓冲区索引 # 保存设备缓冲区索引
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size  # 保存设备缓冲区大小 # 保存设备缓冲区大小

        self.req_device_buffer_tokens[  # 设置设备缓冲区Token # 设置设备缓冲区Token
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (  # 设置设备缓冲区Token位置 # 设置设备缓冲区Token位置
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(  # 扩展设备缓冲区 # 扩展设备缓冲区
        self,
        seq_lens: torch.Tensor,  # 序列长度张量 # 序列长度张量
        req_pool_indices: torch.Tensor,  # 请求池索引张量 # 请求池索引张量
        seq_lens_cpu: torch.Tensor,  # CPU端序列长度张量 # CPU端序列长度张量
        req_pool_indices_cpu: torch.Tensor,  # CPU端请求池索引张量 # CPU端请求池索引张量
    ) -> torch.Tensor:  # 返回保留位置的张量 # 返回保留位置的张量
        """Grow device buffers for requests whose sequence length exceeds current capacity.
        为序列长度超过当前容量的请求扩展设备缓冲区。"""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]  # 获取当前容量 # 获取当前容量
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size  # 识别短请求 # 识别短请求
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)  # 识别需要扩展的请求 # 识别需要扩展的请求

        if torch.any(needs_grow_cpu):  # 如果有需要扩展的请求 # 如果有需要扩展的请求
            page_size = self.mem_pool_device.page_size  # 获取页面大小 # 获取页面大小
            grow_indices = torch.where(needs_grow_cpu)[0]  # 获取需要扩展的索引 # 获取需要扩展的索引

            # Compute all grow sizes on CPU, then do a single bulk allocation
            # 在CPU上计算所有扩展大小，然后进行一次批量分配
            req_idxs = []  # 请求索引列表 # 请求索引列表
            old_caps = []  # 旧容量列表 # 旧容量列表
            new_caps = []  # 新容量列表 # 新容量列表
            grow_sizes = []  # 扩展大小列表 # 扩展大小列表
            total_grow = 0  # 总扩展大小 # 总扩展大小
            for i in grow_indices.tolist():  # 遍历需要扩展的索引 # 遍历需要扩展的索引
                req_idx = int(req_pool_indices_cpu[i])  # 获取请求索引 # 获取请求索引
                current_cap = int(current_caps[i])  # 获取当前容量 # 获取当前容量
                seq_len = int(seq_lens_cpu[i])  # 获取序列长度 # 获取序列长度

                new_cap = min(  # 计算新容量 # 计算新容量
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:  # 如果达到设备缓冲区大小 # 如果达到设备缓冲区大小
                    new_cap = self.padded_buffer_size  # 使用填充后的缓冲区大小 # 使用填充后的缓冲区大小
                grow_size = new_cap - current_cap  # 计算扩展大小 # 计算扩展大小
                if grow_size <= 0:  # 如果扩展大小<=0则跳过 # 如果扩展大小<=0则跳过
                    continue  # 继续下一个 # 继续下一个
                req_idxs.append(req_idx)  # 添加请求索引 # 添加请求索引
                old_caps.append(current_cap)  # 添加旧容量 # 添加旧容量
                new_caps.append(new_cap)  # 添加新容量 # 添加新容量
                grow_sizes.append(grow_size)  # 添加扩展大小 # 添加扩展大小
                total_grow += grow_size  # 累加总扩展大小 # 累加总扩展大小

            if total_grow > 0:  # 如果总扩展大小>0 # 如果总扩展大小>0
                all_new_indices = (  # 批量分配新索引 # 批量分配新索引
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:  # 如果分配失败 # 如果分配失败
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )  # 记录错误日志 # 记录错误日志
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )  # 抛出运行时错误 # 抛出运行时错误

                offset = 0  # 初始化偏移量 # 初始化偏移量
                for req_idx, current_cap, new_cap, grow_size in zip(  # 遍历每个请求 # 遍历每个请求
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]  # 获取分配的索引块 # 获取分配的索引块
                    offset += grow_size  # 更新偏移量 # 更新偏移量
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk  # 更新设备缓冲区映射 # 更新设备缓冲区映射
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk  # 更新设备缓冲区Token位置 # 更新设备缓冲区Token位置
                    self.req_device_buffer_size[req_idx] = new_cap  # 更新设备缓冲区大小 # 更新设备缓冲区大小

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)  # 计算保留位置 # 计算保留位置
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]  # 返回保留位置对应的缓冲区索引 # 返回保留位置对应的缓冲区索引

    def has_ongoing_staging(self) -> bool:  # 是否有正在进行的暂存 # 是否有正在进行的暂存
        return len(self.ack_staging_queue) > 0  # 返回暂存队列是否非空 # 返回暂存队列是否非空

    def collect_ready_reqs(self) -> List[Req]:  # 收集已就绪的请求 # 收集已就绪的请求
        ready_reqs: List[Req] = []  # 初始化就绪请求列表 # 初始化就绪请求列表
        if len(self.ack_staging_queue) == 0:  # 如果暂存队列为空 # 如果暂存队列为空
            return ready_reqs  # 返回空列表 # 返回空列表

        finish_count = 0  # 初始化完成计数 # 初始化完成计数
        for _, finish_event, _ in self.ack_staging_queue:  # 遍历确认队列 # 遍历确认队列
            if not finish_event.query():  # 如果事件未完成 # 如果事件未完成
                break  # 跳出循环 # 跳出循环
            finish_count += 1  # 增加完成计数 # 增加完成计数
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")  # 创建队列大小张量 # 创建队列大小张量
        if self.tp_world_size > 1:  # 如果张量并行世界大小>1 # 如果张量并行世界大小>1
            # synchronize TP workers to make sure the same update to scheduler
            # 同步TP工作节点以确保对调度器的相同更新
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )  # 全归约操作取最小值 # 全归约操作取最小值
        finish_count = int(queue_size.item())  # 获取完成计数 # 获取完成计数
        while finish_count > 0:  # 处理已完成的请求 # 处理已完成的请求
            _, _, req = self.ack_staging_queue.pop(0)  # 从队列头部弹出 # 从队列头部弹出
            # prepare device buffer and update req
            # 准备设备缓冲区并更新请求
            self.alloc_device_buffer(req)  # 分配设备缓冲区 # 分配设备缓冲区
            self._skip_first_backup[req.req_pool_idx] = True  # 设置跳过首次备份标志 # 设置跳过首次备份标志
            req.hisparse_staging = False  # 标记暂存完成 # 标记暂存完成
            finish_count -= 1  # 减少完成计数 # 减少完成计数
            ready_reqs.append(req)  # 添加到就绪请求列表 # 添加到就绪请求列表
        return ready_reqs  # 返回就绪请求列表 # 返回就绪请求列表

    def map_last_loc_to_buffer(  # 将最后位置映射到缓冲区 # 将最后位置映射到缓冲区
        self,
        seq_lens: torch.Tensor,  # 序列长度张量 # 序列长度张量
        out_cache_loc: torch.Tensor,  # 输出缓存位置张量 # 输出缓存位置张量
        req_pool_indices: torch.Tensor,  # 请求池索引张量 # 请求池索引张量
        seq_lens_cpu: torch.Tensor,  # CPU端序列长度张量 # CPU端序列长度张量
        req_pool_indices_cpu: torch.Tensor,  # CPU端请求池索引张量 # CPU端请求池索引张量
    ) -> None:  # 无返回值 # 无返回值
        self._eager_backup_previous_token(  # 执行急切备份前一个Token # 执行急切备份前一个Token
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:  # 如果不是DSV4 HiSparse # 如果不是DSV4 HiSparse
            # Grow device buffers if needed and resolve the latest-token slot.
            # 如果需要则扩展设备缓冲区并解析最新Token的槽位。
            reserved_buffer_loc = self._grow_device_buffers(  # 扩展设备缓冲区 # 扩展设备缓冲区
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)  # 更新Token位置 # 更新Token位置

            # No need to clear prior mappings: the only consumer of the mapping
            # for past tokens is the swap-in kernel, and it goes through
            # top_k_device_locs returned by swap_in_selected_pages -- not via
            # mapping[old_out_cache_loc] -- so stale entries are harmless.
            # 无需清除先前的映射：过去Token映射的唯一消费者是换入内核，
            # 它通过swap_in_selected_pages返回的top_k_device_locs访问，
            # 而不是通过mapping[old_out_cache_loc]，因此过期条目无害。
            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(  # 获取最后位置的压缩索引 # 获取最后位置的压缩索引
                out_cache_loc
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc  # 更新索引映射 # 更新索引映射
            return  # 返回 # 返回

        active_reqs = seq_lens % self.compress_ratio == 0  # 识别活跃请求（对齐到压缩比） # 识别活跃请求（对齐到压缩比）
        if not torch.any(active_reqs):  # 如果没有活跃请求 # 如果没有活跃请求
            return  # 返回 # 返回

        active_seq_lens = seq_lens[active_reqs]  # 获取活跃请求的序列长度 # 获取活跃请求的序列长度
        active_out_cache_loc = out_cache_loc[active_reqs]  # 获取活跃请求的输出缓存位置 # 获取活跃请求的输出缓存位置
        active_req_pool_indices = req_pool_indices[active_reqs]  # 获取活跃请求的池索引 # 获取活跃请求的池索引

        compressed_seq_lens = active_seq_lens // self.compress_ratio  # 计算压缩序列长度 # 计算压缩序列长度
        reserved_positions = (compressed_seq_lens - 1).clamp(  # 计算保留位置 # 计算保留位置
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[  # 获取保留位置的缓冲区索引 # 获取保留位置的缓冲区索引
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)  # 更新Token位置 # 更新Token位置

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(  # 获取最后位置的压缩索引 # 获取最后位置的压缩索引
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (  # 更新索引映射 # 更新索引映射
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(  # 急切备份前一个Token # 急切备份前一个Token
        self,
        seq_lens: torch.Tensor,  # 序列长度张量 # 序列长度张量
        req_pool_indices: torch.Tensor,  # 请求池索引张量 # 请求池索引张量
        seq_lens_cpu: torch.Tensor,  # CPU端序列长度张量 # CPU端序列长度张量
        req_pool_indices_cpu: torch.Tensor,  # CPU端请求池索引张量 # CPU端请求池索引张量
    ) -> None:  # 无返回值 # 无返回值
        """Back up the previous compressed token to host memory.
        将前一个压缩Token备份到主机内存。

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.
        每个新生成的压缩Token（每`compress_ratio`个解码步骤产生一个）必须
        备份到主机，以便换入内核后续恢复。

        Two cases are skipped:
        两种情况被跳过：
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - 暂存后的第一个解码步骤：所有预填充Token在暂存期间已备份，
          因此没有新内容需要保存。
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        - `(seq_len - 1) % compress_ratio != 0`的步骤：此步骤未产生新的压缩Token。
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        # 构建需要主机备份的批次位置列表。
        # 跳过暂存后的第一个解码步骤（预填充已备份），
        # 以及未产生新压缩Token的非对齐步骤。
        backup_indices = []  # 初始化备份索引列表 # 初始化备份索引列表
        for i in range(len(seq_lens_cpu)):  # 遍历所有请求 # 遍历所有请求
            req_idx = int(req_pool_indices_cpu[i])  # 获取请求索引 # 获取请求索引
            if self._skip_first_backup[req_idx]:  # 如果需要跳过首次备份 # 如果需要跳过首次备份
                self._skip_first_backup[req_idx] = False  # 清除标志 # 清除标志
                continue  # 继续下一个 # 继续下一个
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:  # 如果需要对齐备份 # 如果需要对齐备份
                backup_indices.append(i)  # 添加到备份索引列表 # 添加到备份索引列表

        if not backup_indices:  # 如果没有需要备份的请求 # 如果没有需要备份的请求
            return  # 返回 # 返回

        backup_indices_gpu = torch.tensor(  # 将备份索引转换为GPU张量 # 将备份索引转换为GPU张量
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]  # 获取备份请求索引 # 获取备份请求索引

        # The previous compressed token's position and its device buffer slot:
        # 前一个压缩Token的位置及其设备缓冲区槽位：
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - short: slot = compressed_pos          （在常规缓冲区内）
        #  - long:  slot = device_buffer_size      (the reserved slot)
        #  - long:  slot = device_buffer_size      （保留槽位）
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1  # 计算前一个序列长度 # 计算前一个序列长度
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio  # 计算压缩后的前序列长度 # 计算压缩后的前序列长度
        actual_compressed_pos = compressed_prev_seq_lens - 1  # 计算实际压缩位置 # 计算实际压缩位置

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)  # 裁剪缓冲区槽位 # 裁剪缓冲区槽位

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]  # 获取设备位置 # 获取设备位置

        host_locs_list = []  # 初始化主机位置列表 # 初始化主机位置列表
        for i in backup_indices:  # 遍历备份索引 # 遍历备份索引
            req_idx = int(req_pool_indices_cpu[i])  # 获取请求索引 # 获取请求索引
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1  # 计算起始位置 # 计算起始位置
            host_locs = self.mem_pool_host.alloc_paged_token_slots(  # 在主机端分配分页Token槽位 # 在主机端分配分页Token槽位
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)  # 添加到主机位置列表 # 添加到主机位置列表
        host_locs = torch.cat(host_locs_list)  # 拼接所有主机位置 # 拼接所有主机位置

        self.wait_for_pending_backup()  # 等待待完成的备份 # 等待待完成的备份
        schedule_stream = device_module.current_stream()  # 获取当前调度流 # 获取当前调度流
        with device_module.stream(self.decode_backup_stream):  # 在解码备份流中执行 # 在解码备份流中执行
            self.decode_backup_stream.wait_stream(schedule_stream)  # 等待调度流 # 等待调度流
            if self.decode_producer_stream is not None:  # 如果解码生产者流存在 # 如果解码生产者流存在
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)  # 等待解码生产者流 # 等待解码生产者流
            self.mem_pool_host.backup_from_device_all_layer(  # 从设备备份所有层到主机 # 从设备备份所有层到主机
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()  # 记录备份完成事件 # 记录备份完成事件
            if host_locs.is_cuda:  # 如果主机位置在CUDA上 # 如果主机位置在CUDA上
                host_locs.record_stream(self.decode_backup_stream)  # 记录流 # 记录流
            if backup_req_indices.is_cuda:  # 如果备份请求索引在CUDA上 # 如果备份请求索引在CUDA上
                backup_req_indices.record_stream(self.decode_backup_stream)  # 记录流 # 记录流
            if actual_compressed_pos.is_cuda:  # 如果实际压缩位置在CUDA上 # 如果实际压缩位置在CUDA上
                actual_compressed_pos.record_stream(self.decode_backup_stream)  # 记录流 # 记录流
            if device_locs.is_cuda:  # 如果设备位置在CUDA上 # 如果设备位置在CUDA上
                device_locs.record_stream(self.decode_backup_stream)  # 记录流 # 记录流
        self._has_pending_backup = True  # 设置有待完成的备份标志 # 设置有待完成的备份标志

    def wait_for_pending_backup(self) -> None:  # 等待待完成的备份 # 等待待完成的备份
        if not self._has_pending_backup:  # 如果没有待完成的备份 # 如果没有待完成的备份
            return  # 返回 # 返回
        self._backup_done_event.wait(device_module.current_stream())  # 等待备份完成事件 # 等待备份完成事件
        self._has_pending_backup = False  # 清除待完成备份标志 # 清除待完成备份标志

    def naive_load_topk(  # 朴素加载Top-K Token # 朴素加载Top-K Token
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引张量 # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度张量 # 序列长度张量
        top_k_tokens: torch.Tensor,  # Top-K Token位置张量 # Top-K Token位置张量
        layer_id: int,  # 层ID # 层ID
    ) -> torch.Tensor:  # 返回设备KV缓存索引 # 返回设备KV缓存索引
        """Load top-k selected tokens into device memory and return their device indices.
        将Top-K选中的Token加载到设备内存并返回其设备索引。

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.
        这是一个朴素的逐请求循环实现，用于调试/验证。
        生产代码使用swap_in_selected_pages（JIT CUDA内核）代替。

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).
        注意：不支持dsv4 hisparse — DeepSeekV4SingleKVPoolHost没有
        load_to_device_per_layer，且索引位于压缩空间中。目前仅在
        test_hisparse_unit.py（非dsv4路径）中用作内核参考。

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            req_pool_indices: 每个请求的池索引。形状：(num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            seq_lens: 每个请求的序列长度。形状：(num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            top_k_tokens: 每个请求选中的Token位置。形状：(num_reqs, top_k)
            layer_id: The layer to load KV cache for.
            layer_id: 要加载KV缓存的层。

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
            选中Token的设备KV缓存索引。形状：(num_reqs, top_k)
        """
        assert (  # 断言不是DSV4 HiSparse # 断言不是DSV4 HiSparse
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)  # 获取请求数量 # 获取请求数量
        top_k_indices = torch.full(  # 创建Top-K索引张量 # 创建Top-K索引张量
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):  # 遍历每个请求 # 遍历每个请求
            seq_len = int(seq_lens[i].item())  # 获取序列长度 # 获取序列长度
            top_n = min(seq_len, self.top_k)  # 计算实际Top-N值 # 计算实际Top-N值
            if top_n == 0:  # 如果Top-N为0则跳过 # 如果Top-N为0则跳过
                continue  # 继续下一个 # 继续下一个

            req_idx = int(req_pool_indices[i].item())  # 获取请求索引 # 获取请求索引
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)  # 获取选中的Token位置 # 获取选中的Token位置

            assert torch.all(  # 断言选中位置非负 # 断言选中位置非负
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (  # 断言选中位置在范围内 # 断言选中位置在范围内
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:  # 如果序列长度<=设备缓冲区大小 # 如果序列长度<=设备缓冲区大小
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]  # 直接从缓冲区获取索引 # 直接从缓冲区获取索引
            else:  # 否则需要从主机加载 # 否则需要从主机加载
                device_indices = torch.empty(  # 创建空的设备索引张量 # 创建空的设备索引张量
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)  # 识别最新Token # 识别最新Token
                needs_host_load = ~is_latest_token  # 识别需要从主机加载的Token # 识别需要从主机加载的Token

                device_indices[is_latest_token] = self.req_to_device_buffer[  # 设置最新Token的设备索引 # 设置最新Token的设备索引
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())  # 计算需要加载的数量 # 计算需要加载的数量
                if num_to_load > 0:  # 如果有需要加载的Token # 如果有需要加载的Token
                    tokens_to_load = selected_tokens[needs_host_load]  # 获取需要加载的Token位置 # 获取需要加载的Token位置
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]  # 获取主机位置 # 获取主机位置

                    invalid_mask = host_locs < 0  # 识别无效的主机位置 # 识别无效的主机位置
                    if torch.any(invalid_mask):  # 如果有无效位置 # 如果有无效位置
                        bad_positions = tokens_to_load[invalid_mask].tolist()  # 获取无效位置列表 # 获取无效位置列表
                        raise AssertionError(  # 抛出断言错误 # 抛出断言错误
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]  # 获取缓冲区位置 # 获取缓冲区位置
                    device_indices[needs_host_load] = buffer_locs  # 设置需要加载Token的设备索引 # 设置需要加载Token的设备索引

                    self.mem_pool_host.load_to_device_per_layer(  # 从主机按层加载到设备 # 从主机按层加载到设备
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)  # 存储Top-K索引 # 存储Top-K索引

        return top_k_indices  # 返回Top-K索引 # 返回Top-K索引

    def abort_staging_request(self, req: Req) -> None:  # 中止暂存中的请求 # 中止暂存中的请求
        """Remove a request from the staging queue and free its host + device resources.
        从暂存队列中移除请求并释放其主机+设备资源。

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        当中止一个已接纳到暂存区但尚未完成的请求时必须调用
        （即req.hisparse_staging为True）。
        """
        # Remove from staging queue
        # 从暂存队列中移除
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]  # 过滤掉目标请求 # 过滤掉目标请求
        # Wait for any in-flight staging DMA to complete before freeing
        # 等待任何进行中的暂存DMA完成后再释放
        self.write_staging_stream.synchronize()  # 同步写入暂存流 # 同步写入暂存流

        prefill_len = len(req.fill_ids)  # 获取预填充长度 # 获取预填充长度
        allocated_locs = self.req_to_token_pool.req_to_token[  # 获取已分配的位置 # 获取已分配的位置
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)  # 释放HiSparse分配 # 释放HiSparse分配

        # Free host memory that was allocated during admit_request_into_staging
        # 释放在admit_request_into_staging期间分配的主机内存
        host_indices = self.mem_pool_host.allocated_host_indices(  # 获取已分配的主机索引 # 获取已分配的主机索引
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:  # 如果有已分配的主机索引 # 如果有已分配的主机索引
            self.mem_pool_host.free(host_indices)  # 释放主机内存 # 释放主机内存
        self.req_to_host_pool[req.req_pool_idx, :] = -1  # 重置主机池映射 # 重置主机池映射
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0  # 重置已分配长度 # 重置已分配长度
        self._skip_first_backup[req.req_pool_idx] = False  # 清除跳过首次备份标志 # 清除跳过首次备份标志
        req.hisparse_staging = False  # 标记暂存完成 # 标记暂存完成

    def retract_req(self, req: Req) -> None:  # 撤回请求 # 撤回请求
        if req.hisparse_staging:  # 如果请求正在暂存中 # 如果请求正在暂存中
            self.abort_staging_request(req)  # 中止暂存请求 # 中止暂存请求
        else:  # 否则 # 否则
            self.request_finished(req)  # 处理请求完成 # 处理请求完成

    def request_finished(self, req: Req):  # 请求完成处理 # 请求完成处理
        # release resources only after the execution of a potential overlapped batch
        # 仅在潜在重叠批次执行完成后释放资源
        if self.decode_producer_stream is not None:  # 如果解码生产者流存在 # 如果解码生产者流存在
            device_module.current_stream().wait_stream(self.decode_producer_stream)  # 等待解码生产者流 # 等待解码生产者流
        self.wait_for_pending_backup()  # 等待待完成的备份 # 等待待完成的备份

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        # 使用kv_allocated_len（而非seqlen）：在推测解码下，分配器可能
        # 超额分配超出已提交的seqlen，这些额外槽位可能携带指向刚通过
        # free_hisparse_indices(all_hi)释放的缓冲区槽位的过期映射条目。
        # 如果不清理，后续的release_kv_cache -> allocator.free -> free_hisparse
        # 路径会再次释放它们（对页面分配器空闲列表的双重释放）。
        allocated_len = req.kv_allocated_len  # 获取KV分配长度 # 获取KV分配长度

        # release memory -- only free actually-allocated buffer indices
        # 释放内存 -- 仅释放实际分配的缓冲区索引
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])  # 获取当前容量 # 获取当前容量
        if current_cap > 0:  # 如果当前容量>0 # 如果当前容量>0
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]  # 获取设备缓冲区索引 # 获取设备缓冲区索引
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])  # 获取所有正索引 # 获取所有正索引
            if all_hi.numel() > 0:  # 如果有正索引 # 如果有正索引
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)  # 释放HiSparse索引 # 释放HiSparse索引

        allocated_locs = self.req_to_token_pool.req_to_token[  # 获取已分配的位置 # 获取已分配的位置
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(  # 转换为压缩位置 # 转换为压缩位置
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0  # 清除索引映射 # 清除索引映射

        host_indices = self.mem_pool_host.allocated_host_indices(  # 获取已分配的主机索引 # 获取已分配的主机索引
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:  # 如果有已分配的主机索引 # 如果有已分配的主机索引
            self.mem_pool_host.free(host_indices)  # 释放主机内存 # 释放主机内存

        # clear req info
        # 清除请求信息
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1  # 重置设备缓冲区Token # 重置设备缓冲区Token
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1  # 重置设备缓冲区Token位置 # 重置设备缓冲区Token位置
        self.req_to_device_buffer[req.req_pool_idx, :] = 0  # 重置设备缓冲区映射 # 重置设备缓冲区映射
        self.req_device_buffer_size[req.req_pool_idx] = 0  # 重置设备缓冲区大小 # 重置设备缓冲区大小
        self.req_to_host_pool[req.req_pool_idx, :] = -1  # 重置主机池映射 # 重置主机池映射
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0  # 重置已分配长度 # 重置已分配长度
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)  # 重置LRU槽位 # 重置LRU槽位
        self._skip_first_backup[req.req_pool_idx] = False  # 清除跳过首次备份标志 # 清除跳过首次备份标志

    def swap_in_selected_pages(  # 换入选中的Top-K页面 # 换入选中的Top-K页面
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引张量 # 请求池索引张量
        compressed_seq_lens: torch.Tensor,  # 压缩序列长度张量 # 压缩序列长度张量
        top_k_result: torch.Tensor,  # Top-K结果张量 # Top-K结果张量
        layer_id: int,  # 层ID # 层ID
    ) -> torch.Tensor:  # 返回Top-K设备索引 # 返回Top-K设备索引
        """Swap selected top-k tokens into device memory and return their indices.
        将选中的Top-K Token换入设备内存并返回其索引。"""
        num_reqs = req_pool_indices.size(0)  # 获取请求数量 # 获取请求数量

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]  # 获取Top-K设备位置缓冲区 # 获取Top-K设备位置缓冲区
        top_k_indices.fill_(-1)  # 填充为-1 # 填充为-1

        # todo, adjustable for performance
        # 待办，可调整以优化性能
        block_size = 1024  # 设置块大小为1024 # 设置块大小为1024
        swap_in_fn = (  # 选择换入函数 # 选择换入函数
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        swap_in_fn(  # 调用换入函数 # 调用换入函数
            top_k_tokens=top_k_result,  # Top-K Token位置 # Top-K Token位置
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],  # 设备缓冲区Token # 设备缓冲区Token
            host_cache_locs=self.req_to_host_pool,  # 主机缓存位置 # 主机缓存位置
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],  # 设备缓冲区位置 # 设备缓冲区位置
            host_cache=self.mem_pool_host.kv_buffer[layer_id],  # 主机端KV缓存 # 主机端KV缓存
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],  # 设备端KV缓存 # 设备端KV缓存
            top_k_device_locs=top_k_indices,  # Top-K设备位置输出 # Top-K设备位置输出
            req_pool_indices=req_pool_indices,  # 请求池索引 # 请求池索引
            seq_lens=compressed_seq_lens,  # 压缩序列长度 # 压缩序列长度
            lru_slots=self.lru_slots[layer_id],  # LRU槽位 # LRU槽位
            item_size_bytes=self.item_size_bytes,  # 每个条目的字节数 # 每个条目的字节数
            num_top_k=self.top_k,  # Top-K值 # Top-K值
            hot_buffer_size=self.device_buffer_size,  # 热缓冲区大小 # 热缓冲区大小
            page_size=1,  # 页面大小 # 页面大小
            block_size=block_size,  # 块大小 # 块大小
            num_real_reqs=self.num_real_reqs,  # 真实请求数 # 真实请求数
        )
        return top_k_indices  # 返回Top-K设备索引 # 返回Top-K设备索引
