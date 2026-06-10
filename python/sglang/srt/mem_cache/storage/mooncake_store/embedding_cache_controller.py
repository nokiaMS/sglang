# 嵌入缓存控制器模块
# 本模块实现了基于Mooncake分布式存储的嵌入向量缓存控制器，
# 提供预取、插入、存在性检查等功能，支持多级缓存（本地+全局）
# 和张量并行下的缓存一致性，用于加速多模态模型中图像嵌入的复用。

import asyncio  # 导入异步IO模块 # 导入异步IO模块
import logging  # 导入日志模块 # 导入日志模块
import threading  # 导入线程模块 # 导入线程模块
import time  # 导入时间模块 # 导入时间模块
from queue import Empty, Queue  # 导入队列相关类 # 导入队列相关类
from typing import List, Optional  # 导入类型提示工具 # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架 # 导入PyTorch深度学习框架

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (  # 导入Mooncake嵌入存储 # 导入Mooncake嵌入存储
    MooncakeEmbeddingStore,
)

logger = logging.getLogger(__name__)  # 获取当前模块日志记录器 # 获取当前模块日志记录器


class ContiguousMemoryAllocator:  # 连续内存分配器类
    """
    A simple allocator to manage variable-sized contiguous blocks
    within a large pre-allocated flat buffer.
    简单的分配器，用于管理大型预分配扁平缓冲区中可变大小的连续块。
    """

    def __init__(self, total_size_bytes: int):  # 初始化方法 # 初始化方法
        self.total_size = total_size_bytes  # 总大小（字节） # 总大小（字节）
        # List of (offset, size) for free blocks # 空闲块列表，元素为(偏移量, 大小) # 空闲块列表
        self.free_blocks = [(0, total_size_bytes)]  # 初始时整个缓冲区为一个空闲块 # 初始时整个缓冲区为一个空闲块
        self.allocated_map = {}  # {handle: (offset, size)} # 已分配映射，句柄到偏移量和大小的映射 # 已分配映射
        self.lock = threading.Lock()  # 线程锁 # 线程锁

    def allocate(self, size_bytes: int) -> Optional[int]:  # 分配指定大小的内存块 # 分配内存块
        with self.lock:  # 加锁保证线程安全 # 加锁
            # Simple First-Fit allocation # 简单的首次适配分配算法 # 简单的首次适配分配算法
            for i, (offset, block_size) in enumerate(self.free_blocks):  # 遍历空闲块 # 遍历空闲块
                if block_size >= size_bytes:  # 如果空闲块足够大 # 如果空闲块足够大
                    # Allocate from this block # 从该块分配 # 从该块分配
                    remaining_size = block_size - size_bytes  # 剩余大小 # 剩余大小
                    if remaining_size > 0:  # 如果有剩余空间 # 如果有剩余空间
                        self.free_blocks[i] = (offset + size_bytes, remaining_size)  # 更新空闲块为剩余部分 # 更新空闲块
                    else:  # 如果刚好用完 # 如果刚好用完
                        self.free_blocks.pop(i)  # 移除该空闲块 # 移除该空闲块
                    return offset  # 返回分配的偏移量 # 返回分配的偏移量
            return None  # 没有足够大的空闲块，返回None # 没有足够大的空闲块

    def free(self, offset: int, size_bytes: int):  # 释放指定偏移量和大小的内存块 # 释放内存块
        with self.lock:  # 加锁保证线程安全 # 加锁
            # Return block and merge adjacent free blocks # 归还块并合并相邻空闲块 # 归还块并合并相邻空闲块
            self.free_blocks.append((offset, size_bytes))  # 将释放的块添加到空闲列表 # 添加到空闲列表
            self.free_blocks.sort()  # 按偏移量排序 # 按偏移量排序

            merged = []  # 合并后的空闲块列表 # 合并后的空闲块列表
            if not self.free_blocks:  # 如果没有空闲块 # 如果没有空闲块
                return  # 返回 # 返回

            curr_offset, curr_size = self.free_blocks[0]  # 初始化当前块 # 初始化当前块
            for next_offset, next_size in self.free_blocks[1:]:  # 遍历后续块 # 遍历后续块
                if curr_offset + curr_size == next_offset:  # 如果当前块与下一块相邻 # 如果相邻
                    curr_size += next_size  # 合并大小 # 合并大小
                else:  # 不相邻 # 不相邻
                    merged.append((curr_offset, curr_size))  # 保存当前块 # 保存当前块
                    curr_offset, curr_size = next_offset, next_size  # 移动到下一块 # 移动到下一块
            merged.append((curr_offset, curr_size))  # 保存最后一个块 # 保存最后一个块
            self.free_blocks = merged  # 更新空闲块列表 # 更新空闲块列表


class EmbeddingPrefetchOperation:  # 嵌入预取操作类
    """Groups all missing images of a request for a single batch GET.
    将请求中所有缺失的图像分组为一次批量GET操作。
    """

    def __init__(self, req_id: str, keys: List[str], ptrs: List[int], sizes: List[int]):  # 初始化方法 # 初始化方法
        self.req_id = req_id  # 请求ID # 请求ID
        self.keys = keys  # 键列表 # 键列表
        self.ptrs = ptrs  # 指针列表 # 指针列表
        self.sizes = sizes  # 大小列表 # 大小列表
        self.is_finished = False  # 是否完成 # 是否完成
        self.success = False  # 是否成功 # 是否成功
        self._lock = threading.Lock()  # 线程锁 # 线程锁

    def mark_done(self, success: bool):  # 标记操作完成 # 标记操作完成
        with self._lock:  # 加锁 # 加锁
            self.success = success  # 设置成功标志 # 设置成功标志
            self.is_finished = True  # 设置完成标志 # 设置完成标志


class EmbeddingInsertOperation:  # 嵌入插入操作类
    """Groups all newly computed images of a request for a single batch PUT.
    将请求中所有新计算的图像分组为一次批量PUT操作。
    """

    def __init__(self, keys: List[str], ptrs: List[int], sizes: List[int]):  # 初始化方法 # 初始化方法
        self.keys = keys  # 键列表 # 键列表
        self.ptrs = ptrs  # 指针列表 # 指针列表
        self.sizes = sizes  # 大小列表 # 大小列表


class EmbeddingCacheController:  # 嵌入缓存控制器类
    def __init__(  # 初始化方法 # 初始化方法
        self,
        tp_rank,  # 张量并行rank # 张量并行rank
        tp_size,  # 张量并行大小 # 张量并行大小
        max_pool_size_gb=4.0,  # 最大池大小（GB），默认4GB # 最大池大小（GB）
        hidden_dims: dict = None,  # 隐藏维度字典 # 隐藏维度字典
        tp_group=None,  # 张量并行进程组 # 张量并行进程组
        all_rank_get=False,  # 是否所有rank都执行GET操作 # 是否所有rank都执行GET操作
    ):
        self.tp_world_size = tp_size  # 张量并行世界大小 # 张量并行世界大小
        self.tp_group = tp_group  # 张量并行进程组 # 张量并行进程组
        self.all_rank_get = all_rank_get  # 是否所有rank都GET # 是否所有rank都GET
        self.hidden_dims = hidden_dims or {}  # 隐藏维度字典，默认空 # 隐藏维度字典
        self.element_size = torch.float32.itemsize  # float32元素大小（字节） # float32元素大小

        # 1. Mooncake Backend & Pinned Buffer # Mooncake后端和固定缓冲区 # Mooncake后端和固定缓冲区
        self.mooncake_store = MooncakeEmbeddingStore()  # 创建Mooncake嵌入存储 # 创建Mooncake嵌入存储
        self.total_pool_size_bytes = int(max_pool_size_gb * 1024**3)  # 总池大小（字节） # 总池大小（字节）
        self.cpu_pool = torch.empty(  # 创建CPU固定内存池 # 创建CPU固定内存池
            self.total_pool_size_bytes, dtype=torch.uint8, pin_memory=True  # uint8类型，启用固定内存 # uint8类型，启用固定内存
        )
        self.mooncake_store.register_buffer(self.cpu_pool)  # 将CPU池注册到Mooncake存储 # 注册缓冲区

        # 2. Variable Size Memory Management # 可变大小内存管理 # 可变大小内存管理
        self.allocator = ContiguousMemoryAllocator(self.total_pool_size_bytes)  # 创建连续内存分配器 # 创建连续内存分配器
        # {hash: (offset, num_tokens, embedding_dim, size_bytes)} # 哈希到元数据的映射 # 哈希到元数据的映射
        self.hash_to_metadata = {}  # 哈希到元数据字典 # 哈希到元数据字典

        # 3. Task Tracking # 任务追踪 # 任务追踪
        self.ongoing_prefetch = {}  # {req_id: EmbeddingPrefetchOperation} # 正在进行的预取操作 # 正在进行的预取操作
        self.prefetch_queue = Queue()  # 预取操作队列 # 预取操作队列
        self.insert_queue = Queue()  # 插入操作队列 # 插入操作队列

        self.lock = threading.Lock()  # 线程锁 # 线程锁
        self.stop_event = threading.Event()  # 停止事件 # 停止事件
        self.io_thread = threading.Thread(target=self._io_loop, daemon=True)  # 创建IO工作线程 # 创建IO工作线程
        self.io_thread.start()  # 启动IO线程 # 启动IO线程

        if self.tp_world_size > 1:  # 如果张量并行度大于1 # 如果张量并行度大于1
            if self.tp_group is None:  # 如果没有提供进程组 # 如果没有提供进程组
                raise ValueError("tp_group must be provided when tp_size > 1")  # 抛出异常 # 抛出异常
            from sglang.srt.distributed.parallel_state import (  # 导入并行状态工具 # 导入并行状态工具
                create_custom_parallel_group,
            )

            group_ranks = torch.distributed.get_process_group_ranks(self.tp_group)  # 获取进程组的rank列表 # 获取进程组的rank列表
            self.prefetch_tp_group = create_custom_parallel_group(  # 创建自定义并行组 # 创建自定义并行组
                group_ranks=group_ranks, backend="gloo"  # 使用gloo后端 # 使用gloo后端
            )
        else:  # 单卡模式 # 单卡模式
            self.prefetch_tp_group = None  # 不需要并行组 # 不需要并行组

    def prefetch(  # 预取嵌入数据 # 预取嵌入数据
        self,
        req_id: str,  # 请求ID # 请求ID
        image_hashes: List[str],  # 图像哈希列表 # 图像哈希列表
        expected_tokens: List[int],  # 期望token数列表 # 期望token数列表
        modality=None,  # 模态类型，可选 # 模态类型
    ):
        """Issues ONE batch GET for all missing images in the request.
        对请求中所有缺失的图像发起一次批量GET操作。
        """
        dim = self.hidden_dims.get(modality) if modality is not None else None  # 获取对应模态的维度 # 获取对应模态的维度
        if not dim:  # 如果维度未知 # 如果维度未知
            logger.warning(
                f"Req {req_id}: Unknown dim for modality={modality}, skipping prefetch (will fallback to ViT)."  # 未知维度，跳过预取 # 未知维度，跳过预取
            )
            return  # 返回 # 返回
        keys, ptrs, sizes = [], [], []  # 初始化键、指针、大小列表 # 初始化键、指针、大小列表

        with self.lock:  # 加锁 # 加锁
            for h, num_tokens in zip(image_hashes, expected_tokens):  # 遍历每个图像哈希和期望token数 # 遍历图像哈希
                if h in self.hash_to_metadata:  # 如果已在本地元数据中 # 如果已在本地元数据中
                    logger.debug(
                        f"Req {req_id}: Hash  already in local metadata, skipping prefetch."  # 已在本地，跳过预取 # 已在本地，跳过预取
                    )
                    continue  # 跳过 # 跳过

                size_bytes = num_tokens * dim * self.element_size  # 计算所需字节数 # 计算所需字节数
                offset = self.allocator.allocate(size_bytes)  # 分配内存 # 分配内存
                if offset is None:  # 如果分配失败 # 如果分配失败
                    continue  # 跳过 # 跳过

                self.hash_to_metadata[h] = (offset, num_tokens, dim, size_bytes)  # 记录元数据 # 记录元数据
                keys.append(h)  # 添加键 # 添加键
                ptrs.append(self.cpu_pool.data_ptr() + offset)  # 添加指针（基地址+偏移） # 添加指针
                sizes.append(size_bytes)  # 添加大小 # 添加大小

            if not keys:  # 如果没有需要预取的键 # 如果没有需要预取的键
                return  # 返回 # 返回

            logger.info(
                f"Req {req_id}: Starting global fetch for {len(keys)} images from Mooncake."  # 开始从Mooncake全局获取 # 开始全局获取
            )

            op = EmbeddingPrefetchOperation(req_id, keys, ptrs, sizes)  # 创建预取操作 # 创建预取操作
            self.ongoing_prefetch[req_id] = op  # 记录进行中的预取操作 # 记录进行中的预取操作
            self.prefetch_queue.put(op)  # 将操作放入预取队列 # 放入预取队列

    def insert_batch(  # 批量插入嵌入数据 # 批量插入嵌入数据
        self, image_hashes: List[str], embedding_tensors: List[torch.Tensor]  # 图像哈希和嵌入张量列表 # 图像哈希和嵌入张量列表
    ):
        """Issues ONE batch PUT for all embeddings computed by this request.
        对本请求计算的所有嵌入发起一次批量PUT操作。

        Note: Even if the embedding exists locally, we still push to Mooncake
        to ensure multi-node cache consistency. Mooncake's batch_put has
        built-in deduplication to avoid redundant transfers.
        注意：即使嵌入已在本地存在，我们仍然推送到Mooncake以确保多节点缓存一致性。
        Mooncake的batch_put具有内置去重功能以避免冗余传输。
        """
        keys, ptrs, sizes = [], [], []  # 初始化键、指针、大小列表 # 初始化键、指针、大小列表
        local_hit_count = 0  # 本地命中计数 # 本地命中计数
        new_count = 0  # 新增计数 # 新增计数
        skipped_count = 0  # 跳过计数 # 跳过计数

        with self.lock:  # 加锁 # 加锁
            for h, tensor in zip(image_hashes, embedding_tensors):  # 遍历哈希和对应张量 # 遍历哈希和张量
                if h in self.hash_to_metadata:  # 如果本地缓存命中 # 如果本地缓存命中
                    # Local cache hit: ensure Mooncake has it # 本地缓存命中：确保Mooncake有该数据 # 确保Mooncake有该数据
                    offset, num_tokens, dim, size_bytes = self.hash_to_metadata[h][:4]  # 获取元数据 # 获取元数据

                    # Still push to Mooncake for multi-node sharing
                    # (Mooncake batch_put will deduplicate if already exists)
                    # 仍然推送到Mooncake用于多节点共享（Mooncake batch_put会去重）
                    keys.append(h)  # 添加键 # 添加键
                    ptrs.append(self.cpu_pool.data_ptr() + offset)  # 添加指针 # 添加指针
                    sizes.append(size_bytes)  # 添加大小 # 添加大小
                    local_hit_count += 1  # 增加本地命中计数 # 增加本地命中计数
                    continue  # 继续下一个 # 继续下一个

                # Local cache miss: allocate and copy # 本地缓存未命中：分配并复制 # 分配并复制
                num_tokens, dim = tensor.shape[0], tensor.shape[1]  # 获取token数和维度 # 获取token数和维度
                size_bytes = num_tokens * dim * self.element_size  # 计算所需字节数 # 计算所需字节数
                offset = self.allocator.allocate(size_bytes)  # 分配内存 # 分配内存
                if offset is None:  # 如果分配失败 # 如果分配失败
                    skipped_count += 1  # 增加跳过计数 # 增加跳过计数
                    continue  # 跳过 # 跳过

                # Copy to pinned pool for RDMA # 复制到固定内存池用于RDMA # 复制到固定内存池用于RDMA
                target_view = (  # 目标视图 # 目标视图
                    self.cpu_pool[offset : offset + size_bytes]  # 切片对应区域 # 切片对应区域
                    .view(torch.float32)  # 视图为float32 # 视图为float32
                    .view(num_tokens, dim)  # 视图为(token数, 维度) # 视图为(token数, 维度)
                )
                target_view.copy_(tensor.cpu())  # 将张量数据复制到目标视图 # 复制数据
                self.hash_to_metadata[h] = (offset, num_tokens, dim, size_bytes)  # 记录元数据 # 记录元数据

                keys.append(h)  # 添加键 # 添加键
                ptrs.append(self.cpu_pool.data_ptr() + offset)  # 添加指针 # 添加指针
                sizes.append(size_bytes)  # 添加大小 # 添加大小
                new_count += 1  # 增加新增计数 # 增加新增计数

            if keys:  # 如果有需要插入的键 # 如果有需要插入的键
                logger.info(
                    f"Global Cache: Inserting {len(keys)} embeddings into Mooncake cluster "  # 插入嵌入到Mooncake集群 # 插入嵌入到集群
                    f"({new_count} new, {local_hit_count} existing for replication, "  # 新增和已有复制的数量 # 新增和已有复制的数量
                    f"{skipped_count} skipped due to allocation failure)"  # 因分配失败跳过的数量 # 因分配失败跳过的数量
                )
                self.insert_queue.put(EmbeddingInsertOperation(keys, ptrs, sizes))  # 将插入操作放入队列 # 放入插入队列

    def _io_loop(self):  # IO工作线程循环 # IO工作线程循环
        """Asynchronous worker handling both Batch GET and Batch PUT.
        处理批量GET和批量PUT的异步工作线程。
        """
        while not self.stop_event.is_set():  # 循环直到收到停止信号 # 循环直到收到停止信号
            processed_any = False  # 本轮是否处理了任何操作 # 本轮是否处理了任何操作

            try:  # 尝试处理预取操作 # 尝试处理预取操作
                op = self.prefetch_queue.get_nowait()  # 非阻塞获取预取操作 # 非阻塞获取预取操作
                results = self.mooncake_store.batch_get(op.keys, op.ptrs, op.sizes)  # 执行批量GET # 执行批量GET
                success_count = sum(results)  # 成功计数 # 成功计数
                logger.info(
                    f"Mooncake GET Finished: Req {op.req_id}, Successfully fetched {success_count}/{len(op.keys)} images."  # GET完成日志 # GET完成日志
                )
                op.mark_done(all(results))  # 标记操作完成 # 标记操作完成
                self.prefetch_queue.task_done()  # 通知队列任务完成 # 通知队列任务完成
                processed_any = True  # 标记已处理 # 标记已处理
            except Empty:  # 队列为空 # 队列为空
                pass  # 忽略 # 忽略

            try:  # 尝试处理插入操作 # 尝试处理插入操作
                op = self.insert_queue.get_nowait()  # 非阻塞获取插入操作 # 非阻塞获取插入操作
                self.mooncake_store.batch_put(op.keys, op.ptrs, op.sizes)  # 执行批量PUT # 执行批量PUT
                logger.info(
                    f"Mooncake PUT Finished: Successfully stored {len(op.keys)} keys in cluster."  # PUT完成日志 # PUT完成日志
                )
                self.insert_queue.task_done()  # 通知队列任务完成 # 通知队列任务完成
                processed_any = True  # 标记已处理 # 标记已处理
            except Empty:  # 队列为空 # 队列为空
                pass  # 忽略 # 忽略

            if not processed_any:  # 如果本轮没有处理任何操作 # 如果本轮没有处理任何操作
                time.sleep(0.001)  # 短暂休眠避免忙等待 # 短暂休眠避免忙等待

    def check_prefetch_progress(self, req_id: str) -> bool:  # 检查预取进度 # 检查预取进度
        """TP-Group barrier: ensures all cards have the request batch ready.
        TP组屏障：确保所有卡都已准备好请求的批次数据。
        """
        local_ready = False  # 本地是否就绪 # 本地是否就绪
        with self.lock:  # 加锁 # 加锁
            if req_id not in self.ongoing_prefetch:  # 如果请求不在进行中的预取中 # 如果请求不在进行中的预取中
                local_ready = True  # 本地已就绪 # 本地已就绪
            else:  # 否则 # 否则
                op = self.ongoing_prefetch[req_id]  # 获取预取操作 # 获取预取操作
                if op.is_finished:  # 如果操作已完成 # 如果操作已完成
                    local_ready = op.success  # 本地就绪取决于操作是否成功 # 本地就绪取决于操作是否成功

        if self.all_rank_get and self.tp_world_size > 1:  # 如果所有rank都GET且并行度大于1 # 如果所有rank都GET
            ready_tensor = torch.tensor(  # 创建就绪状态张量 # 创建就绪状态张量
                [1 if local_ready else 0], dtype=torch.int, device="cpu"  # 1表示就绪，0表示未就绪 # 1表示就绪
            )
            torch.distributed.all_reduce(  # 执行全局归约 # 执行全局归约
                ready_tensor,  # 就绪状态张量 # 就绪状态张量
                op=torch.distributed.ReduceOp.MIN,  # 取最小值（所有rank都就绪才算就绪） # 取最小值
                group=self.prefetch_tp_group,  # 使用预取并行组 # 使用预取并行组
            )
            local_ready = ready_tensor.item() == 1  # 判断是否所有rank都就绪 # 判断是否所有rank都就绪

        if local_ready:  # 如果本地就绪 # 如果本地就绪
            with self.lock:  # 加锁 # 加锁
                self.ongoing_prefetch.pop(req_id, None)  # 移除进行中的预取记录 # 移除进行中的预取记录
            return True  # 返回True表示就绪 # 返回True
        return False  # 返回False表示未就绪 # 返回False

    def get_embeddings(self, image_hashes: List[str]) -> List[torch.Tensor]:  # 获取嵌入张量 # 获取嵌入张量
        """Final reconstruction for model input.
        最终重建用于模型输入的嵌入张量。
        """
        with self.lock:  # 加锁 # 加锁
            tensors = []  # 张量列表 # 张量列表
            for h in image_hashes:  # 遍历图像哈希 # 遍历图像哈希
                offset, num_tokens, dim, size_bytes = self.hash_to_metadata[h]  # 获取元数据 # 获取元数据
                tensors.append(  # 添加张量 # 添加张量
                    self.cpu_pool[offset : offset + size_bytes]  # 切片对应区域 # 切片对应区域
                    .view(torch.float32)  # 视图为float32 # 视图为float32
                    .view(num_tokens, dim)  # 视图为(token数, 维度) # 视图为(token数, 维度)
                )
            return tensors  # 返回张量列表 # 返回张量列表

    async def batch_is_exist(self, image_hashes: List[str]) -> List[bool]:  # 异步批量检查是否存在 # 异步批量检查是否存在
        with self.lock:  # 加锁 # 加锁
            local_results = [h in self.hash_to_metadata for h in image_hashes]  # 检查本地是否存在 # 检查本地是否存在
        local_hit_count = sum(local_results)  # 本地命中数 # 本地命中数

        global_hit_count = 0  # 全局命中数 # 全局命中数
        if not all(local_results):  # 如果有本地未命中的 # 如果有本地未命中的
            missing_indices = [i for i, res in enumerate(local_results) if not res]  # 未命中的索引 # 未命中的索引
            missing_hashes = [image_hashes[i] for i in missing_indices]  # 未命中的哈希 # 未命中的哈希

            global_exists = await asyncio.to_thread(  # 在线程池中异步执行全局存在性检查 # 异步执行全局检查
                self.mooncake_store.batch_is_exist, missing_hashes  # 调用Mooncake批量存在性检查 # 调用Mooncake批量检查
            )
            global_hit_count = sum(global_exists)  # 全局命中数 # 全局命中数

            for i, exists in zip(missing_indices, global_exists):  # 更新本地结果 # 更新本地结果
                local_results[i] = exists  # 设置全局检查结果 # 设置全局检查结果

        total = len(image_hashes)  # 总数 # 总数
        miss_count = total - local_hit_count - global_hit_count  # 未命中数 # 未命中数
        logger.info(
            f"=== Multi-Level Cache Check === "  # 多级缓存检查 # 多级缓存检查
            f"Total: {total} | "  # 总数 # 总数
            f"Local Hits: {local_hit_count} | "  # 本地命中数 # 本地命中数
            f"Global Hits: {global_hit_count} | "  # 全局命中数 # 全局命中数
            f"Misses (GPU Work): {miss_count}"  # 未命中数（需要GPU计算） # 未命中数
        )
        return local_results  # 返回存在性结果列表 # 返回存在性结果列表
