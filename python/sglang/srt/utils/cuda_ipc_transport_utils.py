# CUDA IPC传输工具模块
# 实现基于CUDA IPC句柄和共享内存的跨进程张量数据传输
# 包括多模态特征内存池管理、同步缓冲区、张量代理以及池级IPC句柄缓存

import fcntl  # 文件锁模块
import logging  # 日志记录模块
import threading  # 线程模块
import time  # 时间模块
from multiprocessing import shared_memory  # 共享内存模块
from typing import Any, Tuple  # 类型注解

import numpy as np  # NumPy数值计算库
import torch  # PyTorch深度学习框架

from sglang.srt.environ import envs  # SGLang环境变量配置
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

MM_FEATURE_CACHE_SIZE = envs.SGLANG_MM_FEATURE_CACHE_MB.get() * 1024 * 1024  # 多模态特征缓存大小（字节）

MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL = (  # 多模态项内存池回收间隔（秒）
    envs.SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC.get()
)

SHM_LOCK_FILE = "/tmp/shm_wr_lock.lock"  # 共享内存写锁文件路径


# Cache for pool-level IPC handles on the consumer side.
# Key: the pool CUDA IPC handle tuple. Value: opened UntypedStorage.
# 消费端池级IPC句柄缓存。
# 键：池CUDA IPC句柄元组。值：已打开的UntypedStorage。
_pool_storage_cache: dict = {}  # 池级IPC句柄缓存字典
_pool_cache_lock = threading.Lock()  # 池缓存访问锁


def _normalize_pool_cache_key(pool_handle, pool_device_index: int) -> tuple[Any, ...]:  # 规范化池缓存键，将句柄转为元组
    normalized_handle = (  # 规范化IPC句柄
        pool_handle if isinstance(pool_handle, tuple) else tuple(pool_handle)  # 如果不是元组则转换
    )
    return (pool_device_index, normalized_handle)  # 返回(设备索引, 规范化句柄)元组


def _open_pooled_storage_uncached(pool_handle):  # 无缓存地打开池级CUDA IPC共享存储
    return torch.UntypedStorage._new_shared_cuda(*pool_handle)  # 使用IPC句柄创建共享存储


def _pool_handle_cache_get_or_open(cache_key, pool_handle):  # 从缓存获取或打开池级IPC共享存储（双重检查锁）
    storage = _pool_storage_cache.get(cache_key)  # 第一次检查缓存（无锁）
    if storage is None:  # 缓存未命中
        with _pool_cache_lock:  # 获取缓存锁
            storage = _pool_storage_cache.get(cache_key)  # 第二次检查缓存（有锁）
            if storage is None:  # 确认缓存未命中
                storage = _open_pooled_storage_uncached(pool_handle)  # 打开新的共享存储
                _pool_storage_cache[cache_key] = storage  # 写入缓存
    return storage  # 返回共享存储


def _pool_handle_cache_set(cache_key, storage):  # 设置池级IPC句柄缓存
    with _pool_cache_lock:  # 获取缓存锁
        _pool_storage_cache[cache_key] = storage  # 写入缓存


def _pool_handle_cache_invalidate(cache_key):  # 使指定键的池级IPC句柄缓存失效
    with _pool_cache_lock:  # 获取缓存锁
        _pool_storage_cache.pop(cache_key, None)  # 移除缓存项


def _pool_handle_cache_clear():  # 清空所有池级IPC句柄缓存
    with _pool_cache_lock:  # 获取缓存锁
        _pool_storage_cache.clear()  # 清空缓存字典


class ShmSyncBuffer:  # 共享内存同步缓冲区，用于跨进程同步标志
    def __init__(self, byte_size: int = 4):  # 初始化共享内存同步缓冲区，默认4字节
        self.buffer = shared_memory.SharedMemory(create=True, size=byte_size)  # 创建共享内存
        self.buffer_wrapper = np.ndarray(1, dtype=np.float32, buffer=self.buffer.buf)  # 包装为NumPy数组
        self.buffer_wrapper *= 0  # 初始化为零
        self.meta_data = {  # 同步缓冲区的元数据
            "handle": self.buffer.name,  # 共享内存名称
            "shape": self.buffer_wrapper.shape,  # 数组形状
            "dtype": str(self.buffer_wrapper.dtype),  # 数据类型字符串
        }

    def __del__(self):  # 析构时关闭并释放共享内存
        if isinstance(self.buffer, shared_memory.SharedMemory):  # 确认是共享内存对象
            self.buffer.close()  # 关闭共享内存映射
            self.buffer.unlink()  # 释放共享内存资源


class MmItemMemoryChunk:  # 多模态项内存块，表示内存池中一个区域及其同步标志
    def __init__(self, area: Tuple, sync_buffer: ShmSyncBuffer):  # 初始化内存块
        self.area = area  # 内存区域范围(起始, 结束)
        self.sync_flag = sync_buffer  # 同步标志缓冲区

    @property
    def mem_size(self):  # 内存块大小（字节）
        return self.area[1] - self.area[0]  # 结束位置减起始位置

    @property
    def start(self):  # 内存块起始位置
        return self.area[0]  # 返回起始位置

    @property
    def end(self):  # 内存块结束位置
        return self.area[1]  # 返回结束位置

    def try_to_recycle(self) -> bool:  # 尝试回收内存块，检查所有消费者是否已完成读取
        try:  # 捕获获取服务器参数的异常
            tp_num = get_global_server_args().tp_size  # 获取TP大小
        except Exception:  # 服务器参数未初始化
            logger.info(  # 输出信息日志
                "get_global_server_args has not been inited , skip this turn 's recycle"  # 服务器参数未初始化，跳过本次回收
            )
            return False  # 回收失败

        val = float(self.sync_flag.buffer_wrapper.item())  # 读取同步标志值
        logger.debug(f"[try_to_recycle] area={self.area}, flag={val}, tp_size={tp_num}")  # 输出调试日志

        if val == float(tp_num):  # 如果所有TP rank都已确认读取
            self.sync_flag.buffer_wrapper *= 0.0  # 重置同步标志为零
            return True  # 回收成功

        return False  # 尚未全部确认，回收失败


class MmItemMemoryPool:  # 多模态项内存池，管理GPU显存的分配、回收和合并
    def __init__(self, memory_size, recycle_interval):  # 初始化内存池
        self.memory_pool = torch.empty(  # 在CUDA上分配连续内存池
            memory_size, dtype=torch.int8, device="cuda"  # int8类型，CUDA设备
        ).contiguous()  # 确保内存连续
        storage = self.memory_pool.untyped_storage()  # 获取非类型化存储
        self._pool_ipc_handle = storage._share_cuda_()  # 获取CUDA IPC句柄
        self._pool_device_index = self.memory_pool.device.index  # 获取设备索引

        self.sync_flag_list = []  # 同步标志列表（用于复用）

        init_chunk = MmItemMemoryChunk((0, memory_size), self.pop_sync_buffer())  # 创建初始完整内存块
        self.available_chunks = [init_chunk]  # 可用内存块列表
        self.occupied_chunks = []  # 已占用内存块列表

        self._lock = threading.Lock()  # 线程锁
        self._pool_full_warned = False  # 内存池满警告标志

        self._recycle_interval = recycle_interval  # 回收间隔（秒）
        self._stop_recycler = False  # 停止回收线程标志
        self._recycle_thread = threading.Thread(  # 创建回收线程
            target=self._recycle_loop, name="MmItemMemoryPoolRecycler", daemon=True  # 守护线程
        )
        self._recycle_thread.start()  # 启动回收线程

        logger.debug(  # 输出调试日志
            f"[MmItemMemoryPool] init: memory_size={memory_size}, "  # 内存大小
            f"recycle_interval={recycle_interval}s"  # 回收间隔
        )

    def shutdown(self):  # 关闭内存池，停止回收线程
        self._stop_recycler = True  # 设置停止标志
        if self._recycle_thread.is_alive():  # 如果回收线程仍在运行
            self._recycle_thread.join(timeout=1.0)  # 等待线程结束（最多1秒）

    def _recycle_loop(self):  # 回收线程的主循环
        while not self._stop_recycler:  # 未收到停止信号时循环
            try:  # 捕获回收过程中的异常
                with self._lock:  # 获取线程锁
                    self.recycle_chunks()  # 执行内存块回收
                    self.merge_chunks()  # 执行内存块合并
            except Exception as e:  # 捕获异常
                logger.warning(  # 输出警告日志
                    f"[MmItemMemoryPool] recycle loop error: {e}", exc_info=True  # 异常信息
                )

            time.sleep(self._recycle_interval)  # 按回收间隔休眠

    def clear_sync_flag_list(self):  # 清空同步标志列表
        # call each chunk's __del__
        # 调用每个块的__del__
        self.sync_flag_list.clear()  # 清空列表

    def pop_sync_buffer(self):  # 弹出或创建同步缓冲区
        if len(self.sync_flag_list) == 0:  # 如果没有可复用的同步缓冲区
            try:  # 捕获创建异常
                new_sync_buffer = ShmSyncBuffer()  # 创建新的同步缓冲区
                return new_sync_buffer  # 返回新缓冲区
            except:  # 创建失败
                logger.info("allocate shm buffer failed")  # 输出信息日志
                raise RuntimeError  # 抛出运行时错误
        else:  # 有可复用的同步缓冲区
            return self.sync_flag_list.pop()  # 弹出并返回

    def push_sync_buffer(self, sync_buffer):  # 将同步缓冲区放回复用列表
        self.sync_flag_list.append(sync_buffer)  # 添加到列表末尾

    def get_available_chunk(self, src_tensor: torch.Tensor) -> MmItemMemoryChunk:  # 获取可用内存块（最佳适配）
        # find currently available_chunks contain a available chunk or not
        # if not, return None
        # 查找当前可用内存块中是否有足够大的块，如果没有返回None
        src_tensor_size = src_tensor.numel() * src_tensor.element_size()  # 计算源张量所需字节数
        min_size = self.memory_pool.numel() * self.memory_pool.element_size() + 1  # 初始化最小匹配大小为池大小+1
        selected_chunk = None  # 选中的内存块
        for chunk in self.available_chunks:  # 遍历所有可用内存块
            if chunk.mem_size >= src_tensor_size:  # 如果块大小足够
                if chunk.mem_size < min_size:  # 如果是更小的匹配（最佳适配）
                    min_size = chunk.mem_size  # 更新最小匹配大小
                    selected_chunk = chunk  # 更新选中的块

        if selected_chunk:  # 如果找到了合适的内存块
            occupied_chunk_area = (  # 计算占用区域
                selected_chunk.start,  # 起始位置
                selected_chunk.start + src_tensor_size,  # 起始位置+张量大小
            )
            occupied_chunk_sync_flag = selected_chunk.sync_flag  # 获取同步标志
            new_occupied_chunk = MmItemMemoryChunk(  # 创建新的已占用内存块
                occupied_chunk_area, occupied_chunk_sync_flag  # 区域和同步标志
            )

            self.occupied_chunks.append(new_occupied_chunk)  # 添加到已占用列表
            self.available_chunks.remove(selected_chunk)  # 从可用列表移除

            available_split_chunk_area = (new_occupied_chunk.end, selected_chunk.end)  # 计算剩余可用区域
            # add a new chunk
            # 添加新的可用块
            if available_split_chunk_area[0] != available_split_chunk_area[1]:  # 如果剩余区域非空
                split_available_chunk = MmItemMemoryChunk(  # 创建剩余可用内存块
                    available_split_chunk_area, self.pop_sync_buffer()  # 区域和新同步标志
                )
                self.available_chunks.append(split_available_chunk)  # 添加到可用列表

            return new_occupied_chunk  # 返回已占用的内存块

        return None  # 无可用内存块

    def return_a_slice_tensor_with_flag(self, src_tensor: torch.Tensor):  # 分配内存块并返回同步标志和切片张量
        with self._lock:  # 获取线程锁
            available_chunk = self.get_available_chunk(src_tensor)  # 获取可用内存块
            if available_chunk is not None:  # 如果找到可用块
                return (  # 返回元组
                    available_chunk.sync_flag.meta_data,  # 同步标志元数据
                    self.memory_pool[available_chunk.start : available_chunk.end],  # 内存池切片张量
                    available_chunk.start,  # 起始偏移
                )
        self._warn_pool_full_once(src_tensor)  # 输出内存池满警告（仅一次）
        return None, None, None  # 分配失败，返回三个None

    def _warn_pool_full_once(self, src_tensor: torch.Tensor):  # 仅输出一次内存池满警告
        if self._pool_full_warned:  # 如果已输出过警告
            return  # 直接返回
        self._pool_full_warned = True  # 标记已输出警告
        pool_mb = (  # 计算内存池大小（MB）
            self.memory_pool.numel() * self.memory_pool.element_size() / (1024 * 1024)  # 字节数转MB
        )
        need_mb = src_tensor.numel() * src_tensor.element_size() / (1024 * 1024)  # 计算所需大小（MB）
        logger.warning(  # 输出警告日志
            "MmItemMemoryPool has no free chunk large enough for a %.2f MiB tensor "  # 内存池无足够大的空闲块
            "(pool size: %.2f MiB); falling back to non-IPC transport. "  # 内存池大小，回退到非IPC传输
            "Consider increasing SGLANG_MM_FEATURE_CACHE_MB.",  # 建议增加缓存大小
            need_mb,  # 所需大小
            pool_mb,  # 池大小
        )

    def recycle_chunks(self):  # 回收已确认读取的内存块

        new_occupied_chunks = []  # 新的已占用列表
        for chunk in self.occupied_chunks:  # 遍历所有已占用内存块
            if chunk.try_to_recycle():  # 尝试回收该块
                self.available_chunks.append(chunk)  # 回收成功，添加到可用列表
            else:  # 回收失败
                new_occupied_chunks.append(chunk)  # 保留在已占用列表
        self.occupied_chunks = new_occupied_chunks  # 更新已占用列表

    def merge_chunks(self):  # 合并相邻的可用内存块
        # merge_all_available_chunks
        # 合并所有可用内存块
        merged_chunks = []  # 合并后的内存块列表
        for chunk in sorted(self.available_chunks, key=lambda x: x.start):  # 按起始位置排序遍历
            if len(merged_chunks) == 0:  # 如果结果列表为空
                merged_chunks.append(chunk)  # 直接添加
            else:  # 结果列表非空
                if chunk.start == merged_chunks[-1].end:  # 如果与前一个块相邻
                    to_merge_chunk = merged_chunks.pop()  # 弹出前一个块
                    to_merge_chunk_sync = to_merge_chunk.sync_flag  # 保留前一个块的同步标志
                    merged_chunk_area = (to_merge_chunk.start, chunk.end)  # 合并区域
                    merged_chunks.append(  # 添加合并后的块
                        MmItemMemoryChunk(merged_chunk_area, to_merge_chunk_sync)  # 创建合并块
                    )
                    self.push_sync_buffer(chunk.sync_flag)  # 回收被合并块的同步标志
                else:  # 不相邻
                    merged_chunks.append(chunk)  # 直接添加

        self.available_chunks = merged_chunks  # 更新可用内存块列表


class CudaIpcTensorTransportProxy:  # CUDA IPC张量传输代理，用于跨进程共享GPU张量
    """
    A torch.tensor's proxy used to do inter-process data-sharing
    including:

    torch.tensor(on gpu)'s cuda-ipc-hande infos
    a shm sync buffer's meta data which is used to sync between different process
    """
    # torch.tensor的代理，用于跨进程数据共享，包括：
    # GPU张量的CUDA IPC句柄信息
    # 用于跨进程同步的共享内存同步缓冲区元数据

    def __init__(  # 初始化CUDA IPC张量传输代理
        self,
        data: torch.Tensor,  # 数据张量（GPU上）
        info_data: torch.Tensor,  # 重构目标形状和类型的参考张量
        sync_buffer_meta,  # 同步缓冲区元数据
        pool_ipc_handle=None,  # 池级IPC句柄（可选）
        pool_byte_offset: int = 0,  # 池内字节偏移
        pool_device_index: int = 0,  # 池所在设备索引
    ):

        if (not isinstance(data, torch.Tensor)) or (  # 验证data类型
            not isinstance(info_data, torch.Tensor)  # 验证info_data类型
        ):
            raise TypeError(  # 抛出类型错误
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"  # 错误信息
            )

        if pool_ipc_handle is not None:  # 如果提供了池级IPC句柄（使用池模式）
            self.proxy_state = {  # 构建代理状态（池模式）
                "ipc_extra": {  # IPC额外信息
                    "pool_handle": pool_ipc_handle,  # 池IPC句柄
                    "pool_byte_offset": pool_byte_offset,  # 池内字节偏移
                    "pool_device_index": pool_device_index,  # 池设备索引
                    "shape": data.shape,  # 数据张量形状
                    "dtype": data.dtype,  # 数据张量类型
                    "stride": data.stride,  # 数据张量步幅
                    "storage_offset": 0,  # 存储偏移（池模式为0）
                    "nbytes": data.numel() * data.element_size(),  # 字节大小
                    "recons_shape": info_data.shape,  # 重构目标形状
                    "recons_dtype": info_data.dtype,  # 重构目标类型
                },
                "tensor_data": None,  # 无需张量数据（使用IPC）
            }
        else:  # 非池模式，使用独立IPC句柄
            self.proxy_state = self.get_proxy_state(data, info_data)  # 获取代理状态
        self.reconstruct_tensor = None  # 重构后的张量缓存
        self.sync_data_meta = sync_buffer_meta  # 同步缓冲区元数据
        self.sync_buffer = None  # 共享内存同步缓冲区实例（延迟打开）

    @property
    def get_sync_flag(self):  # 获取同步标志NumPy数组（延迟打开共享内存）
        if not self.sync_buffer:  # 如果共享内存未打开
            shm_name = self.sync_data_meta["handle"]  # 获取共享内存名称
            self.sync_buffer = shared_memory.SharedMemory(name=shm_name)  # 打开已有共享内存

        shape = self.sync_data_meta["shape"]  # 获取数组形状
        dtype = self.sync_data_meta["dtype"]  # 获取数据类型
        return np.ndarray(shape, dtype=dtype, buffer=self.sync_buffer.buf)  # 返回NumPy数组视图

    def close_shm(self):  # 关闭共享内存连接
        self.sync_buffer.close()  # 关闭共享内存映射
        self.sync_buffer = None  # 重置为None

    def get_proxy_state(self, data, info_data):  # 获取张量的代理状态（独立IPC句柄模式）
        # acquire all serialize metadata from _metadata
        # 从_metadata获取所有序列化元数据
        state = {}  # 初始化状态字典

        try:  # 尝试获取CUDA IPC句柄
            storage = data.untyped_storage()  # 获取非类型化存储
            handle = storage._share_cuda_()  # 获取CUDA IPC句柄

            state["ipc_extra"] = {  # IPC额外信息
                "handle": handle,  # CUDA IPC句柄
                "shape": data.shape,  # 张量形状
                "dtype": data.dtype,  # 数据类型
                "stride": data.stride(),  # 步幅
                "device_index": data.device.index,  # 设备索引
                "storage_offset": data.storage_offset(),  # 存储偏移
                "recons_shape": info_data.shape,  # 重构目标形状
                "recons_dtype": info_data.dtype,  # 重构目标类型
            }
            state["tensor_data"] = None  # 无需张量数据
        except Exception as e:  # 获取CUDA IPC句柄失败
            # Failed to get CUDA IPC handle (possibly tp). Falling back to default transport.
            # 获取CUDA IPC句柄失败（可能是TP原因），回退到默认传输。
            state["ipc_extra"] = None  # 无IPC额外信息
            state["tensor_data"] = data  # 直接存储张量数据

        return state  # 返回代理状态

    def _reconstruct_from_ipc_extra(  # 从IPC额外信息重构切片张量（池模式）
        self, ipc_extra, *, use_cache: bool, rebuild_device_idx: int  # IPC信息、是否使用缓存、目标设备索引
    ):
        shape = ipc_extra["shape"]  # 获取张量形状
        dtype = ipc_extra["dtype"]  # 获取数据类型
        stride = ipc_extra["stride"]  # 获取步幅
        # Redirect handle[0] to the consumer's device so _new_shared_cuda's
        # CUDAGuard stays there; peer access handles the cross-GPU open.
        # 将handle[0]重定向到消费者设备，使_new_shared_cuda的CUDAGuard
        # 保持在目标设备上；对等访问处理跨GPU打开。
        pool_handle = ipc_extra["pool_handle"]  # 获取池IPC句柄
        redirected_handle = (rebuild_device_idx,) + tuple(pool_handle)[1:]  # 重定向句柄的设备索引
        target_device = torch.device(f"cuda:{rebuild_device_idx}")  # 目标CUDA设备
        cache_key = _normalize_pool_cache_key(pool_handle, rebuild_device_idx)  # 规范化缓存键

        with torch.cuda.device(target_device):  # 切换到目标设备上下文
            if use_cache:  # 使用缓存模式
                storage = _pool_handle_cache_get_or_open(cache_key, redirected_handle)  # 从缓存获取或打开存储
                storage_to_cache = None  # 无需更新缓存
            else:  # 不使用缓存
                storage = _open_pooled_storage_uncached(redirected_handle)  # 直接打开共享存储
                storage_to_cache = storage  # 保存以更新缓存
            slice_storage = storage[  # 从存储中切片
                ipc_extra["pool_byte_offset"] : ipc_extra["pool_byte_offset"]  # 起始偏移
                + ipc_extra["nbytes"]  # 结束偏移
            ]
            slice_tensor = torch.empty(0, dtype=dtype, device=target_device).set_(  # 创建切片张量视图
                slice_storage,  # 切片存储
                storage_offset=ipc_extra["storage_offset"],  # 存储偏移
                size=shape,  # 张量大小
                stride=stride,  # 步幅
            )

        return slice_tensor, target_device, cache_key, storage_to_cache  # 返回切片张量、目标设备、缓存键和待缓存存储

    def _copy_slice_tensor_to_target(  # 将切片张量复制到目标形状并更新同步标志
        self,
        slice_tensor: torch.Tensor,  # 源切片张量
        rebuild_device: torch.device,  # 目标设备
        recons_shape,  # 重构目标形状
        recons_dtype,  # 重构目标类型
    ):
        with torch.cuda.device(rebuild_device):  # 切换到目标设备上下文
            reconstructed_tensor = torch.empty(  # 分配目标形状的连续张量
                recons_shape, dtype=recons_dtype, device=rebuild_device  # 形状、类型和设备
            ).contiguous()  # 确保内存连续
            reconstructed_tensor.view(torch.int8).view(-1).copy_(slice_tensor)  # 按字节复制切片数据

            open(SHM_LOCK_FILE, "a").close()  # 确保锁文件存在
            # write the shm_sync_buffer with a file lock
            # 使用文件锁写入共享内存同步缓冲区
            with open(SHM_LOCK_FILE, "w+") as f:  # 打开锁文件
                fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
                sync_flag = self.get_sync_flag  # 获取同步标志数组
                sync_flag += 1  # 递增同步计数
                fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁

            self.close_shm()  # 关闭共享内存连接

        return reconstructed_tensor  # 返回重构后的张量

    def reconstruct_on_target_device(self, rebuild_device_idx):  # 在目标设备上重构张量
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")  # 创建目标设备对象
        if (  # 检查是否已有缓存的重构张量
            isinstance(self.reconstruct_tensor, torch.Tensor)  # 是张量类型
            and self.reconstruct_tensor.device == rebuild_device  # 且设备匹配
        ):
            return self.reconstruct_tensor  # 返回缓存的重构张量

        if self.proxy_state["ipc_extra"]:  # 如果有IPC额外信息（IPC传输模式）
            ipc_extra = self.proxy_state["ipc_extra"]  # 获取IPC额外信息
            recons_shape = ipc_extra["recons_shape"]  # 获取重构目标形状
            recons_dtype = ipc_extra["recons_dtype"]  # 获取重构目标类型

            if "pool_handle" in ipc_extra:  # 池模式
                try:  # 尝试使用缓存重构
                    (
                        slice_tensor,  # 切片张量
                        _target_device,  # 目标设备
                        cache_key,  # 缓存键
                        storage_to_cache,  # 待缓存存储
                    ) = self._reconstruct_from_ipc_extra(  # 从IPC信息重构
                        ipc_extra,  # IPC额外信息
                        use_cache=True,  # 使用缓存
                        rebuild_device_idx=rebuild_device_idx,  # 目标设备索引
                    )
                except Exception as e:  # 缓存重构失败
                    cache_key = _normalize_pool_cache_key(  # 规范化缓存键
                        ipc_extra["pool_handle"], rebuild_device_idx  # 池句柄和设备索引
                    )
                    logger.info(  # 输出信息日志
                        "Failed to deserialize from cached pooled CUDA IPC handle (%s). "  # 从缓存的池IPC句柄反序列化失败
                        "Invalidating cache entry and retrying uncached.",  # 使缓存项失效并重试无缓存方式
                        e,  # 异常信息
                    )
                    _pool_handle_cache_invalidate(cache_key)  # 使缓存项失效
                    (
                        slice_tensor,  # 切片张量
                        _target_device,  # 目标设备
                        _cache_key,  # 缓存键
                        storage_to_cache,  # 待缓存存储
                    ) = self._reconstruct_from_ipc_extra(  # 无缓存重构
                        ipc_extra,  # IPC额外信息
                        use_cache=False,  # 不使用缓存
                        rebuild_device_idx=rebuild_device_idx,  # 目标设备索引
                    )
                    if storage_to_cache is not None:  # 如果有待缓存存储
                        _pool_handle_cache_set(cache_key, storage_to_cache)  # 更新缓存
            else:  # 非池模式（独立IPC句柄）
                # Non-pooled path: redirect handle[0] the same way as the pooled path.
                # 非池路径：与池路径相同方式重定向handle[0]。
                try:  # 尝试从独立IPC句柄重构
                    original_handle = ipc_extra["handle"]  # 获取原始IPC句柄
                    redirected_handle = (rebuild_device_idx,) + tuple(original_handle)[  # 重定向设备索引
                        1:
                    ]
                    target_device = torch.device(f"cuda:{rebuild_device_idx}")  # 目标设备
                    with torch.cuda.device(target_device):  # 切换到目标设备上下文
                        storage = torch.UntypedStorage._new_shared_cuda(  # 从IPC句柄打开共享存储
                            *redirected_handle  # 传入重定向后的句柄
                        )
                        slice_tensor = torch.empty(  # 创建空张量
                            0, dtype=ipc_extra["dtype"], device=target_device  # 使用原始类型和目标设备
                        ).set_(  # 设置为共享存储的视图
                            storage,  # 共享存储
                            storage_offset=ipc_extra["storage_offset"],  # 存储偏移
                            size=ipc_extra["shape"],  # 张量形状
                            stride=ipc_extra["stride"],  # 步幅
                        )
                except Exception as e:  # 从IPC句柄重构失败
                    logger.info("Failed to deserialize from CUDA IPC handle (%s).", e)  # 输出日志
                    raise  # 重新抛出异常

            reconstructed_tensor = self._copy_slice_tensor_to_target(  # 复制切片到目标并更新同步标志
                slice_tensor, rebuild_device, recons_shape, recons_dtype  # 切片张量、目标设备、目标形状和类型
            )
        elif isinstance(self.proxy_state["tensor_data"], torch.Tensor):  # 回退模式：直接传输张量数据
            reconstructed_tensor = self.proxy_state["tensor_data"].to(  # 将张量传到目标设备
                rebuild_device, non_blocking=True  # 非阻塞传输
            )
        else:  # 无效的代理状态
            raise TypeError("invalid proxy_state")  # 抛出类型错误

        self.reconstruct_tensor = reconstructed_tensor  # 缓存重构后的张量
        return self.reconstruct_tensor  # 返回重构后的张量
