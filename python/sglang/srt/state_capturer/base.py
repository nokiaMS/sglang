# 状态捕获器基础模块
# 提供设备端缓存、主机端缓存以及Topk捕获输出的基类实现
# 用于在模型前向推理过程中捕获和传输topk索引数据

import dataclasses  # 数据类装饰器
import logging  # 日志记录模块
from typing import Optional  # 可选类型注解

import torch  # PyTorch深度学习框架

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # 请求到token的映射池
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批处理信息

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_GB = 1024 * 1024 * 1024  # 吉字节常量
_MB = 1024 * 1024  # 兆字节常量


def get_tensor_size_bytes(t: torch.Tensor) -> int:  # 获取张量的字节大小
    return t.numel() * t.element_size()  # 元素数量乘以每个元素的字节数


class BaseDeviceCache:  # 设备端缓存基类，在GPU上存储topk索引
    def __init__(  # 初始化设备端缓存
        self,
        max_batch_size: int,  # 最大批处理大小
        num_layers: int,  # 层数
        topk_size: int,  # topk的大小
        device: str,  # 设备类型（如cuda:0）
        name: str,  # 缓存名称
    ):
        self.buffer = torch.zeros(  # 分配GPU上的零初始化缓冲区
            (max_batch_size, num_layers, topk_size),  # 三维张量形状
            dtype=torch.int32,  # 32位整数类型
            device=device,  # 存储设备
        )
        self.num_layers = num_layers  # 保存层数
        self.topk_size = topk_size  # 保存topk大小
        self.name = name  # 保存缓存名称
        self._log_allocation()  # 记录分配信息

    def capture(self, layer_id: int, topk_indices: torch.Tensor):  # 捕获指定层的topk索引
        batch = topk_indices.shape[0]  # 获取当前批大小
        self.buffer[:batch, layer_id, :] = topk_indices  # 将topk索引写入缓冲区对应层位置

    def get_buffer_size_bytes(self):  # 获取缓冲区的字节大小
        return get_tensor_size_bytes(self.buffer)  # 使用辅助函数计算

    def _log_allocation(self):  # 记录缓冲区分配信息
        size_mb = self.get_buffer_size_bytes() / _MB  # 转换为兆字节
        logger.info(  # 输出日志信息
            f"DeviceCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "  # 形状信息
            f"size={size_mb:.2f} MB"  # 大小信息
        )


class BaseHostCache:  # 主机端缓存基类，在CPU上存储topk索引（持久化）
    def __init__(self, num_tokens: int, num_layers: int, topk_size: int, name: str):  # 初始化主机端缓存
        self.buffer = torch.zeros(  # 分配CPU上的零初始化缓冲区
            (num_tokens, num_layers, topk_size),  # 三维张量形状
            dtype=torch.int32,  # 32位整数类型
            device="cpu",  # 存储在CPU上
            pin_memory=True,  # 锁页内存，加速GPU传输
        )
        self.num_tokens = num_tokens  # 保存token数量
        self.num_layers = num_layers  # 保存层数
        self.topk_size = topk_size  # 保存topk大小
        self.name = name  # 保存缓存名称
        self._log_allocation()  # 记录分配信息

    def get_buffer_size_bytes(self):  # 获取缓冲区的字节大小
        return get_tensor_size_bytes(self.buffer)  # 使用辅助函数计算

    def _log_allocation(self):  # 记录缓冲区分配信息
        size_gb = self.get_buffer_size_bytes() / _GB  # 转换为吉字节
        logger.info(  # 输出日志信息
            f"HostCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "  # 形状信息
            f"size={size_gb:.2f} GB"  # 大小信息
        )


@dataclasses.dataclass  # 数据类装饰器
class TopkCaptureOutput:  # Topk捕获输出数据类，持有前向过程中捕获的GPU张量用于重叠调度
    """Holds GPU tensors captured during forward for overlap scheduling.
    Call copy_to_cpu() inside forward stream (before copy_done.record()),
    then finalize() after copy_done.synchronize().
    """
    # 在前向流中调用copy_to_cpu()（在copy_done.record()之前），
    # 然后在copy_done.synchronize()之后调用finalize()。

    out_cache_loc: torch.Tensor  # 输出缓存位置索引
    topk: torch.Tensor  # topk索引张量
    host_cache: BaseHostCache  # 主机端缓存引用

    def copy_to_cpu(self):  # 将GPU张量异步复制到CPU
        self.out_cache_loc = self.out_cache_loc.to("cpu", non_blocking=True)  # 异步传输缓存位置到CPU
        self.topk = self.topk.to("cpu", non_blocking=True)  # 异步传输topk索引到CPU

    def finalize(self):  # 在CPU同步完成后将topk数据写入主机缓存
        self.host_cache.buffer[self.out_cache_loc] = self.topk  # 根据缓存位置索引写入topk数据


class BaseTopkCapturer:  # Topk捕获器基类，协调设备缓存和主机缓存的捕获流程
    def __init__(  # 初始化Topk捕获器
        self,
        num_tokens: int,  # token总数
        max_batch_size: int,  # 最大批处理大小
        num_layers: int,  # 层数
        topk_size: int,  # topk大小
        device: str,  # 设备类型
        name: str,  # 捕获器名称
        device_topk_size: Optional[int] = None,  # 设备端topk大小（可与主机端不同）
    ):
        """device_topk_size defaults to topk_size; pass a different value when
        the device buffer needs extra columns (e.g. fused shared experts) that
        are dropped before writing to host_cache via [:topk_size] truncation.
        """
        # device_topk_size默认为topk_size；当设备缓冲区需要额外列
        # （如融合共享专家）时传入不同值，写入host_cache前通过[:topk_size]截断丢弃。
        self.num_layers = num_layers  # 保存层数
        self.topk_size = topk_size  # 保存topk大小

        self.host_cache = BaseHostCache(num_tokens, num_layers, topk_size, name=name)  # 创建主机端缓存
        self.device_cache = BaseDeviceCache(  # 创建设备端缓存
            max_batch_size,  # 最大批大小
            num_layers,  # 层数
            device_topk_size if device_topk_size is not None else topk_size,  # 使用指定的设备topk大小或默认值
            device,  # 设备类型
            name=name,  # 缓存名称
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):  # 捕获指定层的topk索引到设备缓存
        self.device_cache.capture(layer_id, topk_indices)  # 委托给设备缓存执行捕获

    def _get_local_slice(  # 获取当前前向批的设备缓存切片（GPU上驻留）
        self,
        forward_batch: ForwardBatch,  # 前向批处理信息
        can_run_graph: bool,  # 是否可以运行CUDA图
        cuda_graph_batch: Optional[int],  # CUDA图批大小
    ) -> torch.Tensor:
        """Return the device_cache slice for this forward batch, GPU-resident.

        Default assumes per-rank-local capture: each rank writes [:local_num_tokens)
        to its own device_cache. Subclasses with global-tensor capture semantics
        (e.g. shared cuda graph buffer indexed by dp_rank) should override and
        consume can_run_graph / cuda_graph_batch.
        """
        # 返回当前前向批的设备缓存切片，驻留在GPU上。
        # 默认假设每个rank本地捕获：每个rank将[:local_num_tokens)写入自己的device_cache。
        # 具有全局张量捕获语义的子类（如按dp_rank索引的共享CUDA图缓冲区）应重写此方法
        # 并使用can_run_graph / cuda_graph_batch参数。
        del can_run_graph, cuda_graph_batch  # reserved for subclass override  # 预留给子类重写
        num_tokens = forward_batch.out_cache_loc.shape[0]  # 获取本地token数量
        return self.device_cache.buffer[:num_tokens, :, : self.topk_size]  # 返回本地切片，截断到topk_size

    def get_topk(  # 根据请求池索引和序列长度获取历史topk索引
        self,
        req_pool_idx: int,  # 请求池索引
        seqlen: int,  # 序列长度
        req_to_token_pool: ReqToTokenPool,  # 请求到token的映射池
        start_len: int = 0,  # 起始长度（默认0）
    ) -> torch.Tensor:
        if start_len < 0:  # 检查起始长度非负
            raise ValueError(f"{start_len=} must be non-negative")  # 抛出异常
        start_len = min(start_len, seqlen - 1)  # 确保起始长度不超过序列末尾
        cache_pool_idx = (  # 获取缓存池索引
            req_to_token_pool.req_to_token[req_pool_idx][start_len : seqlen - 1]  # 根据请求池索引和长度范围获取token位置
            .cpu()  # 传输到CPU
            .clone()  # 克隆以避免共享内存问题
        )
        return self.host_cache.buffer[cache_pool_idx]  # 从主机缓存中获取对应的topk数据

    def on_forward_end(  # 前向结束时将设备缓存数据传输到主机缓存
        self,
        forward_batch: ForwardBatch,  # 前向批处理信息
        can_run_graph: bool,  # 是否可以运行CUDA图
        cuda_graph_batch: Optional[int],  # CUDA图批大小
        no_copy_to_cpu: bool = False,  # 是否跳过同步复制到CPU（用于重叠调度）
    ) -> Optional[TopkCaptureOutput]:
        """If no_copy_to_cpu is True, return a TopkCaptureOutput holding GPU tensors so
        the overlap thread can do non-blocking D2H + finalize itself. Otherwise sync
        D2H inline and return None (legacy non-overlap path).
        """
        # 如果no_copy_to_cpu为True，返回持有GPU张量的TopkCaptureOutput，
        # 以便重叠线程执行非阻塞D2H传输和finalize。否则同步D2H并返回None（传统非重叠路径）。
        slice_gpu = self._get_local_slice(  # 获取GPU上的本地切片
            forward_batch, can_run_graph, cuda_graph_batch
        )
        if no_copy_to_cpu:  # 如果启用重叠调度模式
            return TopkCaptureOutput(  # 返回持有GPU张量的捕获输出
                out_cache_loc=forward_batch.out_cache_loc,  # 缓存位置索引
                topk=slice_gpu,  # GPU上的topk切片
                host_cache=self.host_cache,  # 主机缓存引用
            )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()  # 同步传输缓存位置到CPU
        self.host_cache.buffer[out_cache_loc_cpu] = slice_gpu.cpu()  # 同步传输topk数据并写入主机缓存
        return None  # 传统路径返回None
