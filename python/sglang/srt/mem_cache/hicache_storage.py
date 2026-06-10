# HiCache存储后端抽象模块
# 定义分层缓存（HiCache）的存储后端接口和文件系统实现
# 提供KV缓存在GPU、CPU和外部存储之间的多级数据传输能力
# 支持批量存在性检查、读写操作，以及多池联合传输（KV/Mamba/SWA/Indexer等）

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
import os  # 导入操作系统模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from dataclasses import dataclass  # 导入数据类装饰器
from enum import Enum  # 导入枚举类型
from typing import TYPE_CHECKING, Any, List, Optional, Set  # 导入类型提示工具

import torch  # 导入PyTorch张量库

from sglang.srt.environ import envs  # 导入环境变量配置

if TYPE_CHECKING:  # 仅在类型检查时执行的代码块
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # 导入主机KV缓存类型

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# Max pages per batched storage IO call.
# 每次批量存储IO调用的最大页数。
STORAGE_BATCH_SIZE = 128  # 存储批量大小常量


@dataclass  # HiCache存储配置数据类
class HiCacheStorageConfig:
    tp_rank: int  # 张量并行排名
    tp_size: int  # 张量并行大小
    pp_rank: int  # 流水线并行排名
    pp_size: int  # 流水线并行大小
    attn_cp_rank: int  # 注意力上下文并行排名
    attn_cp_size: int  # 注意力上下文并行大小
    is_mla_model: bool  # 是否为MLA模型
    enable_storage_metrics: bool  # 是否启用存储指标收集
    is_page_first_layout: bool  # 是否使用页优先内存布局
    model_name: Optional[str]  # 模型名称
    tp_lcm_size: Optional[int] = None  # 张量并行最小公倍数大小
    should_split_heads: bool = False  # 是否需要拆分注意力头
    extra_config: Optional[dict] = None  # 额外配置字典


@dataclass  # HiCache存储额外信息数据类
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = None  # 前缀键列表
    extra_info: Optional[dict] = None  # 额外信息字典


@dataclass(frozen=True)  # 预取超时配置（不可变数据类）
class PrefetchTimeoutConfig:
    """Knobs for the linear prefetch-timeout policy used by HiCache.
    HiCache使用的线性预取超时策略的配置参数。"""

    base: float = 2.0  # seconds, fixed overhead unrelated to token count  # 秒，与token数量无关的固定开销
    per_ki_token: float = 0.1  # seconds per 1024 tokens  # 每1024个token的秒数
    max: float = 30.0  # seconds, upper bound for the linear timeout  # 秒，线性超时的上限


class PoolName(str, Enum):  # 池名称枚举
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers.
    用作PoolTransfer/PoolEntry标识符的已知池名称。"""

    KV = "kv"  # KV缓存池
    MAMBA = "mamba"  # Mamba状态池
    SWA = "swa"  # 滑动窗口注意力池
    INDEXER = "indexer"  # 索引器池
    # TODO(hzh0425): Current DeepSeek V4 pool naming is verbose; will be normalized to
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE' in the next PR.
    # TODO(hzh0425): 当前DeepSeek V4池命名过于冗长；将在下一个PR中规范化为
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE'。
    DEEPSEEK_V4_C4 = "deepseek_v4_c4"  # DeepSeek V4 C4压缩KV池
    DEEPSEEK_V4_C4_INDEXER = "deepseek_v4_c4_indexer"  # DeepSeek V4 C4索引器池
    DEEPSEEK_V4_C128 = "deepseek_v4_c128"  # DeepSeek V4 C128压缩KV池
    DEEPSEEK_V4_C4_STATE = "deepseek_v4_c4_state"  # DeepSeek V4 C4状态池
    DEEPSEEK_V4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"  # DeepSeek V4 C4索引器状态池
    DEEPSEEK_V4_C128_STATE = "deepseek_v4_c128_state"  # DeepSeek V4 C128状态池

    def __str__(self) -> str:  # 字符串表示方法
        return self.value  # 返回枚举值


class PoolHitPolicy(str, Enum):  # 池命中策略枚举
    """Hit policy for batch_exists_v2 per-pool prefix matching.
    batch_exists_v2中每个池前缀匹配的命中策略。

    ALL_PAGES      : every page in [0, kv_hit) must exist (e.g. DSA).
                     [0, kv_hit)范围内的每个页面都必须存在（例如DSA）。
    TRAILING_PAGES : only the last N pages must exist (e.g. Mamba/SWA states).
                     仅最后N个页面必须存在（例如Mamba/SWA状态）。
    """

    ALL_PAGES = "all_pages"  # 所有页面命中策略
    TRAILING_PAGES = "trailing_pages"  # 尾部页面命中策略


@dataclass  # 池传输描述符数据类
class PoolTransfer:
    """Unified per-pool transfer descriptor for batch v2 interface.
    批量v2接口的统一每池传输描述符。

    device<->host path : host_indices + device_indices
    设备<->主机路径：主机索引 + 设备索引
    host<->storage path: host_indices + keys
    主机<->存储路径：主机索引 + 键
    nodes_to_load      : evicted nodes this transfer covers
    要加载的节点：此传输覆盖的被淘汰节点
    """

    name: PoolName  # 池名称
    host_indices: Optional[torch.Tensor] = None  # 主机端索引张量
    device_indices: Optional[torch.Tensor] = None  # 设备端索引张量
    keys: Optional[List[str]] = None  # 存储键列表
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES  # 命中策略，默认为全部页面
    nodes_to_load: Optional[List[Any]] = None  # 要加载的节点列表
    indices_from_pool: Optional[PoolName] = None  # 索引来源池名称


@dataclass(frozen=True)  # 伴生池规格（不可变数据类）
class SidecarPoolSpec:
    """Pool whose transfer indices are reused from one real source pool.
    其传输索引从一个真实源池复用的伴生池。"""

    pool_name: PoolName  # 伴生池名称
    indices_from_pool: PoolName  # 索引来源池名称
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES  # 命中策略


@dataclass  # 池传输结果数据类
class PoolTransferResult:
    """Tracks how many pages were successfully processed per pool.
    跟踪每个池成功处理的页数。"""

    kv_hit_pages: int  # KV命中页数
    extra_pool_hit_pages: dict[str, int]  # 额外池命中页数字典

    @classmethod  # 类方法：创建空的传输结果
    def empty(cls) -> "PoolTransferResult":
        return cls(0, {})  # 返回零命中结果

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:  # 更新KV命中页数
        """Accumulate kv_hit_pages across batches (max = last successful batch).
        跨批次累加kv_hit_pages（取最大值 = 最后一次成功批次）。"""
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)  # 取最大值

    def update_extra_pool_hit_pages(self, results: dict[str, List[bool]]) -> None:  # 更新额外池命中页数
        """Record actual load/write success counts per extra pool.
        记录每个额外池的实际加载/写入成功计数。"""
        self.extra_pool_hit_pages.update(
            {name: sum(rs) for name, rs in results.items()}  # 统计每个池的成功数
        )


class HiCacheStorage(ABC):  # HiCache存储抽象基类
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    HiCacheStorage是一个类，提供用于存储和检索KV缓存的通用键值接口。
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    它抽象了底层存储机制，允许使用不同的实现。
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    # todo，存储后端的页面大小不必与主机内存池的页面大小相同
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):  # 注册主机内存池
        self.mem_pool_host = mem_pool_host  # 保存主机内存池引用

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):  # 注册主机内存池（v2接口）
        if not hasattr(self, "registered_pools"):  # 如果尚未初始化注册池字典
            self.registered_pools = {}  # 创建注册池字典
        self.registered_pools[host_pool_name] = host_pool  # 将主机池注册到字典中

    def batch_exists_v2(  # 批量检查缓存页是否存在于存储中（v2接口）
        self,
        keys: List[str],  # 键列表
        pool_transfers: Optional[List[PoolTransfer]] = None,  # 池传输描述符列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> PoolTransferResult:
        """Check which cache pages exist in storage, respecting per-pool hit policies.
        检查存储中存在哪些缓存页，遵守每个池的命中策略。

        Longest-prefix semantics
        最长前缀语义
        Extra-pool hit policies (``PoolTransfer.hit_policy``)
        额外池命中策略（``PoolTransfer.hit_policy``）
        ------------------------------------------------------
        Each ``PoolTransfer`` in ``pool_transfers`` describes a secondary
        cache pool (e.g. Mamba SSM states) that must be co-present with the
        KV pages.  The final ``final_pages`` is the minimum across all pools,
        so a missing auxiliary page shrinks the usable prefix.
        每个``PoolTransfer``在``pool_transfers``中描述了一个辅助缓存池
        （例如Mamba SSM状态），该池必须与KV页共同存在。最终的``final_pages``
        是所有池中的最小值，因此缺失的辅助页面会缩小可用的前缀。

        - ``"all_pages"`` (default):  every page in [0, kv_hit) must exist
          for this pool.  Used for pools that are required for every token
          in the prefix (e.g. DeepSeek DSA pool).
          ``"all_pages"``（默认）：该池在[0, kv_hit)范围内的每个页面都必须存在。
          用于前缀中每个token都需要的池（例如DeepSeek DSA池）。

        - ``"trailing_pages"``:  only the *last* ``len(transfer.keys)`` pages
          of the KV prefix need to exist.  Used for pools whose data covers
          only the tail of a prefix (e.g. Mamba/SWA Pool).
          ``"trailing_pages"``：仅KV前缀的最后``len(transfer.keys)``个页面需要存在。
          用于数据仅覆盖前缀尾部的池（例如Mamba/SWA池）。

        Returns
        返回
        -------
        PoolTransferResult
            ``kv_hit_pages`` = length of the usable KV prefix.
            ``kv_hit_pages`` = 可用KV前缀的长度。
            ``extra_pool_hit_pages`` maps each pool name to the number of pages
            that were found.
            ``extra_pool_hit_pages``将每个池名称映射到找到的页数。
        """
        raise NotImplementedError()  # 抛出未实现异常

    def batch_get_v2(  # 批量从存储读取数据到主机内存（v2接口）
        self,
        transfers: List[PoolTransfer],  # 池传输描述符列表
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,  # 额外信息
    ) -> dict[str, List[bool]]:
        """Read data from storage into host memory for each PoolTransfer.
        为每个PoolTransfer从存储读取数据到主机内存。

        Returns a dict mapping pool name to a per-entry success list.
        返回一个将池名称映射到每条记录成功列表的字典。
        """
        raise NotImplementedError()  # 抛出未实现异常

    def batch_set_v2(  # 批量将主机内存数据写入存储（v2接口）
        self,
        transfers: List[PoolTransfer],  # 池传输描述符列表
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,  # 额外信息
    ) -> dict[str, List[bool]]:
        """Write data from host memory to storage for each PoolTransfer.
        为每个PoolTransfer将主机内存数据写入存储。

        Returns a dict mapping pool name to a per-entry success list.
        返回一个将池名称映射到每条记录成功列表的字典。
        """
        raise NotImplementedError()  # 抛出未实现异常

    def batch_get_v1(  # 批量获取多个键的值（v1接口）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        检索多个键的值。
        Returns a list of booleans indicating success for each key.
        返回一个布尔列表，指示每个键的操作是否成功。
        """
        pass  # 默认无操作

    def batch_set_v1(  # 批量存储多个键值对（v1接口）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        存储多个键值对。
        Returns a list of booleans indicating success for each key.
        返回一个布尔列表，指示每个键的操作是否成功。
        """
        pass  # 默认无操作

    @abstractmethod  # 抽象方法：获取单个键的值
    def get(
        self,
        key: str,  # 键
        target_location: Optional[Any] = None,  # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        检索与给定键关联的值。
        Returns None if the key does not exist.
        如果键不存在则返回None。
        """
        pass  # 抽象方法，子类必须实现

    # TODO: Deprecate  # TODO: 弃用
    @abstractmethod  # 抽象方法：批量获取多个键的值（待弃用）
    def batch_get(
        self,
        keys: List[str],  # 键列表
        target_locations: Optional[Any] = None,  # 目标位置列表
        target_sizes: Optional[Any] = None,  # 目标大小列表
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        检索多个键的值。
        Returns a list of tensors or None for each key.
        返回每个键对应的张量列表或None。
        """
        pass  # 抽象方法，子类必须实现

    @abstractmethod  # 抽象方法：存储单个键值对
    def set(
        self,
        key: str,  # 键
        value: Optional[Any] = None,  # 值
        target_location: Optional[Any] = None,  # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> bool:
        """
        Store the value associated with the given key.
        存储与给定键关联的值。
        Returns True if the operation was successful, False otherwise.
        如果操作成功返回True，否则返回False。
        """
        pass  # 抽象方法，子类必须实现

    # TODO: Deprecate  # TODO: 弃用
    @abstractmethod  # 抽象方法：批量存储多个键值对（待弃用）
    def batch_set(
        self,
        keys: List[str],  # 键列表
        values: Optional[Any] = None,  # 值列表
        target_locations: Optional[Any] = None,  # 目标位置列表
        target_sizes: Optional[Any] = None,  # 目标大小列表
    ) -> bool:
        """
        Store multiple key-value pairs.
        存储多个键值对。
        Returns True if all operations were successful, False otherwise.
        如果所有操作成功返回True，否则返回False。
        """
        pass  # 抽象方法，子类必须实现

    @abstractmethod  # 抽象方法：检查键是否存在
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        检查键是否存在于存储中。
        Returns True if the key exists, False otherwise.
        如果键存在返回True，否则返回False。
        """
        pass  # 抽象方法，子类必须实现

    # TODO: Use a finer-grained return type (e.g., List[bool])  # TODO: 使用更细粒度的返回类型（如List[bool]）
    def batch_exists(  # 批量检查键是否连续存在
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None  # 键列表和额外信息
    ) -> int:
        """
        Check if the keys exist in the storage.
        检查键是否存在于存储中。
        return the number of consecutive existing keys from the start.
        返回从开头开始的连续存在键的数量。
        Can be overridden by subclasses for more efficient implementation.
        可以被子类覆盖以实现更高效的实现。
        """
        for i in range(len(keys)):  # 遍历所有键
            if not self.exists(keys[i]):  # 如果某个键不存在
                return i  # 返回第一个不存在的索引
        return len(keys)  # 所有键都存在，返回键总数

    def clear(self) -> None:  # 清空存储
        pass  # 默认无操作

    def get_stats(self):  # 获取存储统计信息
        return None  # 默认返回None


class HiCacheFile(HiCacheStorage):  # 基于文件系统的HiCache存储实现

    def __init__(  # 初始化文件存储后端
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"  # 存储配置和文件路径
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path  # 从环境变量或参数获取路径

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (  # 解包存储配置
            storage_config.tp_rank,  # 张量并行排名
            storage_config.tp_size,  # 张量并行大小
            storage_config.pp_rank,  # 流水线并行排名
            storage_config.pp_size,  # 流水线并行大小
            storage_config.model_name,  # 模型名称
            storage_config.is_mla_model,  # 是否为MLA模型
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""  # 处理模型名称中的斜杠
        enable_pp = pp_size > 1  # 是否启用流水线并行
        self.config_suffix = f"_{model_name}"  # 配置后缀包含模型名称
        if not is_mla_model:  # 非MLA模型添加张量并行信息
            self.config_suffix += f"_{tp_rank}_{tp_size}"  # 追加TP排名和大小
        if enable_pp:  # 启用流水线并行时添加流水线信息
            self.config_suffix += f"_{pp_size}_{pp_rank}"  # 追加PP大小和排名
        if not os.path.exists(self.file_path) and tp_rank == 0:  # 如果目录不存在且为主排名
            os.makedirs(self.file_path)  # 创建存储目录
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")  # 记录创建信息

    def _get_suffixed_key(self, key: str) -> str:  # 获取带后缀的键
        return key + self.config_suffix  # 拼接键和配置后缀

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:  # 获取组件键
        if component_name is None or component_name in ("__default__", PoolName.KV):  # 默认或KV组件
            return self._get_suffixed_key(key)  # 返回带后缀的键
        return self._get_suffixed_key(f"{key}.{component_name}")  # 返回带组件名的后缀键

    def _get_component_path(  # 获取组件文件路径
        self, key: str, component_name: Optional[str] = None  # 键和组件名称
    ) -> str:
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"  # 拼接目录和文件名
        )

    def get(  # 获取单个键的值
        self,
        key: str,  # 键
        target_location: torch.Tensor,  # 目标位置张量
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)  # 获取带后缀的键
        tensor_path = os.path.join(self.file_path, f"{key}.bin")  # 构建张量文件路径
        try:
            expected = target_location.numel() * target_location.element_size()  # 计算期望读取的字节数
            with open(tensor_path, "rb", buffering=0) as f:  # 以二进制无缓冲方式打开文件
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())  # 创建目标缓冲区视图
                if f.readinto(buf) != expected:  # 如果读取字节数不匹配
                    raise IOError(f"Short read for {key}")  # 抛出IO错误
            return target_location  # 返回目标位置张量
        except FileNotFoundError:  # 文件不存在
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")  # 记录警告
            return None  # 返回None

    def batch_get(  # 批量获取多个键的值（待弃用）
        self,
        keys: List[str],  # 键列表
        target_locations: List[torch.Tensor],  # 目标位置列表
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)  # 逐个获取值
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)  # 配对键和目标位置
            )
        ]

    def set(  # 存储单个键值对
        self,
        key: str,  # 键
        value: Optional[Any] = None,  # 值
        target_location: Optional[Any] = None,  # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> bool:
        if self.exists(key):  # 如果键已存在
            logger.debug(f"Key {key} already exists. Skipped.")  # 记录调试信息
            return True  # 跳过写入

        key = self._get_suffixed_key(key)  # 获取带后缀的键
        tensor_path = os.path.join(self.file_path, f"{key}.bin")  # 构建张量文件路径
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)  # 将张量写入文件
            return True  # 写入成功
        except Exception as e:  # 捕获异常
            logger.error(f"Failed to save tensor {key}: {e}")  # 记录错误
            return False  # 写入失败

    def batch_set(  # 批量存储多个键值对（待弃用）
        self,
        keys: List[str],  # 键列表
        values: Optional[Any] = None,  # 值列表
        target_locations: Optional[Any] = None,  # 目标位置列表
        target_sizes: Optional[Any] = None,  # 目标大小列表
    ) -> bool:
        for key, value in zip(keys, values):  # 遍历键值对
            if not self.set(key, value):  # 如果写入失败
                return False  # 返回失败
        return True  # 全部写入成功

    def exists(self, key: str) -> bool:  # 检查键是否存在
        key = self._get_suffixed_key(key)  # 获取带后缀的键
        tensor_path = os.path.join(self.file_path, f"{key}.bin")  # 构建文件路径
        return os.path.exists(tensor_path)  # 返回文件是否存在

    def _collect_existing_component_keys(  # 收集存在的组件键集合
        self,
        keys: List[str],  # 键列表
        pool_transfers: Optional[List[PoolTransfer]] = None,  # 池传输描述符列表
    ) -> Set[str]:
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}  # 收集所有目标KV文件名
        for transfer in pool_transfers or []:  # 遍历池传输描述符
            for key in keys:  # 遍历每个键
                target_files.add(f"{self._get_component_key(key, transfer.name)}.bin")  # 添加组件文件名

        existing_files = set()  # 已存在文件集合
        with os.scandir(self.file_path) as entries:  # 扫描存储目录
            for entry in entries:  # 遍历目录项
                if entry.is_file() and entry.name in target_files:  # 如果是目标文件
                    existing_files.add(entry.name)  # 添加到已存在集合
        return existing_files  # 返回已存在文件集合

    def batch_exists_v2(  # 批量检查缓存页是否存在（v2接口，支持多池命中策略）
        self,
        keys: List[str],  # 键列表
        pool_transfers: Optional[List[PoolTransfer]] = None,  # 池传输描述符列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> PoolTransferResult:
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)  # 收集已存在的组件键

        def has_component(page_idx: int, name: str) -> bool:  # 检查指定页面和组件是否存在
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files  # 在已存在集合中查找
            )

        # Longest contiguous KV prefix present in storage.
        # 存储中存在的最长连续KV前缀。
        kv_pages = next(
            (
                i
                for i in range(len(keys))  # 遍历所有键的索引
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files  # 找到第一个不存在的
            ),
            len(keys),  # 如果全部存在则返回总长度
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}  # KV命中计数
        final_pages = kv_pages  # 最终可用页数

        for transfer in pool_transfers or []:  # 遍历池传输描述符
            if final_pages == 0:  # 如果已无可用的KV前缀
                break  # 退出循环
            name = transfer.name  # 获取池名称
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:  # 全部页面命中策略
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)), kv_pages  # 找到第一个缺失的页面
                )
            else:  # trailing_pages  # 尾部页面命中策略
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)  # 计算需要检查的尾部页数
                boundary = 0  # 初始化边界
                for prefix_len in range(kv_pages, 0, -1):  # 从长到短搜索
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)  # 检查尾部页面
                    ):
                        boundary = prefix_len  # 找到匹配的前缀长度
                        break  # 退出搜索
            if boundary:  # 如果有命中
                hit_count[name] = boundary  # 记录命中数
            final_pages = min(final_pages, boundary)  # 取最小值确定最终可用页数

        return PoolTransferResult(final_pages, hit_count)  # 返回传输结果

    def _log_key(self, pool_name: str, key: str) -> str:  # 生成日志键
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"  # KV池直接返回键，其他池附加池名

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:  # 从存储读取一个页面到主机池
        """Read one page from storage into host_pool at page_offset.
        从存储读取一个页面到host_pool的page_offset位置。"""
        storage_key = self._log_key(pool_name, key)  # 生成存储键
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())  # 从存储获取数据页
        if data_page is None:  # 如果获取失败
            return False  # 返回失败
        host_pool.set_from_flat_data_page(page_offset, data_page)  # 将数据写入主机池
        return True  # 返回成功

    def _write_page(  # 将一个页面从主机池写入存储
        self, pool_name: str, key: str, host_pool, page_offset: int  # 池名、键、主机池、页偏移
    ) -> bool:
        """Write one page from host_pool at page_offset to storage as raw bytes.
        将host_pool中page_offset位置的一个页面作为原始字节写入存储。"""
        storage_key = self._log_key(pool_name, key)  # 生成存储键
        data_page = host_pool.get_data_page(page_offset, flat=True)  # 从主机池获取扁平数据页
        return self.set(storage_key, data_page)  # 写入存储并返回结果

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):  # 批量IO操作（v2接口核心实现）
        results: dict[str, List[bool]] = {}  # 结果字典
        for transfer in transfers:  # 遍历传输描述符
            host_pool = self.registered_pools[transfer.name]  # 获取对应的主机池
            keys = transfer.keys or []  # 获取键列表
            page_size = getattr(host_pool, "page_size", 1) or 1  # 获取页面大小
            expected = len(keys) * page_size  # 计算期望的索引数量
            host_indices = transfer.host_indices  # 获取主机索引

            if host_indices is None or host_indices.numel() != expected:  # 索引数量不匹配
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,  # 操作函数名
                    transfer.name,  # 池名称
                    expected,  # 期望数量
                    host_indices.numel() if host_indices is not None else 0,  # 实际数量
                )
                results[transfer.name] = [False] * len(keys)  # 标记全部失败
                continue  # 跳过此传输

            results[transfer.name] = [  # 逐页执行IO操作
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())  # 对每个键执行操作
                for i, key in enumerate(keys)  # 遍历键
            ]
        return results  # 返回结果字典

    def batch_get_v2(  # 批量从存储读取数据（v2接口）
        self,
        transfers: List[PoolTransfer],  # 池传输描述符列表
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,  # 额外信息
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)  # 使用读取页面函数执行批量IO

    def batch_set_v2(  # 批量将数据写入存储（v2接口）
        self,
        transfers: List[PoolTransfer],  # 池传输描述符列表
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,  # 额外信息
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)  # 使用写入页面函数执行批量IO

    def clear(self) -> bool:  # 清空文件存储
        try:
            for filename in os.listdir(self.file_path):  # 遍历存储目录
                file_path = os.path.join(self.file_path, filename)  # 构建文件路径
                if os.path.isfile(file_path):  # 如果是文件
                    os.remove(file_path)  # 删除文件
            logger.info("Cleared all entries in HiCacheFile storage.")  # 记录清空信息
            return True  # 返回成功
        except Exception as e:  # 捕获异常
            logger.error(f"Failed to clear HiCacheFile storage: {e}")  # 记录错误
            return False  # 返回失败
