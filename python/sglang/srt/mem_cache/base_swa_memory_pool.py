# 滑动窗口注意力（SWA）KV池基类
# 定义了SWA类型KV池的抽象接口，用于处理滑动窗口状态与完整KV状态的分离
import abc  # 导入抽象基类模块
from typing import List, Tuple  # 导入类型提示

import torch  # 导入PyTorch张量库

from sglang.srt.mem_cache.memory_pool import KVCache  # 导入KV缓存基类


class BaseSWAKVPool(KVCache):  # SWA KV池抽象基类，继承自KVCache
    """ABC for SWA-like KV pools.
    SWA类型KV池的抽象基类

    Subclasses expose a `swa_kv_pool` sub-pool plus a full -> swa index
    mapping. Used by `SWATokenToKVPoolAllocator` and the disagg paths to
    handle SWA state separately from the full KV state.
    子类暴露一个swa_kv_pool子池以及完整索引到SWA索引的映射。
    由SWATokenToKVPoolAllocator和解聚路径使用，用于将SWA状态
    与完整KV状态分开处理。
    """

    swa_kv_pool: KVCache  # SWA子池，存储滑动窗口内的KV缓存

    def invalidate_loc_cache(self) -> None:  # 使位置缓存失效
        pass  # 基类中为空操作

    @abc.abstractmethod
    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:  # 注册完整索引到SWA索引的映射
        raise NotImplementedError()  # 抽象方法，子类必须实现

    @abc.abstractmethod
    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:  # 将完整KV索引转换为SWA索引
        raise NotImplementedError()  # 抽象方法，子类必须实现

    @abc.abstractmethod
    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:  # 获取状态缓冲区信息（大小、偏移等）
        raise NotImplementedError()  # 抽象方法，子类必须实现
