# AIBrix KVCache 存储后端实现
# 该模块实现了基于 AIBrix KVCache 的 HiCache 存储后端，用于 KV 缓存的远程存储和检索

import logging  # 导入日志模块
from typing import Any, List, Optional  # 导入类型提示

import torch  # 导入PyTorch深度学习框架
from aibrix_kvcache import (  # 从aibrix_kvcache包导入KV缓存相关组件
    BaseKVCacheManager,  # KV缓存基础管理器
    BlockHashes,  # 块哈希类
    KVCacheBlockLayout,  # KV缓存块布局
    KVCacheBlockSpec,  # KV缓存块规格
    KVCacheConfig,  # KV缓存配置
    KVCacheTensorSpec,  # KV缓存张量规格
    ModelSpec,  # 模型规格
)
from aibrix_kvcache.common.absl_logging import log_every_n_seconds  # 导入限频日志函数

from sglang.srt.mem_cache.hicache_storage import (  # 从SGLang的hicache_storage模块导入基础类
    HiCacheStorage,  # HiCache存储基类
    HiCacheStorageConfig,  # HiCache存储配置类
    HiCacheStorageExtraInfo,  # HiCache存储额外信息类
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # 导入主机端KV缓存内存池

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class AibrixKVCacheStorage(HiCacheStorage):  # AIBrix KVCache存储类，继承自HiCacheStorage基类
    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool: HostKVCache):  # 初始化方法，接收存储配置和内存池
        if storage_config is not None:  # 如果存储配置不为空
            self.is_mla_backend = storage_config.is_mla_model  # 判断是否为MLA模型后端
            self.local_rank = storage_config.tp_rank  # 获取当前张量并行秩
        else:  # 如果存储配置为空
            self.is_mla_backend = False  # 默认不是MLA后端
            self.local_rank = 0  # 默认秩为0
        kv_cache = mem_pool.device_pool  # 获取设备端KV缓存池
        self.page_size = mem_pool.page_size  # 获取页面大小
        self.kv_cache_dtype = kv_cache.dtype  # 获取KV缓存数据类型
        self.layer_num = kv_cache.layer_num  # 获取层数
        self.kv_head_ids = [  # 计算当前秩对应的KV头ID列表
            self.local_rank * kv_cache.head_num + i for i in range(kv_cache.head_num)  # 根据秩和头数计算头ID
        ]
        if not self.is_mla_backend:  # 如果不是MLA后端
            self.layer_ids = range(  # 计算层ID范围
                kv_cache.start_layer, kv_cache.end_layer  # 用于流水线并行的层范围
            )  # for pipeline parallel  # 用于流水线并行

            self.block_spec = KVCacheBlockSpec(  # 创建KV缓存块规格
                block_ntokens=self.page_size,  # 每块的token数量
                block_dtype=self.kv_cache_dtype,  # 块的数据类型
                block_layout=KVCacheBlockLayout(KVCacheBlockLayout.NCLD),  # 块的布局为NCLD格式
                tensor_spec=KVCacheTensorSpec(  # 张量规格
                    heads=self.kv_head_ids,  # 头ID列表
                    layers=self.layer_ids,  # 层ID列表
                    head_size=kv_cache.head_dim,  # 每个头的维度
                ),
            )
            logger.info(self.block_spec)  # 记录块规格信息
            config = KVCacheConfig(  # 创建KV缓存配置
                block_spec=self.block_spec, model_spec=ModelSpec(102400)  # 设置块规格和模型规格（最大token数102400）
            )
            self.kv_cache_manager = BaseKVCacheManager(config)  # 使用配置创建KV缓存管理器
        else:  # 如果是MLA后端
            raise NotImplementedError(  # 抛出未实现异常
                "MLA is not supported by AibrixKVCacheStorage yet."  # AIBrix KVCache存储尚不支持MLA
            )

    def _aibrix_kvcache_metrics_report(self):  # 内部方法：报告AIBrix KVCache的度量指标
        self.kv_cache_manager.metrics.summary()  # 输出度量摘要
        self.kv_cache_manager.metrics.reset()  # 重置度量数据

    def batch_get(  # 批量获取KV缓存数据
        self,
        keys: List[str],  # 键列表
        target_locations: List[torch.Tensor],  # 目标存储位置张量列表
        target_sizes: Optional[Any] = None,  # 目标大小（可选）
    ) -> List[torch.Tensor | None]:  # 返回张量列表或None
        block_hash = BlockHashes(keys, self.page_size)  # 根据键和页面大小创建块哈希
        status = self.kv_cache_manager.acquire(None, block_hash)  # 从KV缓存管理器获取数据
        log_every_n_seconds(  # 限频日志输出
            logger, logging.INFO, self._aibrix_kvcache_metrics_report(), 1  # 每秒最多输出一次指标报告
        )
        if status.is_ok():  # 如果获取成功
            num_fetched_tokens, handle = status.value  # 解包获取的token数量和句柄
            kv_blocks = handle.to_tensors()  # 将句柄转换为张量列表
            assert len(kv_blocks) == len(target_locations)  # 断言块数量与目标位置数量一致
            for i in range(len(kv_blocks)):  # 遍历每个块
                assert (  # 断言字节数匹配
                    target_locations[i].nbytes == kv_blocks[i].nbytes  # 检查目标位置和KV块的字节数是否一致
                ), f"{target_locations[i].nbytes}, {kv_blocks[i].nbytes}"  # 不匹配时输出两者字节数
                target_locations[i].copy_(kv_blocks[i].flatten())  # 将KV块展平后复制到目标位置
            handle.release()  # 释放句柄
            return target_locations  # 返回目标位置张量列表

        return [None] * len(keys)  # 获取失败则返回None列表

    def get(  # 获取单个KV缓存数据
        self,
        key: str,  # 键
        target_location: Optional[Any] = None,  # 目标存储位置（可选）
        target_size: Optional[Any] = None,  # 目标大小（可选）
    ) -> torch.Tensor | None:  # 返回张量或None
        return self.batch_get([key], [target_location], [target_size])[0]  # 委托给batch_get方法，取第一个结果

    def batch_set(  # 批量设置KV缓存数据
        self,
        keys: List[str],  # 键列表
        values: Optional[Any] = None,  # 值列表（可选）
        target_locations: Optional[Any] = None,  # 目标位置列表（可选）
        target_sizes: Optional[Any] = None,  # 目标大小列表（可选）
    ) -> bool:  # 返回是否成功
        block_hash = BlockHashes(keys, self.page_size)  # 根据键和页面大小创建块哈希
        status = self.kv_cache_manager.allocate_for(None, block_hash)  # 为块哈希分配空间
        if not status.is_ok():  # 如果分配失败
            logger.warning(  # 记录警告日志
                f"aibrix_kvcache set allocate failed, error_code {status.error_code}"  # 输出分配失败的错误码
            )
            return False  # 返回失败
        handle = status.value  # 获取分配句柄
        tensors = handle.to_tensors()  # 将句柄转换为张量列表
        if len(tensors) != len(values):  # 如果张量数量与值数量不匹配
            logger.warning("aibrix_kvcache set allocate not enough")  # 记录分配不足警告
            return False  # 返回失败
        for i in range(len(tensors)):  # 遍历每个张量
            assert (  # 断言字节数匹配
                tensors[i].nbytes == values[i].nbytes  # 检查张量和值的字节数是否一致
            ), f"{tensors[i].nbytes}, {values[i].nbytes}"  # 不匹配时输出两者字节数
            tensors[i].reshape(values[i].shape).copy_(values[i]).reshape(  # 将值复制到张量中并恢复原始形状
                tensors[i].shape  # 恢复为张量的原始形状
            )
        status = self.kv_cache_manager.put(None, block_hash, handle)  # 将数据写入KV缓存管理器
        if not status.is_ok():  # 如果写入失败
            logger.info(  # 记录信息日志
                f"AIBrix KVCache Storage set failed, error_code {status.error_code}"  # 输出写入失败的错误码
            )
            return False  # 返回失败
        completed = status.value  # 获取已完成的token数
        return completed == len(keys) * self.page_size  # 检查是否所有token都已写入

    def set(  # 设置单个KV缓存数据
        self,
        key: str,  # 键
        value: Optional[Any] = None,  # 值（可选）
        target_location: Optional[Any] = None,  # 目标位置（可选）
        target_size: Optional[Any] = None,  # 目标大小（可选）
    ) -> bool:  # 返回是否成功
        return self.batch_set([key], [value], [target_location], [target_size])  # 委托给batch_set方法

    def batch_exists(  # 批量检查KV缓存键是否存在
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None  # 键列表和额外信息（可选）
    ) -> int:  # 返回存在的页面数量
        block_hash = BlockHashes(keys, self.page_size)  # 根据键和页面大小创建块哈希
        status = self.kv_cache_manager.exists(None, block_hash)  # 检查键是否存在
        if status.is_ok():  # 如果检查成功
            return status.value // self.page_size  # 返回存在的token数除以页面大小得到的页面数
        return 0  # 检查失败则返回0

    def exists(self, key: str) -> bool | dict:  # 检查单个键是否存在
        return self.batch_exists([key]) > 0  # 委托给batch_exists方法，判断结果是否大于0
