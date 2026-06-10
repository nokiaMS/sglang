# Mooncake嵌入存储模块
# 本模块实现了基于Mooncake分布式存储的嵌入向量存储，
# 提供批量GET、PUT和存在性检查操作，用于在分布式环境中
# 存储和检索图像嵌入向量，支持RDMA高效传输。

import logging  # 导入日志模块 # 导入日志模块
from typing import Any, List  # 导入类型提示工具 # 导入类型提示工具

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeBaseStore  # 导入Mooncake基础存储类 # 导入Mooncake基础存储类

logger = logging.getLogger(__name__)  # 获取当前模块日志记录器 # 获取当前模块日志记录器


class MooncakeEmbeddingStore(MooncakeBaseStore):  # Mooncake嵌入存储类，继承自MooncakeBaseStore # Mooncake嵌入存储类
    def __init__(  # 初始化方法 # 初始化方法
        self,
        storage_config: Any = None,  # 存储配置，可选 # 存储配置
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化

        MooncakeDistributedStore = self._import_mooncake_store()  # 动态导入Mooncake分布式存储类 # 动态导入Mooncake分布式存储类
        self.store = MooncakeDistributedStore()  # 创建分布式存储实例 # 创建分布式存储实例
        self.config = self._load_config(storage_config)  # 加载存储配置 # 加载存储配置
        ret_code = self.store.setup(  # 设置分布式存储 # 设置分布式存储
            self.config.local_hostname,  # 本地主机名 # 本地主机名
            self.config.metadata_server,  # 元数据服务器地址 # 元数据服务器地址
            self.config.global_segment_size,  # 全局段大小 # 全局段大小
            16 * 1024 * 1024,  # Internal local buffer size # 内部本地缓冲区大小（16MB） # 内部本地缓冲区大小
            self.config.protocol,  # 通信协议 # 通信协议
            self.config.device_name,  # 设备名称 # 设备名称
            self.config.master_server_address,  # 主服务器地址 # 主服务器地址
        )
        if ret_code != 0:  # 如果设置失败 # 如果设置失败
            raise RuntimeError(f"Failed to setup Mooncake Embedding Store: {ret_code}")  # 抛出运行时错误 # 抛出运行时错误

        logger.info("Mooncake Embedding Store initialized successfully.")  # 记录初始化成功日志 # 记录初始化成功日志

    def get_key(self, image_hash: str) -> str:  # 根据图像哈希生成存储键 # 根据图像哈希生成存储键
        return f"emb_{image_hash}"  # 添加emb_前缀作为键 # 添加emb_前缀作为键

    def batch_get(  # 批量获取嵌入数据 # 批量获取嵌入数据
        self, hashes: List[str], ptrs: List[int], sizes: List[int]  # 哈希列表、指针列表、大小列表 # 哈希列表、指针列表、大小列表
    ) -> List[bool]:  # 返回每个键的获取成功标志列表 # 返回成功标志列表
        keys = [self.get_key(h) for h in hashes]  # 将哈希转换为存储键 # 将哈希转换为存储键
        results = self.store.batch_get_into(keys, ptrs, sizes)  # 批量读取数据到指定指针 # 批量读取数据
        return [res > 0 for res in results]  # 结果大于0表示成功 # 结果大于0表示成功

    def batch_put(  # 批量存储嵌入数据 # 批量存储嵌入数据
        self, hashes: List[str], ptrs: List[int], sizes: List[int]  # 哈希列表、指针列表、大小列表 # 哈希列表、指针列表、大小列表
    ) -> List[bool]:  # 返回每个键的存储成功标志列表 # 返回成功标志列表
        keys = [self.get_key(h) for h in hashes]  # 将哈希转换为存储键 # 将哈希转换为存储键
        exists = self.store.batch_is_exist(keys)  # 批量检查键是否已存在 # 批量检查键是否已存在

        put_keys, put_ptrs, put_sizes, indices = [], [], [], []  # 需要PUT的键、指针、大小和索引列表 # 需要PUT的列表
        success_map = [True] * len(hashes)  # 初始化成功映射，默认全部成功 # 初始化成功映射

        for i, status in enumerate(exists):  # 遍历存在性检查结果 # 遍历存在性检查结果
            if status != 1:  # 如果键不存在 # 如果键不存在
                put_keys.append(keys[i])  # 添加需要PUT的键 # 添加需要PUT的键
                put_ptrs.append(ptrs[i])  # 添加需要PUT的指针 # 添加需要PUT的指针
                put_sizes.append(sizes[i])  # 添加需要PUT的大小 # 添加需要PUT的大小
                indices.append(i)  # 记录索引 # 记录索引

        if put_keys:  # 如果有需要PUT的键 # 如果有需要PUT的键
            results = self.store.batch_put_from(put_keys, put_ptrs, put_sizes)  # 批量写入数据 # 批量写入数据
            for i, res in enumerate(results):  # 遍历写入结果 # 遍历写入结果
                success_map[indices[i]] = res == 0  # 返回码0表示成功 # 返回码0表示成功
        return success_map  # 返回成功映射 # 返回成功映射

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:  # 批量检查嵌入是否存在 # 批量检查嵌入是否存在
        keys = [self.get_key(h) for h in hashes]  # 将哈希转换为存储键 # 将哈希转换为存储键
        results = self.store.batch_is_exist(keys)  # 批量检查键是否存在 # 批量检查键是否存在
        return [res == 1 for res in results]  # 结果等于1表示存在 # 结果等于1表示存在
