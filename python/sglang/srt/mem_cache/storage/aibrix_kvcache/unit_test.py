# AIBrix KVCache 存储后端单元测试
# 该模块对 AibrixKVCacheStorage 的批量设置、获取和存在性检查功能进行测试

import logging  # 导入日志模块
import os  # 导入操作系统模块

import torch  # 导入PyTorch深度学习框架
import torch.distributed  # 导入分布式训练模块
from aibrix_kvcache.common.absl_logging import log_every_n_seconds  # 导入限频日志函数
from aibrix_kvcache_storage import AibrixKVCacheStorage  # 导入AIBrix KVCache存储实现

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # 导入HiCache存储配置类
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # 导入MHA token到KV池的映射
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost  # 导入主机端MHA token到KV池

logging.basicConfig(  # 配置日志基本设置
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"  # 设置日志级别为INFO，格式包含时间、级别和消息
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def setup():  # 初始化分布式环境设置
    os.environ["RANK"] = "0"  # 设置当前进程的秩为0
    os.environ["WORLD_SIZE"] = "1"  # 设置总进程数为1
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点地址为本地
    os.environ["MASTER_PORT"] = "63886"  # 设置主节点端口为63886


class AIBrixKVCacheStorageTest:  # AIBrix KVCache存储测试类
    def test_with_page_size(self):  # 测试不同页面大小下的KV缓存操作
        config = HiCacheStorageConfig(  # 创建存储配置
            tp_rank=0,  # 张量并行秩为0
            tp_size=1,  # 张量并行大小为1
            is_mla_model=False,  # 不是MLA模型
            is_page_first_layout=True,  # 使用页优先布局
            model_name="test",  # 模型名称为test
        )
        for page_size in range(1, 3):  # 遍历页面大小1和2
            logger.info(f"page_size: {page_size}")  # 记录当前测试的页面大小
            batch_size = 2  # 批量大小为2
            head_num = 1  # 注意力头数为1
            layer_num = 64  # 层数为64
            head_dim = 128  # 每个头的维度为128
            kv_cache = MHATokenToKVPool(  # 创建MHA token到KV池的映射
                1024,  # 最大token数
                page_size,  # 页面大小
                torch.float16,  # 数据类型为float16
                head_num,  # 头数
                head_dim,  # 头维度
                layer_num,  # 层数
                "cpu",  # 设备为CPU
                False,  # 不使用混合精度
                0,  # 起始层
                layer_num,  # 结束层
            )
            mem_pool = MHATokenToKVPoolHost(kv_cache, 2, 0, page_size, "layer_first")  # 创建主机端MHA KV缓存池
            query_length = batch_size * 2  # 查询长度为批量大小的2倍
            partial = batch_size  # 部分键的数量等于批量大小
            self.aibrix_kvcache = AibrixKVCacheStorage(config, mem_pool)  # 创建AIBrix KVCache存储实例
            target_shape = (2, layer_num, page_size, head_num, head_dim)  # 目标张量形状
            rand_tensor = [  # 生成随机张量列表作为测试数据
                torch.rand(target_shape, dtype=torch.float16)  # 生成指定形状的随机float16张量
                for _ in range(query_length)  # 生成query_length个随机张量
            ]
            keys = ["hash" + str(i) for i in range(query_length)]  # 生成测试键列表
            partial_keys = keys[batch_size:query_length]  # 取部分键用于部分匹配测试
            assert self.aibrix_kvcache.batch_exists(keys) == 0  # 断言初始状态下所有键都不存在
            assert self.aibrix_kvcache.batch_set(keys, rand_tensor)  # 断言批量设置成功
            get_tensor = [  # 创建用于接收获取结果的张量列表
                torch.rand(target_shape, dtype=torch.float16).flatten()  # 生成展平的随机float16张量
                for _ in range(query_length)  # 生成query_length个张量
            ]
            self.aibrix_kvcache.batch_get(keys, get_tensor)  # 批量获取KV缓存数据
            for i in range(query_length):  # 遍历所有查询
                assert torch.equal(get_tensor[i], rand_tensor[i].flatten())  # 断言获取的数据与设置的数据一致
            ret = self.aibrix_kvcache.batch_exists(keys)  # 批量检查键是否存在
            assert self.aibrix_kvcache.batch_exists(keys) == query_length  # 断言所有键都存在
            assert self.aibrix_kvcache.batch_exists(partial_keys) == partial  # 断言部分键存在的数量正确
            partial_get_tensor = [  # 创建部分键获取结果的张量列表
                torch.rand(target_shape, dtype=torch.float16).flatten()  # 生成展平的随机float16张量
                for _ in range(partial)  # 生成partial个张量
            ]
            self.aibrix_kvcache.batch_get(partial_keys, partial_get_tensor)  # 批量获取部分键的KV缓存数据
            for i in range(partial):  # 遍历部分键
                assert torch.equal(  # 断言部分获取的数据与原始数据一致
                    partial_get_tensor[i], rand_tensor[i + partial].flatten()  # 比较部分获取结果和对应的原始数据
                )
            log_every_n_seconds(  # 限频日志输出
                logger,  # 日志记录器
                logging.INFO,  # 日志级别
                self.aibrix_kvcache.kv_cache_manager.metrics.summary(),  # 输出度量摘要
                1,  # 每秒最多输出一次
            )


if __name__ == "__main__":  # 主程序入口
    setup()  # 初始化分布式环境
    test = AIBrixKVCacheStorageTest()  # 创建测试实例
    test.test_with_page_size()  # 运行页面大小测试
