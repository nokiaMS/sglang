# 文件说明：HiCacheSiMM存储后端的测试模块
# 包含单键操作测试和批量操作测试，验证SiMM存储后端的set/get/exists/batch_set/batch_get等功能，
# 支持MHA和MLA两种模式的配置测试。

import logging  # 日志记录库
import uuid  # 唯一标识符生成

import torch  # PyTorch深度学习框架

from python.sglang.srt.mem_cache.storage.simm.hicache_simm import HiCacheSiMM  # 导入SiMM存储后端
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # 导入存储配置类

logging.basicConfig(  # 配置日志基本设置
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"  # 设置日志级别和格式
)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def generate_batch_query_keys(kv_num: int, config: HiCacheStorageConfig):  # 生成批量查询用的键集合
    keys = ["test_" + str(uuid.uuid4()) for _ in range(kv_num)]  # 生成指定数量的唯一测试键
    set_keys = []  # 初始化设置键列表
    for key in keys:  # 遍历所有键
        if config.is_mla_model:  # 如果是MLA模型
            set_keys.append(key + "_k")  # MLA只添加K键
        else:  # MHA模型
            set_keys.append(key + f"_{config.tp_rank}_k")  # MHA添加带TP排名的K键
            set_keys.append(key + f"_{config.tp_rank}_v")  # MHA添加带TP排名的V键
    get_keys = set_keys  # 获取键与设置键相同
    exist_keys = keys  # 存在性检查使用原始键
    return set_keys, get_keys, exist_keys  # 返回设置键、获取键和存在性检查键


def create_mock_host_kv_cache(buffer_size, dtype=torch.float32):  # 创建模拟的HostKVCache对象用于测试
    """Create a mock HostKVCache-like object for testing."""  # 创建用于测试的模拟HostKVCache对象
    buffer = torch.randn(buffer_size, dtype=dtype)  # 生成随机数据作为缓冲区

    class MockHostKVCache:  # 模拟HostKVCache类
        def __init__(self, buffer):  # 初始化方法
            self.kv_buffer = buffer  # KV缓冲区
            self.layout = "page_first"  # 使用页优先布局
            self.page_size = 1  # Simple page size for testing  # 简单页大小用于测试

        def get_page_buffer_meta(self, indices):  # 获取页缓冲区元数据的模拟实现
            """Mock implementation of get_page_buffer_meta."""  # get_page_buffer_meta的模拟实现
            ptr_list = []  # 指针列表
            element_size_list = []  # 元素大小列表
            for idx in indices:  # 遍历所有索引
                # Create mock pointers and sizes for each page  # 为每个页创建模拟指针和大小
                ptr_list.append(idx * self.page_size * self.kv_buffer.element_size())  # 计算模拟指针偏移
                element_size_list.append(self.page_size * self.kv_buffer.element_size())  # 计算模拟元素大小
            return ptr_list, element_size_list  # 返回指针列表和元素大小列表

    return MockHostKVCache(buffer), buffer  # 返回模拟对象和缓冲区


def test_single_operation():  # 测试单键操作（set/get/exists）
    """Test the set API with a single key-value pair."""  # 测试单个键值对的set API
    print("=" * 100)  # 打印分隔线
    print("Testing single operation")  # 打印测试标题

    buffer_size = 1024 * 1024 * 16  # 16MB  # 缓冲区大小16MB
    value_elements = 1024  # 值元素数量
    store = HiCacheSiMM()  # 创建SiMM存储实例（无配置）
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(buffer_size)  # 创建模拟主机KV缓存

    # Register the memory pool host - this is the proper workflow  # 注册主机端内存池 - 这是正确的工作流程
    store.register_mem_pool_host(mock_host_kv_cache)  # 注册内存池

    value_size = value_elements * buffer.element_size()  # 计算值的大小（字节）

    key = str(uuid.uuid4())  # 生成唯一键
    set_slice = buffer[:value_elements]  # 获取设置用的缓冲区切片
    get_slice = buffer[value_elements : 2 * value_elements]  # 获取读取用的缓冲区切片
    set_location = set_slice.data_ptr()  # 获取设置切片的数据指针
    get_location = get_slice.data_ptr()  # 获取读取切片的数据指针

    # Test set operation  # 测试set操作
    result = store.set(key, target_location=set_location, target_sizes=value_size)  # 执行set操作
    assert result is True, f"❌set operation failed for key: {key}"  # 断言set成功

    # Test exists operation  # 测试exists操作
    assert store.exists(key), f"❌key {key} should exist after set operation"  # 断言键存在

    # Test get operation  # 测试get操作
    result = store.get(key, target_location=get_location, target_sizes=value_size)  # 执行get操作
    assert result is True, f"❌get operation failed for key: {key}"  # 断言get成功

    # Compare the data using proper tensor indices  # 使用正确的张量索引比较数据
    assert torch.allclose(
        set_slice, get_slice, atol=1e-6  # 比较设置和读取的数据是否一致
    ), f"❌get operation failed for key: {key}"  # 断言数据一致

    logger.info(f"✅ Single operation passed")  # 记录单操作测试通过


def test_batch_operation(config: HiCacheStorageConfig):  # 测试批量操作（batch_set/batch_get/batch_exists）
    """Test the batch set/get APIs with multiple key-value pairs."""  # 测试多键值对的批量set/get API
    print("=" * 100)  # 打印分隔线
    print(f"Testing batch operation with config: {config}")  # 打印测试标题和配置

    buffer_size = 1024 * 1024 * 16  # 16MB  # 缓冲区大小16MB
    value_elements = 256  # 每个值的元素数量
    kv_num = 13  # 键值对数量
    store = HiCacheSiMM(config)  # 创建带配置的SiMM存储实例
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(buffer_size)  # 创建模拟主机KV缓存

    store.register_mem_pool_host(mock_host_kv_cache)  # 注册内存池

    value_size = value_elements * buffer.element_size()  # 计算值的大小（字节）

    set_keys, get_keys, exist_keys = generate_batch_query_keys(kv_num, config)  # 生成批量查询键
    set_slices = [  # 创建设置用的缓冲区切片列表
        buffer[i * value_elements : (i + 1) * value_elements]  # 每个切片包含value_elements个元素
        for i in range(len(set_keys))  # 遍历所有设置键
    ]
    set_indices = torch.cat(set_slices)  # 将所有设置切片拼接为索引张量

    # Test batch set operation  # 测试批量set操作
    result = store.batch_set_v1(set_keys, set_indices)  # 执行批量set操作
    assert all(result), f"❌batch set operation failed"  # 断言全部set成功

    # Test batch exists operation  # 测试批量exists操作
    assert store.batch_exists(
        exist_keys  # 检查所有键是否存在
    ), f"❌keys should exist after batch set operation"  # 断言所有键存在

    # Test batch get operation  # 测试批量get操作
    get_slices = [  # 创建获取用的缓冲区切片列表
        buffer[
            (len(set_keys) + i)
            * value_elements : (len(set_keys) + i + 1)
            * value_elements  # 偏移到set区域之后
        ]
        for i in range(len(get_keys))  # 遍历所有获取键
    ]
    get_indices = torch.cat(get_slices)  # 将所有获取切片拼接为索引张量
    result = store.batch_get_v1(get_keys, get_indices)  # 执行批量get操作
    assert all(result), f"❌batch get operation failed"  # 断言全部get成功
    for i in range(len(get_keys)):  # 遍历所有获取键
        assert torch.allclose(
            set_slices[i], get_slices[i], atol=1e-6  # 比较设置和读取的数据是否一致
        ), f"❌batch get operation failed for key: {get_keys[i]}"  # 断言每个键的数据一致

    logger.info(f"✅ Batch operation passed")  # 记录批量操作测试通过


if __name__ == "__main__":  # 主入口
    test_single_operation()  # 运行单键操作测试
    test_batch_operation(  # 运行批量操作测试 - MHA模式，TP排名0
        HiCacheStorageConfig(
            is_mla_model=False,  # 非MLA模型
            tp_rank=0,  # TP排名为0
            tp_size=1,  # TP大小为1
            model_name=None,  # 无模型名称
            is_page_first_layout=True,  # 使用页优先布局
        )
    )
    test_batch_operation(  # 运行批量操作测试 - MLA模式，TP排名0
        HiCacheStorageConfig(
            is_mla_model=True,  # MLA模型
            tp_rank=0,  # TP排名为0
            tp_size=1,  # TP大小为1
            model_name=None,  # 无模型名称
            is_page_first_layout=True,  # 使用页优先布局
        )
    )
    test_batch_operation(  # 运行批量操作测试 - MHA模式，TP排名1
        HiCacheStorageConfig(
            is_mla_model=False,  # 非MLA模型
            tp_rank=1,  # TP排名为1
            tp_size=4,  # TP大小为4
            model_name=None,  # 无模型名称
            is_page_first_layout=True,  # 使用页优先布局
        )
    )
    test_batch_operation(  # 运行批量操作测试 - MLA模式，TP排名3
        HiCacheStorageConfig(
            is_mla_model=True,  # MLA模型
            tp_rank=3,  # TP排名为3
            tp_size=8,  # TP大小为8
            model_name=None,  # 无模型名称
            is_page_first_layout=True,  # 使用页优先布局
        )
    )
    logger.info(f"✅ All tests passed")  # 记录所有测试通过
