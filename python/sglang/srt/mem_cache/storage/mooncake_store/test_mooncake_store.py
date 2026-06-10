# Mooncake存储后端测试模块
# 本模块包含MooncakeStore的单键操作和批量操作的测试用例，
# 验证set/get/exists等基本功能，以及MLA和MHA模式下不同TP配置的批量操作。

import logging  # 导入logging模块，用于日志记录
import uuid  # 导入uuid模块，用于生成唯一标识符

import torch  # 导入torch模块，用于张量操作

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # 从hicache_storage模块导入存储配置类
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore  # 从mooncake_store模块导入MooncakeStore类

logging.basicConfig(  # 配置日志
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"  # 设置日志级别为INFO，格式包含时间、级别和消息
)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def make_hicache_storage_config(  # 创建HiCache存储配置 # 构建用于测试的HiCacheStorageConfig对象
    *,
    is_mla_model: bool,  # 是否为MLA模型
    tp_rank: int,  # 张量并行rank
    tp_size: int,  # 张量并行大小
) -> HiCacheStorageConfig:  # 返回HiCache存储配置对象
    return HiCacheStorageConfig(  # 返回配置实例
        tp_rank=tp_rank,  # 张量并行rank
        tp_size=tp_size,  # 张量并行大小
        pp_rank=0,  # 流水线并行rank，测试中固定为0
        pp_size=1,  # 流水线并行大小，测试中固定为1
        attn_cp_rank=0,  # 注意力上下文并行rank，测试中固定为0
        attn_cp_size=1,  # 注意力上下文并行大小，测试中固定为1
        is_mla_model=is_mla_model,  # 是否为MLA模型
        enable_storage_metrics=False,  # 不启用存储指标
        is_page_first_layout=True,  # 使用页优先布局
        model_name=None,  # 模型名称，测试中不指定
    )


def generate_batch_query_keys(kv_num: int):  # 生成批量查询键 # 生成指定数量的唯一测试键
    return ["test_" + str(uuid.uuid4()) for _ in range(kv_num)]  # 生成以"test_"为前缀的UUID键列表


def create_mock_host_kv_cache(  # 创建模拟的HostKVCache对象 # 创建用于测试的模拟主机KV缓存对象
    buffer_size,  # 缓冲区大小
    entries_per_page=2,  # 每页条目数，默认2
    page_elements=1,  # 每个条目的元素数，默认1
    dtype=torch.float32,  # 数据类型，默认float32
):
    """Create a mock HostKVCache-like object for testing."""  # 创建用于测试的模拟HostKVCache对象。 # 创建用于测试的模拟HostKVCache对象
    buffer = torch.randn(buffer_size, dtype=dtype)  # 创建随机缓冲区张量

    class MockHostKVCache:  # 模拟HostKVCache内部类
        def __init__(self, buffer, entries_per_page, page_elements):  # 初始化方法
            self.kv_buffer = buffer  # KV缓冲区
            self.layout = "page_first"  # 内存布局为page_first
            self.page_size = 1  # Simple page size for testing  # 简单页大小，用于测试
            self.entries_per_page = entries_per_page  # 每页条目数
            self.page_elements = page_elements  # 每个条目的元素数

        def get_page_buffer_meta(self, indices):  # 获取页缓冲区元数据 # 模拟的get_page_buffer_meta方法实现
            """Mock implementation of get_page_buffer_meta."""  # get_page_buffer_meta的模拟实现。
            ptr_list = []  # 指针列表
            element_size_list = []  # 元素大小列表
            for idx in indices:  # 遍历每个索引
                page_idx = int(idx)  # 将索引转为整数
                page_offset = page_idx * self.entries_per_page * self.page_elements  # 计算页偏移量
                for entry_idx in range(self.entries_per_page):  # 遍历每个条目
                    offset = page_offset + entry_idx * self.page_elements  # 计算条目偏移量
                    ptr_list.append(self.kv_buffer[offset:].data_ptr())  # 添加从偏移量开始的指针
                    element_size_list.append(  # 添加元素大小
                        self.page_elements * self.kv_buffer.element_size()  # 每个条目的字节数
                    )
            return ptr_list, element_size_list  # 返回指针列表和元素大小列表

        def get_ksize_per_token(self):  # 获取每个token的K大小 # 返回每个token的KV缓存字节数
            return (  # 返回计算结果
                self.entries_per_page  # 每页条目数
                * self.page_elements  # 每个条目的元素数
                * self.kv_buffer.element_size()  # 每个元素的字节大小
            )

    return MockHostKVCache(buffer, entries_per_page, page_elements), buffer  # 返回模拟对象和缓冲区


def test_single_operation():  # 测试单键操作 # 测试set/get/exists等单键操作
    """Test the set API with a single key-value pair."""  # 使用单个键值对测试set API。
    print("=" * 100)  # 打印分隔线
    print("Testing single operation")  # 打印测试标题

    buffer_size = 1024 * 1024 * 16  # 16MB  # 缓冲区大小为16MB
    value_elements = 1024  # 每个值的元素数
    store = MooncakeStore(  # 创建MooncakeStore实例
        make_hicache_storage_config(is_mla_model=False, tp_rank=0, tp_size=1)  # 非MLA模型，TP rank为0，TP大小为1
    )
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(  # 创建模拟主机KV缓存
        buffer_size,  # 缓冲区大小
        entries_per_page=2,  # 每页2个条目
        page_elements=value_elements,  # 每个条目的元素数
    )

    # Register the memory pool host - this is the proper workflow  # 注册主机内存池 - 这是正确的工作流程
    store.register_mem_pool_host(mock_host_kv_cache)  # 注册模拟主机KV缓存

    value_size = value_elements * buffer.element_size()  # 计算值的字节大小

    key = str(uuid.uuid4())  # 生成唯一键
    set_slice = buffer[:value_elements]  # 写入数据切片
    get_slice = buffer[value_elements : 2 * value_elements]  # 读取数据切片
    set_location = set_slice.data_ptr()  # 获取写入位置指针
    get_location = get_slice.data_ptr()  # 获取读取位置指针

    # Test set operation  # 测试set操作
    result = store.set(key, target_location=set_location, target_sizes=value_size)  # 执行set操作
    assert result is True, f"❌set operation failed for key: {key}"  # 断言set操作成功

    # Test exists operation  # 测试exists操作
    assert store.exists(key), f"❌key {key} should exist after set operation"  # 断言键在set后存在

    # Test get operation  # 测试get操作
    result = store.get(key, target_location=get_location, target_sizes=value_size)  # 执行get操作
    assert result is True, f"❌get operation failed for key: {key}"  # 断言get操作成功

    # Compare the data using proper tensor indices  # 使用正确的张量索引比较数据
    assert torch.allclose(  # 断言两个切片数据近似相等
        set_slice, get_slice, atol=1e-6  # 比较写入和读取的数据，容差1e-6
    ), f"❌get operation failed for key: {key}"  # get操作数据不匹配

    logger.info(f"✅ Single operation passed")  # 记录单键操作测试通过


def test_batch_operation(config: HiCacheStorageConfig):  # 测试批量操作 # 测试批量set/get操作
    """Test the batch set/get APIs with multiple key-value pairs."""  # 使用多个键值对测试批量set/get API。
    print("=" * 100)  # 打印分隔线
    print(f"Testing batch operation with config: {config}")  # 打印测试标题和配置

    buffer_size = 1024 * 1024 * 16  # 16MB  # 缓冲区大小为16MB
    value_elements = 256  # 每个值的元素数
    kv_num = 13  # KV对数量
    entries_per_page = 1 if config.is_mla_model else 2  # MLA模型每页1个条目，否则2个
    store = MooncakeStore(config)  # 使用给定配置创建MooncakeStore实例
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(  # 创建模拟主机KV缓存
        buffer_size,  # 缓冲区大小
        entries_per_page=entries_per_page,  # 每页条目数
        page_elements=value_elements,  # 每个条目的元素数
    )

    store.register_mem_pool_host(mock_host_kv_cache)  # 注册模拟主机KV缓存

    keys = generate_batch_query_keys(kv_num)  # 生成批量查询键
    set_slices = [  # 写入数据切片列表
        buffer[i * value_elements : (i + 1) * value_elements]  # 每个切片的起始和结束位置
        for i in range(kv_num * entries_per_page)  # 遍历所有KV对和条目
    ]
    set_indices = torch.arange(kv_num)  # 写入索引张量

    # Test batch set operation  # 测试批量set操作
    result = store.batch_set_v1(keys, set_indices)  # 执行批量set操作
    assert all(result), "batch set operation failed"  # 断言所有set操作成功

    # Test batch exists operation  # 测试批量exists操作
    assert (  # 断言所有键在批量set后存在
        store.batch_exists(keys) == kv_num  # 存在的键数应等于kv_num
    ), "keys should exist after batch set operation"  # 批量set后键应存在

    # Test batch get operation  # 测试批量get操作
    get_slices = [  # 读取数据切片列表
        buffer[  # 从缓冲区获取切片
            (kv_num * entries_per_page + i)  # 偏移量：跳过set使用的数据
            * value_elements : (kv_num * entries_per_page + i + 1)  # 结束位置
            * value_elements  # 乘以元素大小
        ]
        for i in range(kv_num * entries_per_page)  # 遍历所有KV对和条目
    ]
    get_indices = torch.arange(kv_num, 2 * kv_num)  # 读取索引张量（偏移kv_num）
    result = store.batch_get_v1(keys, get_indices)  # 执行批量get操作
    assert all(result), "❌batch get operation failed"  # 断言所有get操作成功
    for i in range(kv_num * entries_per_page):  # 遍历每个条目
        assert torch.allclose(  # 断言写入和读取数据近似相等
            set_slices[i], get_slices[i], atol=1e-6  # 比较写入和读取的切片，容差1e-6
        ), f"❌batch get operation failed for key: {keys[i // entries_per_page]}"  # 批量get操作数据不匹配

    logger.info(f"✅ Batch operation passed")  # 记录批量操作测试通过


if __name__ == "__main__":  # 主入口
    test_single_operation()  # 运行单键操作测试
    test_batch_operation(  # 运行非MLA TP=1的批量操作测试
        make_hicache_storage_config(is_mla_model=False, tp_rank=0, tp_size=1)  # 非MLA模型，TP rank 0，TP大小1
    )
    test_batch_operation(  # 运行MLA TP=1的批量操作测试
        make_hicache_storage_config(is_mla_model=True, tp_rank=0, tp_size=1)  # MLA模型，TP rank 0，TP大小1
    )
    test_batch_operation(  # 运行非MLA TP=4的批量操作测试
        make_hicache_storage_config(is_mla_model=False, tp_rank=1, tp_size=4)  # 非MLA模型，TP rank 1，TP大小4
    )
    test_batch_operation(  # 运行MLA TP=8的批量操作测试
        make_hicache_storage_config(is_mla_model=True, tp_rank=3, tp_size=8)  # MLA模型，TP rank 3，TP大小8
    )
    logger.info(f"✅ All tests passed")  # 记录所有测试通过
