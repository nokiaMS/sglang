# LMCache单元测试模块
# 本模块测试LMCache按层连接器的存储和加载元数据功能，
# 验证KV缓存在存储后能被正确检索和恢复，确保数据一致性。

try:  # 尝试导入LMCache相关模块 # 尝试导入LMCache相关模块
    from lmcache.integration.sglang.sglang_adapter import (  # 导入SGLang适配器 # 导入SGLang适配器
        LMCacheLayerwiseConnector,  # 按层连接器 # 按层连接器
        LoadMetadata,  # 加载元数据 # 加载元数据
        StoreMetadata,  # 存储元数据 # 存储元数据
    )
except ImportError:  # 导入失败时抛出运行时错误 # 导入失败时抛出运行时错误
    raise RuntimeError(
        "LMCache is not installed. Please install it by running `pip install lmcache` in the root directory of LMCache"  # 提示安装LMCache # 提示安装LMCache
    )

import torch  # 导入PyTorch深度学习框架 # 导入PyTorch深度学习框架

from sglang.srt.configs.model_config import ModelConfig  # 导入模型配置类 # 导入模型配置类


def test_load_store_metadata():  # 测试加载和存储元数据功能 # 测试加载和存储元数据功能
    model_config = ModelConfig(  # 创建模型配置对象 # 创建模型配置对象
        model_path="Qwen/Qwen3-4B",  # 模型路径 # 模型路径
    )

    # Generate Dummy KV Cache # 生成虚拟KV缓存 # 生成虚拟KV缓存
    head_num = model_config.num_key_value_heads  # KV头数 # KV头数
    head_dim = model_config.head_dim  # 头维度 # 头维度
    layer_num = model_config.num_hidden_layers  # 隐藏层数 # 隐藏层数
    buffer_size = 256  # 缓冲区大小 # 缓冲区大小
    input_id_len = 16  # 输入ID长度 # 输入ID长度

    k_buffer = [  # K缓冲区列表 # K缓冲区列表
        torch.randn(buffer_size, head_num, head_dim, dtype=torch.bfloat16).cuda()  # 生成随机K缓存数据并放到GPU # 生成随机K缓存数据
        for _ in range(layer_num)  # 每层一个 # 每层一个
    ]
    v_buffer = [  # V缓冲区列表 # V缓冲区列表
        torch.randn(buffer_size, head_num, head_dim, dtype=torch.bfloat16).cuda()  # 生成随机V缓存数据并放到GPU # 生成随机V缓存数据
        for _ in range(layer_num)  # 每层一个 # 每层一个
    ]

    connector = LMCacheLayerwiseConnector(  # 创建按层连接器 # 创建按层连接器
        model_config, 1, 0, k_buffer, v_buffer, config_file="example_config_ip.yaml"  # 传入模型配置、并行参数、缓冲区和配置文件 # 传入模型配置、并行参数、缓冲区和配置文件
    )

    fake_token_ids = torch.randint(0, model_config.vocab_size, (input_id_len,)).tolist()  # 生成伪token ID # 生成伪token ID
    fake_kv_indices = torch.randint(0, buffer_size, (input_id_len,))  # 生成伪KV索引 # 生成伪KV索引
    offset = 0  # 偏移量设为0 # 偏移量设为0

    store_metadata = StoreMetadata(  # 创建存储元数据 # 创建存储元数据
        last_node=None,  # 最后节点为空 # 最后节点为空
        token_ids=fake_token_ids,  # token ID列表 # token ID列表
        kv_indices=fake_kv_indices,  # KV索引 # KV索引
        offset=offset,  # 偏移量 # 偏移量
    )

    load_metadata = LoadMetadata(  # 创建加载元数据 # 创建加载元数据
        token_ids=fake_token_ids,  # token ID列表 # token ID列表
        slot_mapping=fake_kv_indices,  # 槽位映射 # 槽位映射
        offset=offset,  # 偏移量 # 偏移量
    )

    current_stream = torch.cuda.current_stream()  # 获取当前CUDA流 # 获取当前CUDA流

    retrieve_token_num = connector.start_load_kv(load_metadata)  # 尝试加载（此时无数据可加载） # 尝试加载
    assert retrieve_token_num == 0  # 断言没有检索到token（因为还没存储） # 断言没有检索到token

    connector.store_kv(store_metadata)  # 存储KV数据 # 存储KV数据
    current_stream.synchronize()  # 同步当前流确保存储完成 # 同步当前流

    # check retrieve # 检查检索结果 # 检查检索结果
    gt_key_buffer = [  # 真实K值缓冲区（用于对比） # 真实K值缓冲区
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()  # 初始化为零 # 初始化为零
        for _ in range(layer_num)  # 每层一个 # 每层一个
    ]
    gt_value_buffer = [  # 真实V值缓冲区（用于对比） # 真实V值缓冲区
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()  # 初始化为零 # 初始化为零
        for _ in range(layer_num)  # 每层一个 # 每层一个
    ]

    for i in range(layer_num):  # 遍历每一层 # 遍历每一层
        gt_key_buffer[i] = k_buffer[i][fake_kv_indices]  # 从K缓冲区提取真实K值 # 从K缓冲区提取真实K值
        gt_value_buffer[i] = v_buffer[i][fake_kv_indices]  # 从V缓冲区提取真实V值 # 从V缓冲区提取真实V值

    # clear the k_buffer and v_buffer # 清空k_buffer和v_buffer # 清空k_buffer和v_buffer
    for _ in range(layer_num):  # 遍历每一层 # 遍历每一层
        k_buffer[i].zero_()  # 将K缓冲区清零 # 将K缓冲区清零
        v_buffer[i].zero_()  # 将V缓冲区清零 # 将V缓冲区清零

    retrieve_token_num = connector.start_load_kv(load_metadata)  # 再次尝试加载（此时有数据可加载） # 再次尝试加载
    assert retrieve_token_num == input_id_len  # 断言检索到的token数等于输入长度 # 断言检索到的token数等于输入长度

    for i in range(layer_num):  # 逐层加载 # 逐层加载
        current_stream.synchronize()  # 同步当前流 # 同步当前流
        connector.load_kv_layerwise(i)  # 按层加载KV # 按层加载KV

    current_stream.synchronize()  # 同步当前流确保加载完成 # 同步当前流
    test_key_buffer = [  # 测试K值缓冲区 # 测试K值缓冲区
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()  # 初始化为零 # 初始化为零
        for _ in range(layer_num)  # 每层一个 # 每层一个
    ]
    test_value_buffer = [  # 测试V值缓冲区 # 测试V值缓冲区
        torch.zeros(input_id_len, head_num, head_dim, dtype=torch.bfloat16).cuda()  # 初始化为零 # 初始化为零
        for _ in range(layer_num)  # 每层一个 # 每层一个
    ]

    for i in range(layer_num):  # 遍历每一层 # 遍历每一层
        test_key_buffer[i] = k_buffer[i][fake_kv_indices]  # 从加载后的K缓冲区提取测试K值 # 提取测试K值
        test_value_buffer[i] = v_buffer[i][fake_kv_indices]  # 从加载后的V缓冲区提取测试V值 # 提取测试V值

    for i in range(layer_num):  # 逐层验证 # 逐层验证
        assert torch.allclose(test_key_buffer[i], gt_key_buffer[i])  # 断言K值一致 # 断言K值一致
        assert torch.allclose(test_value_buffer[i], gt_value_buffer[i])  # 断言V值一致 # 断言V值一致

    print("================================================")  # 打印分隔线 # 打印分隔线
    print("TEST_LOAD_STORE_METADATA PASSED!")  # 打印测试通过信息 # 打印测试通过信息
    print("================================================")  # 打印分隔线 # 打印分隔线
    connector.close()  # 关闭连接器 # 关闭连接器


if __name__ == "__main__":  # 主入口 # 主入口
    test_load_store_metadata()  # 运行测试 # 运行测试
