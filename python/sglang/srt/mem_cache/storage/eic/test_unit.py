# EIC存储单元测试模块
# 本文件用于测试EIC（External Inference Cache）存储客户端的set、get、exists功能
# 通过YAML配置文件初始化EIC客户端，并执行批量读写和存在性检查操作

import argparse  # 导入命令行参数解析模块 # 命令行参数解析库
import os  # 导入操作系统接口模块 # 操作系统接口库

import eic  # 导入EIC客户端库 # EIC客户端库
import torch  # 导入PyTorch张量库 # PyTorch张量库
import yaml  # 导入YAML配置文件解析库 # YAML解析库


def pase_args():  # 解析命令行参数函数
    parser = argparse.ArgumentParser(description="EIC Storage Unit Test")  # 创建参数解析器，描述为EIC存储单元测试 # 创建命令行参数解析器
    parser.add_argument(  # 添加配置文件参数 # 添加命令行参数
        "--config",  # 参数名：--config # 配置文件路径参数
        "-c",  # 参数短名：-c # 短参数名
        type=str,  # 参数类型为字符串 # 参数类型为字符串
        default="/sgl-workspace/config/remote-eic.yaml",  # 默认配置文件路径 # 默认配置路径
        help="EIC yaml config",  # 参数帮助信息 # EIC的YAML配置文件
    )
    args, _ = parser.parse_known_args()  # 解析已知参数，忽略未知参数 # 解析已知参数
    return args  # 返回解析后的参数对象 # 返回参数


def init_eic_client():  # 初始化EIC客户端函数
    args = pase_args()  # 解析命令行参数 # 获取命令行参数
    config_path = os.path.abspath(args.config)  # 获取配置文件的绝对路径 # 获取配置文件绝对路径
    if not os.path.exists(config_path):  # 检查配置文件是否存在 # 检查配置文件是否存在
        raise FileNotFoundError(f"Config file not found: {config_path}")  # 文件不存在则抛出异常 # 配置文件未找到时抛出异常
    with open(config_path, "r") as fin:  # 打开配置文件 # 打开配置文件
        config = yaml.safe_load(fin)  # 安全加载YAML配置 # 解析YAML配置

    remote_url = config.get("remote_url", None)  # 获取远程URL配置 # 获取远程URL
    if remote_url is None:  # 检查远程URL是否为空 # 检查remote_url是否为空
        AssertionError("remote_url is None")  # 远程URL为空时断言错误 # remote_url为空时触发断言
    endpoint = remote_url[len("eic://") :]  # 去掉协议前缀，提取端点地址 # 从URL中提取端点地址
    eic_instance_id = config.get("eic_instance_id", None)  # 获取EIC实例ID # 获取EIC实例ID
    eic_log_dir = config.get("eic_log_dir", None)  # 获取EIC日志目录 # 获取日志目录
    eic_log_level = config.get("eic_log_level", 2)  # 获取EIC日志级别，默认为2 # 获取日志级别
    eic_trans_type = config.get("eic_trans_type", 3)  # 获取EIC传输类型，默认为3 # 获取传输类型
    eic_flag_file = config.get("eic_flag_file", None)  # 获取EIC标志文件路径 # 获取标志文件路径

    if not os.path.exists(eic_log_dir):  # 检查日志目录是否存在 # 检查日志目录是否存在
        os.makedirs(eic_log_dir, exist_ok=True)  # 不存在则创建日志目录 # 创建日志目录
    eic_client = eic.Client()  # 创建EIC客户端实例 # 创建EIC客户端
    init_option = eic.InitOption()  # 创建EIC初始化选项 # 创建初始化选项
    init_option.log_dir = eic_log_dir  # 设置日志目录 # 设置日志目录
    init_option.log_level = eic.LogLevel(eic_log_level)  # 设置日志级别 # 设置日志级别
    init_option.transport_type = eic.TransportType(eic_trans_type)  # 设置传输类型 # 设置传输类型
    init_option.flag_file = eic_flag_file  # 设置标志文件 # 设置标志文件
    ret = eic_client.init(eic_instance_id, endpoint, init_option)  # 初始化EIC客户端 # 初始化客户端
    if ret != 0:  # 检查初始化返回值 # 检查初始化是否成功
        raise RuntimeError(f"EIC Client init failed with error code: {ret}")  # 初始化失败抛出运行时异常 # 初始化失败时抛出异常
    return eic_client  # 返回初始化好的EIC客户端 # 返回EIC客户端


def test_set(eic_client):  # 测试EIC客户端的set（写入）功能
    test_key = ["test_key_" + str(i) for i in range(16)]  # 生成16个测试键名 # 生成16个测试键
    tensors = [  # 创建16个测试张量 # 创建测试张量列表
        torch.ones([12, 6, 1, 512], dtype=torch.bfloat16, device="cpu")  # 创建全1的bfloat16张量 # 创建全1的bfloat16张量
        for _ in range(16)  # 循环16次 # 循环16次
    ]
    data_keys = eic.StringVector()  # 创建EIC字符串向量用于存放键 # 创建键的字符串向量
    data_vals = eic.IOBuffers()  # 创建EIC IO缓冲区用于存放值 # 创建IO缓冲区
    for i in range(16):  # 遍历16个测试项 # 遍历16次
        data_keys.append(test_key[i])  # 添加键到字符串向量 # 添加键
        data_vals.append(  # 添加值到IO缓冲区 # 添加值
            tensors[i].data_ptr(), tensors[i].numel() * tensors[i].element_size(), False  # 张量数据指针、字节大小、是否复制 # 张量指针、字节数、不复制
        )
    set_opt = eic.SetOption()  # 创建EIC set操作选项 # 创建set选项
    set_opt.ttl_second = 3  # 设置TTL为3秒 # 设置过期时间为3秒
    status_code, set_outcome = eic_client.mset(data_keys, data_vals, set_opt)  # 批量set操作 # 执行批量写入
    assert (  # 断言操作结果 # 断言操作成功
        status_code == eic.StatusCode.SUCCESS  # 检查状态码是否为成功 # 检查状态码是否为成功
    ), f"Set failed with status code: {status_code}"  # 失败时输出状态码 # 失败时的错误信息


def test_get(eic_client):  # 测试EIC客户端的get（读取）功能
    test_key = ["test_key_" + str(i) for i in range(16)]  # 生成16个测试键名 # 生成16个测试键
    tensors = [  # 创建16个零张量用于接收读取数据 # 创建零张量列表
        torch.zeros([12, 6, 1, 512], dtype=torch.bfloat16, device="cpu")  # 创建全0的bfloat16张量 # 创建全0的bfloat16张量
        for _ in range(16)  # 循环16次 # 循环16次
    ]
    data_keys = eic.StringVector()  # 创建EIC字符串向量用于存放键 # 创建键的字符串向量
    data_vals = eic.IOBuffers()  # 创建EIC IO缓冲区用于存放值 # 创建IO缓冲区
    for i in range(16):  # 遍历16个测试项 # 遍历16次
        data_keys.append(test_key[i])  # 添加键到字符串向量 # 添加键
        data_vals.append(  # 添加值缓冲区到IO缓冲区 # 添加值缓冲区
            tensors[i].data_ptr(), tensors[i].numel() * tensors[i].element_size(), False  # 张量数据指针、字节大小、是否复制 # 张量指针、字节数、不复制
        )
    get_opt = eic.GetOption()  # 创建EIC get操作选项 # 创建get选项
    status_code, data_vals, get_outcome = eic_client.mget(data_keys, get_opt, data_vals)  # 批量get操作 # 执行批量读取
    assert (  # 断言操作结果 # 断言操作成功
        status_code == eic.StatusCode.SUCCESS  # 检查状态码是否为成功 # 检查状态码是否为成功
    ), f"Get failed with status code: {status_code}"  # 失败时输出状态码 # 失败时的错误信息


def test_exists(eic_client):  # 测试EIC客户端的exists（存在性检查）功能
    test_key = ["test_key_" + str(i) for i in range(16)]  # 生成16个测试键名 # 生成16个测试键
    data_keys = eic.StringVector()  # 创建EIC字符串向量用于存放键 # 创建键的字符串向量
    for key in test_key:  # 遍历所有测试键 # 遍历测试键
        data_keys.append(key)  # 添加键到字符串向量 # 添加键
    exists_opt = eic.ExistOption()  # 创建EIC exist操作选项 # 创建exist选项
    status_code, exists_outcome = eic_client.mexist(data_keys, exists_opt)  # 批量exist操作 # 执行批量存在性检查
    assert (  # 断言操作结果 # 断言操作成功
        status_code == eic.StatusCode.SUCCESS  # 检查状态码是否为成功 # 检查状态码是否为成功
    ), f"Exists failed with status code: {status_code}"  # 失败时输出状态码 # 失败时的错误信息


def main():  # 主函数，执行所有测试
    eic_client = init_eic_client()  # 初始化EIC客户端 # 初始化EIC客户端
    test_set(eic_client)  # 执行set测试 # 执行写入测试
    test_exists(eic_client)  # 执行exists测试 # 执行存在性检查测试
    test_get(eic_client)  # 执行get测试 # 执行读取测试


if __name__ == "__main__":  # 当作为主脚本运行时 # 主脚本入口
    main()  # 调用主函数 # 执行主函数
