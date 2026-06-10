# HF3FS工具的共享内存读写测试模块
# 本模块通过pytest测试hf3fs_utils C++扩展对共享内存的写入和读取功能，
# 验证数据在写入后能被正确读回，确保共享内存KV缓存的IO正确性。

import multiprocessing.shared_memory  # 导入共享内存模块 # 导入多进程共享内存模块
import sys  # 导入系统模块 # 导入系统模块
from pathlib import Path  # 导入路径模块 # 导入路径处理模块

import pytest  # 导入pytest测试框架 # 导入pytest测试框架
import torch  # 导入PyTorch深度学习框架 # 导入PyTorch深度学习框架
from torch.utils.cpp_extension import load  # 导入C++扩展加载工具 # 导入C++扩展加载工具
from tqdm import tqdm  # 导入进度条工具 # 导入进度条工具

root = Path(__file__).parent.resolve()  # 获取当前文件所在目录的绝对路径 # 获取当前文件所在目录的绝对路径
hf3fs_utils = load(  # 加载C++扩展模块 # 加载C++扩展模块
    name="hf3fs_utils", sources=[f"{root}/hf3fs_utils.cpp"], verbose=True  # 指定扩展名和源文件路径，启用详细输出 # 指定扩展名和源文件路径，启用详细输出
)


def test_rw_shm():  # 测试共享内存读写功能 # 测试共享内存的读写功能
    numel = 8 << 20  # 每页元素数量，8M个元素 # 每页元素数量，8M个元素
    dtype = torch.bfloat16  # 数据类型为bfloat16 # 数据类型为bfloat16
    page_num = 128  # 页面数量 # 页面数量
    page_bytes = numel * dtype.itemsize  # 每页字节数 # 每页字节数
    shm = multiprocessing.shared_memory.SharedMemory(  # 创建共享内存对象 # 创建共享内存对象
        size=page_num * page_bytes, create=True  # 总大小为页数乘以每页字节数，创建新共享内存 # 总大小为页数乘以每页字节数，创建新共享内存
    )
    tshm = torch.frombuffer(shm.buf, dtype=torch.uint8)  # 将共享内存缓冲区转为torch张量 # 将共享内存缓冲区转为torch张量
    a = [  # 准备输入数据列表 # 准备输入数据列表
        torch.randn(numel, dtype=dtype)  # 生成随机数据 # 生成随机数据
        for _ in tqdm(range(page_num), desc="prepare input")  # 循环page_num次，显示进度条 # 循环page_num次，显示进度条
    ]
    b = [  # 准备输出数据列表 # 准备输出数据列表
        torch.empty(numel, dtype=dtype)  # 创建空张量 # 创建空张量
        for _ in tqdm(range(page_num), desc="prepare output")  # 循环page_num次，显示进度条 # 循环page_num次，显示进度条
    ]
    hf3fs_utils.write_shm(a, tshm)  # 将数据列表a写入共享内存 # 将数据列表a写入共享内存
    hf3fs_utils.read_shm(tshm, b)  # 从共享内存读取数据到列表b # 从共享内存读取数据到列表b
    for _a, _b in tqdm(zip(a, b), desc="assert_close"):  # 逐个比较写入和读取的数据 # 逐个比较写入和读取的数据
        torch.testing.assert_close(_a, _b)  # 断言写入和读取的数据一致 # 断言写入和读取的数据一致

    del tshm  # 删除torch张量引用 # 删除torch张量引用
    shm.close()  # 关闭共享内存 # 关闭共享内存
    shm.unlink()  # 释放共享内存资源 # 释放共享内存资源


if __name__ == "__main__":  # 主入口 # 主入口
    sys.exit(pytest.main([__file__]))  # 以pytest方式运行本文件测试 # 以pytest方式运行本文件测试
