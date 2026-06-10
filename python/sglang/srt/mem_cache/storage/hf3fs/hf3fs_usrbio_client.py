# HF3FS UsrBio客户端实现模块
# 本文件实现了基于HF3FS用户态Bio（usrbio）的存储客户端Hf3fsUsrBioClient
# 利用共享内存、io_uring和HF3FS FUSE接口实现高性能的批量读写操作
# 包含读写锁同步装饰器，确保并发安全

import datetime  # 导入日期时间模块 # 日期时间库
import logging  # 导入日志模块 # 日志库
import multiprocessing  # 导入多进程模块 # 多进程库
import os  # 导入操作系统接口模块 # 操作系统接口库
import threading  # 导入线程模块 # 线程库
from functools import wraps  # 导入函数装饰器工具 # 装饰器工具
from pathlib import Path  # 导入路径工具 # 路径工具
from typing import List  # 导入列表类型注解 # 列表类型注解

import torch  # 导入PyTorch张量库 # PyTorch张量库
from torch.utils.cpp_extension import load  # 导入C++扩展加载工具 # C++扩展加载

from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient  # 导入HF3FS客户端抽象基类 # 导入HF3FS客户端基类

root = Path(__file__).parent.resolve()  # 获取当前文件所在目录的绝对路径 # 获取当前目录
hf3fs_utils = load(name="hf3fs_utils", sources=[f"{root}/hf3fs_utils.cpp"])  # 编译加载hf3fs_utils C++扩展 # 加载C++扩展

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器 # 创建日志记录器

HF3FS_AVAILABLE = True  # HF3FS是否可用的标志 # HF3FS可用标志
try:  # 尝试导入HF3FS FUSE IO模块 # 尝试导入
    from hf3fs_fuse.io import (  # 从hf3fs_fuse.io导入IO相关函数 # 导入HF3FS IO函数
        deregister_fd,  # 注销文件描述符 # 注销文件描述符
        extract_mount_point,  # 提取挂载点 # 提取挂载点
        make_ioring,  # 创建io_uring # 创建io_uring
        make_iovec,  # 创建IO向量 # 创建IO向量
        register_fd,  # 注册文件描述符 # 注册文件描述符
    )
except ImportError:  # 导入失败时 # 导入失败
    HF3FS_AVAILABLE = False  # 标记HF3FS不可用 # 标记不可用


def rsynchronized():  # 读同步装饰器，用读锁保护方法
    def _decorator(func):  # 内部装饰器函数 # 内部装饰器
        @wraps(func)  # 保留原函数元信息 # 保留元信息
        def wrapper(self, *args, **kwargs):  # 包装函数 # 包装函数
            with self.rlock:  # 获取读锁 # 获取读锁
                return func(self, *args, **kwargs)  # 执行原函数并返回结果 # 执行原函数

        return wrapper  # 返回包装函数 # 返回包装函数

    return _decorator  # 返回装饰器 # 返回装饰器


def wsynchronized():  # 写同步装饰器，用写锁保护方法
    def _decorator(func):  # 内部装饰器函数 # 内部装饰器
        @wraps(func)  # 保留原函数元信息 # 保留元信息
        def wrapper(self, *args, **kwargs):  # 包装函数 # 包装函数
            with self.wlock:  # 获取写锁 # 获取写锁
                return func(self, *args, **kwargs)  # 执行原函数并返回结果 # 执行原函数

        return wrapper  # 返回包装函数 # 返回包装函数

    return _decorator  # 返回装饰器 # 返回装饰器


class Hf3fsUsrBioClient(Hf3fsClient):  # 基于usrbio的HF3FS客户端实现类 # 基于usrbio的HF3FS客户端
    """HF3FS client implementation using usrbio."""  # 使用usrbio的HF3FS客户端实现 # 使用usrbio的HF3FS客户端实现

    def __init__(  # 初始化UsrBio客户端
        self,
        path: str,  # 存储文件路径 # 文件路径
        size: int,  # 存储文件总大小 # 存储总大小
        bytes_per_page: int,  # 每页字节数 # 每页字节数
        entries: int,  # 批量操作条目数 # 批量操作条目数
        client_timeout: int,  # 客户端超时时间（秒） # 客户端超时时间
    ):
        if not HF3FS_AVAILABLE:  # 检查HF3FS是否可用 # 检查HF3FS是否可用
            raise ImportError(  # 不可用则抛出导入错误 # 抛出导入错误
                "hf3fs_fuse.io is not available. Please install the hf3fs_fuse package."  # 提示安装hf3fs_fuse包 # 提示安装
            )

        self.path = path  # 存储文件路径 # 文件路径
        self.size = size  # 存储文件总大小 # 存储总大小
        self.bytes_per_page = bytes_per_page  # 每页字节数 # 每页字节数
        self.entries = entries  # 批量操作条目数 # 批量操作条目数
        self.client_timeout = client_timeout  # 客户端超时时间 # 客户端超时时间

        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)  # 以读写模式打开或创建文件 # 打开或创建文件
        os.ftruncate(self.file, size)  # 将文件截断到指定大小 # 截断文件
        register_fd(self.file)  # 注册文件描述符到HF3FS # 注册文件描述符

        self.hf3fs_mount_point = extract_mount_point(path)  # 提取HF3FS挂载点 # 提取挂载点
        self.bs = self.bytes_per_page  # 块大小等于每页字节数 # 块大小
        self.shm_r = multiprocessing.shared_memory.SharedMemory(  # 创建读共享内存 # 创建读共享内存
            size=self.bs * self.entries, create=True  # 大小为块大小乘以条目数 # 共享内存大小
        )
        self.shm_w = multiprocessing.shared_memory.SharedMemory(  # 创建写共享内存 # 创建写共享内存
            size=self.bs * self.entries, create=True  # 大小为块大小乘以条目数 # 共享内存大小
        )

        self.shm_r_tensor = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)  # 将读共享内存缓冲区转为uint8张量 # 读共享内存张量
        self.shm_w_tensor = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)  # 将写共享内存缓冲区转为uint8张量 # 写共享内存张量

        self.numa = -1  # NUMA节点号，-1表示不指定 # NUMA节点
        self.ior_r = make_ioring(  # 创建读io_uring # 创建读io_uring
            self.hf3fs_mount_point,  # 挂载点 # 挂载点
            self.entries,  # 条目数 # 条目数
            for_read=True,  # 用于读取 # 用于读取
            timeout=1,  # 超时1秒 # 超时时间
            numa=self.numa,  # NUMA节点 # NUMA节点
        )
        self.ior_w = make_ioring(  # 创建写io_uring # 创建写io_uring
            self.hf3fs_mount_point,  # 挂载点 # 挂载点
            self.entries,  # 条目数 # 条目数
            for_read=False,  # 用于写入 # 用于写入
            timeout=1,  # 超时1秒 # 超时时间
            numa=self.numa,  # NUMA节点 # NUMA节点
        )
        self.iov_r = make_iovec(self.shm_r, self.hf3fs_mount_point)  # 创建读IO向量 # 创建读IO向量
        self.iov_w = make_iovec(self.shm_w, self.hf3fs_mount_point)  # 创建写IO向量 # 创建写IO向量
        self.shm_r.unlink()  # 取消读共享内存的链接（标记为删除） # 取消读共享内存链接
        self.shm_w.unlink()  # 取消写共享内存的链接（标记为删除） # 取消写共享内存链接

        self.rlock = threading.RLock()  # 创建可重入读锁 # 创建读锁
        self.wlock = threading.RLock()  # 创建可重入写锁 # 创建写锁

    @rsynchronized()  # 读同步装饰器 # 读同步
    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:  # 批量读取数据
        self.check(offsets, tensors)  # 校验参数 # 校验参数
        results = [0] * len(offsets)  # 初始化结果列表，默认为0 # 初始化结果
        # prepare  # 准备阶段 # 准备阶段
        current = 0  # 当前共享内存偏移位置 # 当前偏移
        for offset, tensor in zip(offsets, tensors):  # 遍历偏移量和张量 # 遍历
            size = tensor.numel() * tensor.itemsize  # 计算字节数 # 计算字节数
            try:  # 尝试准备读取 # 尝试准备
                self.ior_r.prepare(  # 准备io_uring读取操作 # 准备读取
                    self.iov_r[current : current + size], True, self.file, offset  # 共享内存切片、读取标志、文件描述符、文件偏移 # IO向量、标志、文件、偏移
                )
                current += size  # 更新当前偏移 # 更新偏移
            except Exception as e:  # 捕获异常 # 捕获异常
                logger.error(f"Error preparing batch read: {e}")  # 记录准备阶段错误 # 记录准备错误
                return results  # 返回默认结果 # 返回结果
        # submit  # 提交阶段 # 提交阶段
        ionum = len(offsets)  # IO操作数量 # IO数量
        try:  # 尝试提交和等待 # 尝试提交
            resv = self.ior_r.submit().wait(  # 提交io_uring并等待完成 # 提交并等待
                min_results=ionum,  # 最少等待结果数 # 最少结果数
                timeout=datetime.timedelta(seconds=self.client_timeout),  # 超时时间 # 超时时间
            )
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"Error submitting batch read: {e}")  # 记录提交阶段错误 # 记录提交错误
            return results  # 返回默认结果 # 返回结果
        # results  # 结果处理阶段 # 结果处理阶段
        try:  # 尝试从共享内存读取数据 # 尝试读取
            hf3fs_utils.read_shm(self.shm_r_tensor, tensors)  # 从共享内存张量拷贝数据到目标张量 # 从共享内存读取
            results = [res.result for res in resv]  # 获取每个IO操作的结果 # 获取IO结果
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"[Hf3fsUsrBioClient] read_shm failed: {e}", exc_info=True)  # 记录共享内存读取错误 # 记录读取错误
            return results  # 返回结果 # 返回结果

        return results  # 返回读取结果列表 # 返回结果

    @wsynchronized()  # 写同步装饰器 # 写同步
    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:  # 批量写入数据
        self.check(offsets, tensors)  # 校验参数 # 校验参数
        results = [0] * len(offsets)  # 初始化结果列表，默认为0 # 初始化结果
        # prepare  # 准备阶段 # 准备阶段
        hf3fs_utils.write_shm(tensors, self.shm_w_tensor)  # 将张量数据拷贝到写共享内存 # 写入共享内存
        current = 0  # 当前共享内存偏移位置 # 当前偏移
        for offset, tensor in zip(offsets, tensors):  # 遍历偏移量和张量 # 遍历
            size = tensor.numel() * tensor.itemsize  # 计算字节数 # 计算字节数
            try:  # 尝试准备写入 # 尝试准备
                self.ior_w.prepare(  # 准备io_uring写入操作 # 准备写入
                    self.iov_w[current : current + size], False, self.file, offset  # 共享内存切片、写入标志、文件描述符、文件偏移 # IO向量、标志、文件、偏移
                )
                current += size  # 更新当前偏移 # 更新偏移
            except Exception as e:  # 捕获异常 # 捕获异常
                logger.error(f"Error preparing batch write: {e}")  # 记录准备阶段错误 # 记录准备错误
                return results  # 返回默认结果 # 返回结果

        # submit  # 提交阶段 # 提交阶段
        ionum = len(offsets)  # IO操作数量 # IO数量
        try:  # 尝试提交和等待 # 尝试提交
            resv = self.ior_w.submit().wait(  # 提交io_uring并等待完成 # 提交并等待
                min_results=ionum,  # 最少等待结果数 # 最少结果数
                timeout=datetime.timedelta(seconds=self.client_timeout),  # 超时时间 # 超时时间
            )
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"Error submitting batch write: {e}")  # 记录提交阶段错误 # 记录提交错误
            return results  # 返回默认结果 # 返回结果

        # results  # 结果处理阶段 # 结果处理阶段
        results = [res.result for res in resv]  # 获取每个IO操作的结果 # 获取IO结果

        return results  # 返回写入结果列表 # 返回结果

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:  # 校验批量操作参数
        sizes = [t.numel() * t.itemsize for t in tensors]  # 计算每个张量的字节数 # 计算字节数列表
        if any(  # 检查是否存在任一不合法条件 # 检查合法性
            [
                len(offsets) > self.entries,  # 偏移量数量超过条目数 # 偏移量超过条目数
                len(offsets) != len(sizes),  # 偏移量和大小数量不匹配 # 数量不匹配
                all(  # 检查是否所有偏移都越界 # 检查偏移越界
                    [
                        offset < 0 or offset + size > self.size  # 偏移为负或超出存储大小 # 偏移越界
                        for offset, size in zip(offsets, sizes)
                    ]
                ),
                all([size > self.bytes_per_page for size in sizes]),  # 检查是否所有大小都超过页大小 # 大小超过页大小
            ]
        ):
            self.close()  # 关闭客户端 # 关闭客户端
            raise ValueError(f"Hf3fsClient.check: {offsets=}, {sizes=}")  # 抛出参数校验错误 # 抛出校验错误

    def get_size(self) -> int:  # 获取存储总大小
        return self.size  # 返回存储总大小 # 返回大小

    def close(self) -> None:  # 关闭客户端并清理资源
        deregister_fd(self.file)  # 注销文件描述符 # 注销文件描述符
        os.close(self.file)  # 关闭文件 # 关闭文件
        del self.ior_r  # 删除读io_uring # 删除读io_uring
        del self.ior_w  # 删除写io_uring # 删除写io_uring
        del self.iov_r  # 删除读IO向量 # 删除读IO向量
        del self.iov_w  # 删除写IO向量 # 删除写IO向量
        self.shm_r.close()  # 关闭读共享内存 # 关闭读共享内存
        self.shm_w.close()  # 关闭写共享内存 # 关闭写共享内存

    def flush(self) -> None:  # 将数据刷新到磁盘
        os.fsync(self.file)  # 将文件数据同步到磁盘 # 同步到磁盘
