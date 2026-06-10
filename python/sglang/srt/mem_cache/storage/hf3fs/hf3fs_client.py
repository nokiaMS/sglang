# HF3FS客户端抽象接口与Mock实现模块
# 本文件定义了Hf3fsClient抽象基类，提供HF3FS存储的批量读写、校验、关闭和刷新接口
# 同时提供了Hf3fsMockClient模拟实现，用于CI测试环境下不依赖真实HF3FS硬件的场景

import logging  # 导入日志模块 # 日志库
import os  # 导入操作系统接口模块 # 操作系统接口库
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器 # 抽象基类相关
from typing import List  # 导入列表类型注解 # 列表类型注解

import torch  # 导入PyTorch张量库 # PyTorch张量库


class Hf3fsClient(ABC):  # HF3FS客户端抽象基类 # HF3FS客户端抽象接口
    """Abstract interface for HF3FS clients."""  # HF3FS客户端的抽象接口 # HF3FS客户端的抽象接口

    @abstractmethod
    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):  # 初始化HF3FS客户端的抽象方法
        """Initialize the HF3FS client.  # 初始化HF3FS客户端 # 初始化HF3FS客户端

        Args:
            path: File path for storage  # 参数path：存储文件路径 # 存储文件路径
            size: Total size of storage file  # 参数size：存储文件总大小 # 存储文件总大小
            bytes_per_page: Bytes per page  # 参数bytes_per_page：每页字节数 # 每页字节数
            entries: Number of entries for batch operations  # 参数entries：批量操作的条目数 # 批量操作条目数
        """
        pass  # 抽象方法，子类必须实现 # 抽象方法占位

    @abstractmethod
    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:  # 批量读取数据的抽象方法
        """Batch read from storage."""  # 从存储中批量读取 # 从存储中批量读取
        pass  # 抽象方法，子类必须实现 # 抽象方法占位

    @abstractmethod
    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:  # 批量写入数据的抽象方法
        """Batch write to storage."""  # 批量写入到存储 # 批量写入到存储
        pass  # 抽象方法，子类必须实现 # 抽象方法占位

    @abstractmethod
    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:  # 校验批量操作参数的抽象方法
        """Validate batch operation parameters."""  # 校验批量操作参数 # 校验批量操作参数
        pass  # 抽象方法，子类必须实现 # 抽象方法占位

    @abstractmethod
    def get_size(self) -> int:  # 获取存储总大小的抽象方法
        """Get total storage size."""  # 获取存储总大小 # 获取存储总大小
        pass  # 抽象方法，子类必须实现 # 抽象方法占位

    @abstractmethod
    def close(self) -> None:  # 关闭客户端并清理资源的抽象方法
        """Close the client and cleanup resources."""  # 关闭客户端并清理资源 # 关闭客户端并清理资源
        pass  # 抽象方法，子类必须实现 # 抽象方法占位

    @abstractmethod
    def flush(self) -> None:  # 将数据刷新到磁盘的抽象方法
        """Flush data to disk."""  # 将数据刷新到磁盘 # 将数据刷新到磁盘
        pass  # 抽象方法，子类必须实现 # 抽象方法占位


logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器 # 创建日志记录器


class Hf3fsMockClient(Hf3fsClient):  # HF3FS客户端的Mock模拟实现类，用于CI测试 # HF3FS模拟客户端，用于CI测试
    """Mock implementation of Hf3fsClient for CI testing purposes."""  # 用于CI测试的Hf3fsClient模拟实现 # 用于CI测试的模拟实现

    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):  # 初始化Mock客户端
        """Initialize mock HF3FS client."""  # 初始化模拟HF3FS客户端 # 初始化模拟客户端
        self.path = path  # 存储文件路径 # 文件路径
        self.size = size  # 存储文件总大小 # 存储总大小
        self.bytes_per_page = bytes_per_page  # 每页字节数 # 每页字节数
        self.entries = entries  # 批量操作条目数 # 批量操作条目数

        # Create directory if it doesn't exist  # 如果目录不存在则创建 # 如果目录不存在则创建
        os.makedirs(os.path.dirname(self.path), exist_ok=True)  # 创建文件所在目录 # 创建目录

        # Create and initialize the file  # 创建并初始化文件 # 创建并初始化文件
        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)  # 以读写模式打开或创建文件 # 打开或创建文件
        os.ftruncate(self.file, size)  # 将文件截断到指定大小 # 截断文件到指定大小

        logger.info(  # 记录初始化信息 # 记录日志
            f"Hf3fsMockClient initialized: path={path}, size={size}, "  # 打印路径和大小 # 路径和大小信息
            f"bytes_per_page={bytes_per_page}, entries={entries}"  # 打印每页字节数和条目数 # 每页字节数和条目数
        )

    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:  # 批量读取数据
        """Batch read from mock storage."""  # 从模拟存储中批量读取 # 从模拟存储批量读取
        self.check(offsets, tensors)  # 校验参数 # 校验参数

        results = []  # 存储每次读取的结果 # 读取结果列表

        for offset, tensor in zip(offsets, tensors):  # 遍历偏移量和张量 # 遍历偏移量和张量
            size = tensor.numel() * tensor.itemsize  # 计算需要读取的字节数 # 计算字节数

            try:  # 尝试读取操作 # 尝试读取
                os.lseek(self.file, offset, os.SEEK_SET)  # 定位到指定偏移量 # 定位文件偏移
                bytes_read = os.read(self.file, size)  # 读取指定大小的字节 # 读取字节

                if len(bytes_read) == size:  # 检查读取字节数是否完整 # 检查是否完整读取
                    # Convert bytes to tensor and copy to target  # 将字节转换为张量并复制到目标 # 将字节转为张量并复制
                    bytes_tensor = torch.frombuffer(bytes_read, dtype=torch.uint8)  # 将字节数据转换为uint8张量 # 字节转uint8张量
                    typed_tensor = bytes_tensor.view(tensor.dtype).view(tensor.shape)  # 转换为目标数据类型和形状 # 转换类型和形状
                    tensor.copy_(typed_tensor)  # 将数据复制到目标张量 # 复制数据
                    results.append(size)  # 记录成功读取的字节数 # 记录读取字节数
                else:  # 读取不完整 # 读取不完整
                    logger.warning(  # 记录警告日志 # 记录警告
                        f"Short read: expected {size}, got {len(bytes_read)}"  # 期望与实际读取字节数不匹配 # 读取字节数不匹配
                    )
                    results.append(len(bytes_read))  # 记录实际读取的字节数 # 记录实际读取字节数

            except Exception as e:  # 捕获异常 # 捕获异常
                logger.error(f"Error reading from offset {offset}: {e}")  # 记录错误日志 # 记录错误
                results.append(0)  # 读取失败返回0 # 失败返回0

        return results  # 返回读取结果列表 # 返回结果

    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:  # 批量写入数据
        """Batch write to mock storage."""  # 批量写入到模拟存储 # 批量写入到模拟存储
        self.check(offsets, tensors)  # 校验参数 # 校验参数

        results = []  # 存储每次写入的结果 # 写入结果列表

        for offset, tensor in zip(offsets, tensors):  # 遍历偏移量和张量 # 遍历偏移量和张量
            size = tensor.numel() * tensor.itemsize  # 计算需要写入的字节数 # 计算字节数

            try:  # 尝试写入操作 # 尝试写入
                # Convert tensor to bytes and write directly to file  # 将张量转换为字节并直接写入文件 # 将张量转为字节并写入文件
                tensor_bytes = tensor.contiguous().view(torch.uint8).flatten()  # 将张量转为连续的uint8一维视图 # 转为连续uint8视图
                data = tensor_bytes.numpy().tobytes()  # 转换为numpy数组再转为字节 # 转为字节

                os.lseek(self.file, offset, os.SEEK_SET)  # 定位到指定偏移量 # 定位文件偏移
                bytes_written = os.write(self.file, data)  # 写入数据到文件 # 写入数据

                if bytes_written == size:  # 检查写入字节数是否完整 # 检查是否完整写入
                    results.append(size)  # 记录成功写入的字节数 # 记录写入字节数
                else:  # 写入不完整 # 写入不完整
                    logger.warning(f"Short write: expected {size}, got {bytes_written}")  # 记录警告日志 # 记录警告
                    results.append(bytes_written)  # 记录实际写入的字节数 # 记录实际写入字节数

            except Exception as e:  # 捕获异常 # 捕获异常
                logger.error(f"Error writing to offset {offset}: {e}")  # 记录错误日志 # 记录错误
                results.append(0)  # 写入失败返回0 # 失败返回0

        return results  # 返回写入结果列表 # 返回结果

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:  # 校验批量操作参数
        """Validate batch operation parameters."""  # 校验批量操作参数 # 校验批量操作参数
        pass  # Mock实现中不做校验 # Mock实现中不做校验

    def get_size(self) -> int:  # 获取存储总大小
        """Get total storage size."""  # 获取存储总大小 # 获取存储总大小
        return self.size  # 返回存储总大小 # 返回大小

    def close(self) -> None:  # 关闭Mock客户端并清理资源
        """Close the mock client and cleanup resources."""  # 关闭模拟客户端并清理资源 # 关闭模拟客户端并清理资源
        try:  # 尝试关闭操作 # 尝试关闭
            if hasattr(self, "file") and self.file >= 0:  # 检查文件描述符是否存在且有效 # 检查文件描述符有效性
                os.close(self.file)  # 关闭文件 # 关闭文件
                self.file = -1  # Mark as closed  # 标记为已关闭 # 标记为已关闭
            logger.info(f"MockHf3fsClient closed: {self.path}")  # 记录关闭日志 # 记录关闭日志
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"Error closing MockHf3fsClient: {e}")  # 记录关闭错误日志 # 记录关闭错误

    def flush(self) -> None:  # 将数据刷新到磁盘
        """Flush data to disk."""  # 将数据刷新到磁盘 # 将数据刷新到磁盘
        try:  # 尝试刷新操作 # 尝试刷新
            os.fsync(self.file)  # 将文件数据同步到磁盘 # 同步到磁盘
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"Error flushing MockHf3fsClient: {e}")  # 记录刷新错误日志 # 记录刷新错误
