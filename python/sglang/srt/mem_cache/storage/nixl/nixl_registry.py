# NIXL内存注册辅助模块
# 本模块提供NIXL存储端的内存注册功能，以上下文管理器方式暴露。
# NixlRegistry实例封装了agent、内存类型和文件管理器，
# 其storage()方法是上下文管理器，在进入时执行注册和构建描述符序列，
# 退出时执行注销和关闭文件描述符的清理操作。

"""NIXL memory-registration helpers, exposed as context managers.
NIXL内存注册辅助工具，以上下文管理器方式暴露。

A ``NixlRegistry`` instance bundles the agent, the memory type, and
(optionally) the file manager.  Its ``storage(...)`` method is a context
manager that performs the entire register-and-build-descs sequence for
the storage side of a transfer on entry, yields the ``xfer_descs`` (or
None on failure), and unwinds ``agent.deregister_memory`` plus any
``os.close(fd)`` on exit.
一个``NixlRegistry``实例封装了agent、内存类型和（可选的）文件管理器。
其``storage(...)``方法是上下文管理器，在进入时为传输的存储端执行整个
注册和构建描述符序列，产出``xfer_descs``（失败时为None），在退出时
执行``agent.deregister_memory``和任何``os.close(fd)``的清理操作。

The host side is pre-registered up front by ``HiCacheNixl`` and is not
touched per transfer.
主机端由``HiCacheNixl``预先注册，每次传输不会触及。
"""

import logging  # 导入logging模块，用于日志记录
import threading  # 导入threading模块，用于线程同步
from contextlib import contextmanager  # 从contextlib导入上下文管理器装饰器
from typing import List, Optional  # 从typing模块导入类型注解

from .nixl_utils import NixlFileManager  # 从当前包导入NIXL文件管理器

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def _buffer_sizes(buffers) -> Optional[List[int]]:  # 计算缓冲区大小列表 # 从(addr, len)元组输入中提取每个缓冲区的字节大小
    """Per-buffer byte sizes for ``(addr, len)`` tuple inputs."""  # 用于``(addr, len)``元组输入的每缓冲区字节大小。
    if not buffers or not isinstance(buffers[0], tuple):  # 如果缓冲区为空或第一个元素不是元组
        return None  # 返回None
    return [b[1] for b in buffers]  # 提取每个元组的第二个元素（大小）


class NixlRegistry:  # NIXL注册表类，封装agent、内存类型和文件管理器
    """Owns the (agent, mem_type, file_manager) triple and provides a
    context manager for the storage side of a transfer.
    拥有(agent, mem_type, file_manager)三元组，并为传输的存储端提供上下文管理器。

    A single instance is created once per HiCacheNixl in __init__ and
    reused for every transfer.
    每个HiCacheNixl在__init__中创建一个实例，并在每次传输中复用。
    """

    def __init__(  # 初始化方法
        self,
        agent,  # NIXL代理实例
        mem_type: str,  # 内存类型（FILE或OBJ）
        file_manager: Optional[NixlFileManager] = None,  # 文件管理器，默认None
    ):
        self.agent = agent  # 保存NIXL代理
        self.mem_type = mem_type  # 保存内存类型
        self.file_manager = file_manager  # 保存文件管理器
        # OBJ devIds key a process-wide map in the NIXL OBJ plugin
        # (devIdToObjKey_) that is not protected by a lock, so concurrent
        # OBJ registrations must use disjoint devId ranges. Allocate them
        # from a single monotonic counter.
        # OBJ devIds在NIXL OBJ插件中键控一个进程范围的映射
        # （devIdToObjKey_），该映射没有锁保护，因此并发的
        # OBJ注册必须使用不相交的devId范围。从单一递增计数器分配它们。
        self._obj_devid_lock = threading.Lock()  # OBJ devId分配锁
        self._obj_devid_next = 1  # 下一个可用的OBJ devId

    @contextmanager  # 上下文管理器装饰器
    def _open_files(self, paths: List[str], create: bool):  # 打开文件并管理生命周期 # 打开文件描述符，退出时自动关闭
        """Open fds for ``paths``; close all of them on exit.
        为``paths``打开文件描述符；退出时关闭所有文件描述符。

        Yields the list of fds, or None if any open fails (already-opened
        fds are closed before returning by the same ``finally``).
        产出文件描述符列表，如果任何文件打开失败则产出None
        （已打开的文件描述符在返回前由同一个``finally``关闭）。
        """
        fds: List[int] = []  # 文件描述符列表
        try:  # 尝试打开所有文件
            for path in paths:  # 遍历每个路径
                fd = self.file_manager.open_file(path, create=create)  # 打开文件
                if fd is None:  # 如果打开失败
                    yield None  # 产出None
                    return  # 返回
                fds.append(fd)  # 将文件描述符添加到列表
            yield fds  # 产出文件描述符列表
        finally:  # 最终关闭所有文件
            for fd in fds:  # 遍历所有文件描述符
                self.file_manager.close_file(fd)  # 关闭文件描述符

    @contextmanager  # 上下文管理器装饰器
    def _registered(self, items: List[tuple], mem_type: str):  # 注册内存并在退出时注销 # 注册NIXL内存区域，退出时自动注销
        """Register ``items`` with NIXL; deregister on exit.
        向NIXL注册``items``；退出时注销。

        Yields the registration handle, or None if registration fails.
        产出注册句柄，如果注册失败则产出None。
        """
        reg = None  # 初始化注册句柄为None
        if items:  # 如果有需要注册的项
            reg_descs = self.agent.get_reg_descs(items, mem_type)  # 获取注册描述符
            if reg_descs is not None:  # 如果获取描述符成功
                try:  # 尝试注册内存
                    reg = self.agent.register_memory(reg_descs)  # 注册内存
                except Exception as e:  # 捕获注册异常
                    logger.error(f"Failed to register memory of type {mem_type}: {e}")  # 记录注册失败
        try:  # 尝试产出注册句柄
            yield reg  # 产出注册句柄
        finally:  # 最终注销内存
            if reg is not None:  # 如果注册句柄不为空
                try:  # 尝试注销
                    self.agent.deregister_memory(reg)  # 注销内存
                except Exception as e:  # 捕获注销异常
                    logger.debug("deregister_memory skipped: %s", e)  # 记录注销跳过

    @contextmanager  # 上下文管理器装饰器
    def storage(self, buffers, keys, direction):  # 存储端上下文管理器 # 打开并注册存储端，退出时注销并关闭文件描述符
        """Open + register the storage side; deregister and close fds on exit.
        打开并注册存储端；退出时注销并关闭文件描述符。

        Yields the storage xfer_descs, or None on failure.  For the FILE
        backend, files are created (O_CREAT) when ``direction == "WRITE"``.
        产出存储端的xfer_descs，失败时产出None。对于FILE后端，
        当``direction == "WRITE"``时创建文件（O_CREAT）。
        """
        sizes = _buffer_sizes(buffers)  # 获取缓冲区大小列表
        if sizes is None:  # 如果获取大小失败
            yield None  # 产出None
            return  # 返回

        if self.mem_type == "FILE":  # 如果内存类型为FILE
            with self._open_files(keys, create=(direction == "WRITE")) as fds:  # 打开文件
                if fds is None:  # 如果打开文件失败
                    yield None  # 产出None
                    return  # 返回
                tuples = [(0, sizes[i], fds[i], keys[i]) for i in range(len(keys))]  # 构建注册元组
                with self._registered(tuples, "FILE") as reg:  # 注册FILE内存
                    if reg is None:  # 如果注册失败
                        yield None  # 产出None
                        return  # 返回
                    yield self.agent.get_xfer_descs(  # 产出传输描述符
                        [(0, sizes[i], fds[i]) for i in range(len(fds))], "FILE"  # 构建FILE传输描述符
                    )
        else:  # OBJ  # 内存类型为OBJ
            # Reg tuple: (addr=0, size, devId, metaInfo=key).
            # Xfer tuple: (addr=0, size, devId). devId links each xfer desc
            # back to its registered object's metaInfo, so devIds must be
            # unique within the list AND globally unique across concurrent
            # storage() calls (the OBJ plugin's devIdToObjKey_ map is shared
            # and unlocked). NIXL's pybind layer requires position 3 to be
            # int, hence the key goes in metaInfo (position 4).
            # 注册元组：(addr=0, size, devId, metaInfo=key)。
            # 传输元组：(addr=0, size, devId)。devId将每个传输描述符
            # 链接回其注册对象的metaInfo，因此devIds在列表内必须唯一，
            # 且在并发storage()调用中全局唯一（OBJ插件的devIdToObjKey_映射
            # 是共享且无锁的）。NIXL的pybind层要求位置3为int，
            # 因此键放在metaInfo（位置4）中。
            n = len(keys)  # 键数量
            with self._obj_devid_lock:  # 获取devId分配锁
                base = self._obj_devid_next  # 获取基础devId
                self._obj_devid_next += n  # 递增devId计数器
            dev_ids = list(range(base, base + n))  # 生成devId列表
            tuples = [(0, sizes[i], dev_ids[i], keys[i]) for i in range(n)]  # 构建注册元组
            with self._registered(tuples, "OBJ") as reg:  # 注册OBJ内存
                if reg is None:  # 如果注册失败
                    yield None  # 产出None
                    return  # 返回
                yield self.agent.get_xfer_descs(  # 产出传输描述符
                    [(0, sizes[i], dev_ids[i]) for i in range(n)],  # 构建OBJ传输描述符
                    self.mem_type,  # 内存类型
                )
