# 内存映射分配器模块 - 通过匿名mmap分配主机张量，支持大页(hugepage)和普通页
# 本文件实现了alloc_mmap函数，用于分配主机端张量，支持2MB/1GB大页和普通页，
# 并在张量被释放时自动调用munmap释放映射

import ctypes # 导入C类型绑定模块
import ctypes.util # 导入C类型工具模块
import logging # 导入日志模块
import math # 导入数学模块
import mmap # 导入内存映射模块
import os # 导入操作系统模块
import weakref # 导入弱引用模块

import torch # 导入PyTorch

from sglang.srt.environ import envs # 导入环境变量

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

# Load libc once at module level so munmap is callable safely at GC/shutdown time.
# Resolve the SONAME via find_library so the allocator also works on systems
# whose libc is not named "libc.so.6" (e.g. musl / Alpine).
# 在模块级别加载libc一次，以便在GC/关闭时安全调用munmap。
# 通过find_library解析SONAME，使分配器也能在不名为"libc.so.6"的系统上工作（如musl/Alpine）。
try: # 尝试加载libc
    _libc_name = ctypes.util.find_library("c") or "libc.so.6" # 查找C库路径，默认libc.so.6
    _libc = ctypes.CDLL(_libc_name, use_errno=True) # 加载C库
    _libc.mmap.restype = ctypes.c_void_p # 设置mmap返回类型为void指针
    _libc.mmap.argtypes = [ # 设置mmap参数类型
        ctypes.c_void_p, # 地址（NULL表示自动选择）
        ctypes.c_size_t, # 映射长度
        ctypes.c_int, # 保护标志
        ctypes.c_int, # 映射标志
        ctypes.c_int, # 文件描述符
        ctypes.c_long, # 偏移量
    ]
    _libc.munmap.restype = ctypes.c_int # 设置munmap返回类型为int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t] # 设置munmap参数类型
except OSError: # 如果加载失败
    _libc = None # 设为None


# MAP_POPULATE is in Python's mmap module only since 3.11.
# MAP_POPULATE仅在Python 3.11及以上版本的mmap模块中可用。
_MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0x08000) # 获取MAP_POPULATE标志，默认0x08000
# MAP_HUGETLB and MAP_HUGE_* are Linux-specific and not in Python's mmap module.
# MAP_HUGETLB和MAP_HUGE_*是Linux特有的，不在Python的mmap模块中。
_MAP_HUGETLB = 0x40000 # 大页标志
_MAP_HUGE_2MB = 21 << 26  # 0x1400000 # 2MB大页标志
_MAP_HUGE_1GB = 30 << 26  # 0x78000000 # 1GB大页标志
_MAP_FAILED = ctypes.c_void_p(-1).value # mmap失败返回值


def _alloc_hugepage(n_bytes: int, alloc_bytes: int, extra_flags: int) -> ctypes.Array: # 通过libc调用mmap分配大页内存
    """Call mmap via libc with hugepage flags and return an owning ctypes array.
    通过libc调用带大页标志的mmap，返回一个拥有所有权的ctypes数组。

    munmap fires automatically via weakref.finalize when the array is
    garbage-collected (i.e. when the tensor that wraps it is freed).
    当数组被垃圾回收时（即包装它的张量被释放时），munmap通过weakref.finalize自动触发。
    """
    ptr = _libc.mmap( # 调用libc的mmap
        None, # 地址为NULL（自动选择）
        alloc_bytes, # 映射字节数
        mmap.PROT_READ | mmap.PROT_WRITE, # 读写保护标志
        mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE | extra_flags, # 共享+匿名+预填充+额外标志
        -1, # 文件描述符为-1（匿名映射）
        0, # 偏移量为0
    )
    if ptr is None or ptr == _MAP_FAILED: # 如果mmap失败
        errno = ctypes.get_errno() # 获取错误码
        raise OSError(errno, os.strerror(errno)) # 抛出OSError
    array = (ctypes.c_uint8 * n_bytes).from_address(ptr) # 从指针创建ctypes数组
    weakref.finalize(array, _libc.munmap, ctypes.c_void_p(ptr), alloc_bytes) # 注册弱引用终结器，自动调用munmap
    return array # 返回ctypes数组


def alloc_mmap(dims: tuple, dtype: torch.dtype) -> torch.Tensor: # 通过匿名mmap分配主机张量
    """Allocate a host tensor via anonymous mmap. Set SGLANG_HUGEPAGE_SIZE=2MB or 1GB for hugepages.
    通过匿名mmap分配主机张量。设置SGLANG_HUGEPAGE_SIZE=2MB或1GB以使用大页。

    MAP_SHARED + MAP_POPULATE are both required so cudaHostRegister pins real,
    pre-faulted physical pages (otherwise pinning can race with COW or page
    faults and the device ends up reading stale data).
    MAP_SHARED + MAP_POPULATE都是必需的，这样cudaHostRegister可以锁定真实的、
    预填充的物理页面（否则锁定可能与COW或页面错误竞争，导致设备读取过期数据）。

    The tensor owns the mapping; munmap fires when the tensor is freed.
    张量拥有映射；当张量被释放时munmap自动触发。
    """
    # Re-read per call (not cached) so that envs.SGLANG_HUGEPAGE_SIZE.override()
    # works correctly in tests.
    # 每次调用重新读取（不缓存），以便envs.SGLANG_HUGEPAGE_SIZE.override()在测试中正常工作。
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper() # 获取大页大小配置
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size() # 计算需要的字节数

    if hugepage_size == "": # 如果未指定大页大小
        page_size, extra_flags = mmap.PAGESIZE, 0 # 使用普通页面大小，无额外标志
    elif hugepage_size == "2MB": # 如果指定2MB大页
        page_size, extra_flags = 2 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_2MB # 2MB大页+大页标志
    elif hugepage_size == "1GB": # 如果指定1GB大页
        page_size, extra_flags = 1024 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_1GB # 1GB大页+大页标志
    else: # 否则无法识别的大页大小
        logger.warning( # 记录警告
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'. " # 无法识别的SGLANG_HUGEPAGE_SIZE；期望'2MB'或'1GB'
            "Falling back to plain page-size mmap.", # 回退到普通页面大小的mmap
            envs.SGLANG_HUGEPAGE_SIZE.get(),
        )
        page_size, extra_flags = mmap.PAGESIZE, 0 # 回退到普通页面

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size # 按页面大小对齐分配字节数

    if extra_flags: # 如果有大页标志
        if _libc is None: # 如果libc未加载
            logger.error( # 记录错误
                "Hugepage mmap requested but libc.so.6 could not be loaded; " # 请求大页mmap但无法加载libc.so.6
                "falling back to plain mmap. SGLANG_HUGEPAGE_SIZE=%s will be ignored.", # 回退到普通mmap，SGLANG_HUGEPAGE_SIZE将被忽略
                hugepage_size,
            )
        else: # 否则libc已加载
            try: # 尝试分配大页
                array = _alloc_hugepage(n_bytes, alloc_bytes, extra_flags) # 调用大页分配
                return torch.frombuffer( # 从ctypes数组创建张量
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims) # 重塑为指定维度
            except OSError as e: # 如果分配失败
                logger.error( # 记录错误
                    "Hugepage mmap via libc failed (%s); falling back to plain mmap. " # 通过libc的大页mmap失败；回退到普通mmap
                    "SGLANG_HUGEPAGE_SIZE=%s will be ignored.", # SGLANG_HUGEPAGE_SIZE将被忽略
                    e,
                    hugepage_size,
                )
        alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE # 回退到普通页面对齐

    # Plain mmap path -- used directly when no hugepages requested, or as fallback.
    # 普通mmap路径 -- 直接在没有请求大页时使用，或作为回退。
    # torch.frombuffer keeps a reference to mm inside the tensor storage, so mm
    # stays alive until the tensor is freed and mmap.mmap.__del__ calls munmap.
    # torch.frombuffer在张量存储中保持对mm的引用，因此mm在张量被释放之前一直存活，
    # mmap.mmap.__del__会调用munmap。
    mm = mmap.mmap( # 创建Python mmap对象
        -1, # 文件描述符为-1（匿名映射）
        alloc_bytes, # 映射字节数
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE, # 共享+匿名+预填充
        prot=mmap.PROT_READ | mmap.PROT_WRITE, # 读写保护
    )
    return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims) # 从mmap对象创建张量并重塑维度
