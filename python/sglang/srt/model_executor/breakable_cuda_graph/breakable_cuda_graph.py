# 可中断CUDA图（Breakable CUDA Graph）核心实现模块
# 实现将一段代码捕获为多个torch.cuda.CUDAGraph段的机制，
# 段之间由急切（eager）断点分隔。每个段是真正的CUDAGraph，
# 其析构函数调用共享内存池的releasePool，使池的use_count跟踪存活的段数；
# 只要任一段图存活，池就保持固定，从而让weak_ref_tensor视图在重放间保持有效。

# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Breakable CUDA Graph: capture a region as a sequence of  # 可中断CUDA图：将一个区域捕获为一系列
``torch.cuda.CUDAGraph`` segments separated by eager break points.  # 由急切断点分隔的torch.cuda.CUDAGraph段。

Each segment is a real ``torch.cuda.CUDAGraph``. Its destructor calls  # 每个段是一个真正的torch.cuda.CUDAGraph。其析构函数调用
``releasePool`` on the shared mempool, so the pool's ``use_count`` tracks how  # 共享内存池上的releasePool，因此池的use_count跟踪
many segments are alive; the pool stays pinned as long as any segment graph  # 有多少段存活；只要任一段图存活，池就保持固定。
is alive. This lets ``weak_ref_tensor`` views of intermediate pool-allocated  # 这使得池分配的中间张量的weak_ref_tensor视图
tensors remain valid across replays — we don't need Python-managed bridge  # 在重放间保持有效——我们不需要Python管理的桥接
buffers to keep break-point tensors at stable addresses.  # 缓冲区来保持断点张量在稳定地址。
"""

import logging  # 导入日志模块
import threading  # 导入线程模块
from contextvars import ContextVar  # 导入上下文变量，用于线程安全的上下文传递
from typing import Any, Callable  # 导入类型注解

import torch  # 导入PyTorch

try:  # 尝试导入CUDA绑定
    from cuda.bindings import runtime as rt  # 导入CUDA运行时绑定
except ImportError:  # 如果导入失败
    rt = None  # 设为None，表示不可用

from sglang.srt.model_executor.breakable_cuda_graph.cuda_utils import checkCudaErrors  # 导入CUDA错误检查工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = [  # 模块公开接口列表
    "eager_on_graph",  # 急切图断点装饰器
    "BreakableCUDAGraph",  # 可中断CUDA图容器类
    "BreakableCUDAGraphCapture",  # 可中断CUDA图捕获上下文管理器
    "break_graph",  # 插入图断点的函数
]


def _check_cuda_bindings():  # 检查CUDA绑定是否可用，不可用则抛出导入错误
    """Raise ImportError if cuda-python bindings are not available."""  # 如果cuda-python绑定不可用则抛出ImportError
    if rt is None:  # 如果CUDA绑定不可用
        raise ImportError(  # 抛出导入错误
            "Breakable CUDA graph requires the 'cuda-python' package. "  # 可中断CUDA图需要cuda-python包
            "Install it with: pip install cuda-python"  # 安装命令
        )


# Active BreakableCUDAGraphCapture context for the currently-capturing thread.  # 当前捕获线程的活动BreakableCUDAGraphCapture上下文
# eager_on_graph's wrapper uses this to split the current torch.cuda.CUDAGraph  # eager_on_graph的包装器使用此变量在断点处分割当前的torch.cuda.CUDAGraph
# at break points.  # 在断点处分割。
_current_capture_var: ContextVar["BreakableCUDAGraphCapture | None"] = ContextVar(  # 当前捕获上下文变量
    "current_capture", default=None  # 默认值为None
)
_current_stream_var: ContextVar[torch.cuda.Stream | None] = ContextVar(  # 当前CUDA流上下文变量
    "current_stream", default=None  # 默认值为None
)
_forked_streams_var: ContextVar[set[torch.cuda.Stream] | None] = ContextVar(  # 分叉流集合上下文变量
    "forked_streams", default=None  # 默认值为None
)


def get_current_stream(device: torch.device | None = None) -> torch.cuda.Stream:  # 获取当前捕获上下文中的CUDA流，若不在捕获中则返回默认流
    """Return the stream active in the current BCG capture, or the default stream."""  # 返回当前BCG捕获中的活跃流，或默认流
    stream = _current_stream_var.get()  # 从上下文变量获取当前流
    if stream is None:  # 如果不在BCG捕获上下文中
        return torch.cuda.current_stream(device)  # 返回指定设备的当前CUDA流
    return stream  # 返回BCG捕获上下文中的流


def _capture_status(stream_ptr: int) -> "rt.cudaStreamCaptureStatus":  # 查询指定CUDA流的捕获状态
    """Query the capture status of a CUDA stream."""  # 查询CUDA流的捕获状态
    _check_cuda_bindings()  # 检查CUDA绑定是否可用
    status, *_ = checkCudaErrors(rt.cudaStreamGetCaptureInfo(stream_ptr))  # 调用CUDA API获取流捕获信息
    return status  # 返回捕获状态


def _is_capturing(stream_ptr: int) -> bool:  # 判断指定CUDA流是否正在捕获中
    """Return True if the stream is actively capturing."""  # 如果流正在活跃捕获中则返回True
    _check_cuda_bindings()  # 检查CUDA绑定是否可用
    return (  # 返回判断结果
        _capture_status(stream_ptr)  # 获取流捕获状态
        == rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive  # 是否为活跃捕获状态
    )


# Hook torch.cuda.Stream.wait_stream to track side-stream forks/joins that happen  # 钩住torch.cuda.Stream.wait_stream以跟踪可中断捕获期间发生的侧流分叉/汇合
# during breakable capture. We need this because capture_end() on a torch  # 我们需要这个是因为torch CUDAGraph上的capture_end()
# CUDAGraph fails if there are still side streams participating in the capture  # 在仍有侧流参与捕获时会失败
# — so before ending each segment we auto-join any forked-but-not-rejoined streams.  # ——所以在结束每个段之前，我们自动汇合所有已分叉但未汇合的流。
_original_wait_stream: Callable | None = None  # 保存原始的wait_stream方法
_hook_lock = threading.Lock()  # 钩子安装/卸载的线程锁
_hook_refcount = 0  # 钩子引用计数，用于支持嵌套安装


def _hooked_wait_stream(self: torch.cuda.Stream, other: torch.cuda.Stream):  # 钩住的wait_stream实现，用于跟踪侧流分叉和汇合
    """Replacement for Stream.wait_stream that tracks forked side streams during BCG capture."""  # wait_stream的替换实现，在BCG捕获期间跟踪分叉的侧流
    assert _original_wait_stream is not None  # 断言原始方法已保存
    forked = _forked_streams_var.get()  # 获取当前分叉流集合
    if forked is None:  # 如果不在BCG捕获上下文中
        _original_wait_stream(self, other)  # 调用原始方法
        return  # 直接返回
    capturing = _current_stream_var.get()  # 获取当前捕获流
    if capturing is None:  # 如果没有捕获流
        _original_wait_stream(self, other)  # 调用原始方法
        return  # 直接返回

    cap_ptr = capturing.cuda_stream  # 获取捕获流的CUDA流指针
    is_self_cap = self is capturing or self.cuda_stream == cap_ptr  # 判断self是否为捕获流
    is_other_cap = other is capturing or other.cuda_stream == cap_ptr  # 判断other是否为捕获流

    if is_self_cap and not is_other_cap:  # 捕获流等待非捕获侧流（汇合）
        if (  # 如果侧流不在活跃捕获状态
            _capture_status(other.cuda_stream)  # 查询侧流捕获状态
            != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive  # 不等于活跃捕获
        ):
            return  # 侧流未参与捕获，无需等待
        _original_wait_stream(self, other)  # 调用原始wait_stream进行同步
        forked.discard(other)  # 从分叉集合中移除已汇合的侧流
    elif is_other_cap and not is_self_cap:  # 非捕获流等待捕获流（分叉）
        _original_wait_stream(self, other)  # 调用原始wait_stream进行同步
        forked.add(self)  # 将发起等待的流加入分叉集合
    else:  # 其他情况（两个都是捕获流或都不是）
        _original_wait_stream(self, other)  # 调用原始wait_stream


def _install_wait_stream_hook():  # 安装wait_stream钩子
    """Install the wait_stream hook (refcounted)."""  # 安装wait_stream钩子（引用计数管理）
    global _original_wait_stream, _hook_refcount  # 声明使用全局变量
    with _hook_lock:  # 获取线程锁
        if _hook_refcount == 0:  # 如果是首次安装
            _original_wait_stream = torch.cuda.Stream.wait_stream  # 保存原始方法
            torch.cuda.Stream.wait_stream = _hooked_wait_stream  # type: ignore[assignment]  # 替换为钩住版本
        _hook_refcount += 1  # 增加引用计数


def _uninstall_wait_stream_hook():  # 卸载wait_stream钩子
    """Uninstall the wait_stream hook (refcounted)."""  # 卸载wait_stream钩子（引用计数管理）
    global _original_wait_stream, _hook_refcount  # 声明使用全局变量
    with _hook_lock:  # 获取线程锁
        _hook_refcount -= 1  # 减少引用计数
        if _hook_refcount == 0:  # 如果引用计数归零
            assert _original_wait_stream is not None, "wait_stream hook not installed"  # 断言原始方法已保存
            torch.cuda.Stream.wait_stream = _original_wait_stream  # type: ignore[assignment]  # 恢复原始方法
            _original_wait_stream = None  # 清空保存的原始方法


def _weak_ref_if_tensor(x):  # 如果输入是张量则返回弱引用视图，否则原样返回
    """Return a weak-ref tensor view (shared storage, no refcount) for tensors;  # 对张量返回弱引用张量视图（共享存储，无引用计数）；
    pass-through for non-tensors. Weak-ref'ing captured args lets the shared  # 非张量直接透传。弱引用捕获的参数使共享
    mempool reclaim per-layer intermediates between segments — storage stays  # 内存池可以在段之间回收每层的中间张量——存储通过
    alive for each segment CUDAGraph's lifetime via its pool use_count.  # 每个段CUDAGraph的池use_count在其生命周期内保持存活。

    ``weak_ref_tensors`` is imported lazily: the module hard-raises on  # weak_ref_tensors是延迟导入的：该模块在
    non-CUDA/NPU platforms, and we only reach this code during an active  # 非CUDA/NPU平台上会硬报错，而我们只在活跃的
    BCG capture (which can't happen on CPU-only runners anyway)."""  # BCG捕获期间才会执行此代码（在仅CPU的运行器上不可能发生）。
    if torch.is_tensor(x):  # 如果输入是张量
        from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors  # 延迟导入弱引用张量工具

        return weak_ref_tensors(x)  # 返回弱引用张量视图
    return x  # 非张量直接返回


def _copy_output(dst: Any, src: Any) -> Any:  # 将src输出复制到dst中，支持张量、数据类/对象和字典
    """Copy src output into dst in-place where possible.  # 尽可能就地复制src输出到dst。

    Handles plain tensors, dataclass/object with tensor attributes,  # 处理普通张量、具有张量属性的数据类/对象，
    and dicts of tensors. Returns dst if in-place copy succeeded,  # 以及张量字典。如果就地复制成功则返回dst，
    otherwise returns src.  # 否则返回src。
    """
    if torch.is_tensor(dst) and torch.is_tensor(src):  # 如果源和目标都是张量
        dst.copy_(src)  # 就地复制张量数据
        return dst  # 返回目标张量

    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):  # 如果源和目标都是对象/数据类
        for key, src_val in src.__dict__.items():  # 遍历源对象的所有属性
            dst_val = getattr(dst, key, None)  # 获取目标对象对应属性
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):  # 如果属性都是张量
                dst_val.copy_(src_val)  # 就地复制张量数据
            else:  # 否则
                setattr(dst, key, src_val)  # 直接设置属性值
        return dst  # 返回目标对象

    if isinstance(dst, dict) and isinstance(src, dict):  # 如果源和目标都是字典
        for key, src_val in src.items():  # 遍历源字典的所有键值对
            dst_val = dst.get(key)  # 获取目标字典对应键的值
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):  # 如果值都是张量
                dst_val.copy_(src_val)  # 就地复制张量数据
            else:  # 否则
                dst[key] = src_val  # 直接赋值
        return dst  # 返回目标字典

    return src  # 无法就地复制时返回源值


def eager_on_graph(enable: bool):  # 装饰器，在CUDA图捕获期间在被装饰函数处插入急切断点
    """Decorator that inserts an eager break point during CUDA graph capture.  # 装饰器，在CUDA图捕获期间插入急切断点。

    When ``enable`` is False the callable is returned unchanged.  # 当enable为False时，可调用对象原样返回。
    When inside a BCG capture the current segment is ended, the  # 在BCG捕获期间，当前段结束，
    decorated function runs eagerly, and a new segment begins.  # 被装饰函数急切执行，然后开始新段。
    """
    def decorator(inner: Callable):  # 内部装饰器函数
        if not enable:  # 如果未启用
            return inner  # 返回原始函数

        def wrapper(*args, **kwargs):  # 包装函数
            capture = _current_capture_var.get()  # 获取当前捕获上下文
            if capture is None:  # 如果不在捕获上下文中
                return inner(*args, **kwargs)  # 直接执行原始函数

            logger.debug("Break graph due to function: %s", inner.__name__)  # 记录因哪个函数打断图

            # End the segment that captured up to this break point.  # 结束捕获到此断点的段。
            capture._end_current_segment()  # 结束当前段

            # Run the eager function once so it allocates its outputs and  # 运行一次急切函数以分配其输出并
            # writes real data into them.  # 将真实数据写入其中。
            output = inner(*args, **kwargs)  # 执行急切函数获取输出

            # Weak-ref the closure state. Storage lives with the segment  # 弱引用闭包状态。存储与段CUDAGraph的
            # CUDAGraphs' mempool pin; Python refs don't need to prevent  # 内存池固定共存；Python引用不需要阻止
            # pool reuse across layers.  # 跨层的池重用。
            captured_inner = inner  # 捕获原始函数引用
            captured_args = tuple(_weak_ref_if_tensor(a) for a in args)  # 弱引用所有张量参数
            captured_kwargs = {k: _weak_ref_if_tensor(v) for k, v in kwargs.items()}  # 弱引用所有张量关键字参数
            captured_output = _weak_ref_if_tensor(output)  # 弱引用输出张量

            def replay_fn():  # 重放函数，在图重放时执行急切函数
                new_out = captured_inner(*captured_args, **captured_kwargs)  # 使用捕获的参数执行函数
                return _copy_output(captured_output, new_out)  # 将新输出复制到捕获的输出中

            capture.cuda_graph._break_fns.append(replay_fn)  # 将重放函数添加到断点函数列表

            # Start a fresh CUDAGraph segment for the remainder of the forward.  # 为前向传播的剩余部分开始新的CUDAGraph段。
            capture._begin_new_segment()  # 开始新段
            return output  # 返回急切函数的输出

        return wrapper  # 返回包装函数

    return decorator  # 返回装饰器


class BreakableCUDAGraph:  # 可中断CUDA图容器，持有一系列段图和断点函数
    """Container holding one ``torch.cuda.CUDAGraph`` per segment plus an  # 容器，每个段持有一个torch.cuda.CUDAGraph，加上
    eager break function between consecutive segments."""  # 相邻段之间的急切断点函数。"""

    def __init__(self) -> None:  # 初始化可中断CUDA图容器
        self._segments: list[torch.cuda.CUDAGraph] = []  # 存储所有CUDA图段
        self._break_fns: list[Callable[[], Any]] = []  # 存储段之间的急切断点函数

    def replay(self) -> None:  # 重放所有段图和断点函数
        """Replay all segments and eager break functions in order."""  # 按顺序重放所有段和急切断点函数
        stream = torch.cuda.current_stream()  # 获取当前CUDA流
        token = _current_stream_var.set(stream)  # 设置当前流上下文变量
        try:  # 尝试执行
            for i, seg in enumerate(self._segments):  # 遍历所有段
                seg.replay()  # 重放当前段
                if i < len(self._break_fns):  # 如果当前段之后有断点函数
                    self._break_fns[i]()  # 执行断点函数
        finally:  # 无论如何执行清理
            _current_stream_var.reset(token)  # 恢复流上下文变量


class BreakableCUDAGraphCapture:  # 可中断CUDA图捕获上下文管理器
    """Context manager that captures the enclosed code as one or more  # 上下文管理器，将包含的代码捕获为一个或多个
    ``torch.cuda.CUDAGraph`` segments separated by eager break points.  # 由急切断点分隔的torch.cuda.CUDAGraph段。

    Each segment shares the supplied ``pool`` (``MempoolId_t`` tuple) so  # 每个段共享提供的pool（MempoolId_t元组），因此
    pool-allocated intermediates can be reused across segments. While any  # 池分配的中间张量可以跨段重用。只要任一
    segment is alive, its ``beginAllocateToPool`` call keeps the mempool's  # 段存活，其beginAllocateToPool调用保持内存池的
    ``use_count`` > 0, which makes ``weak_ref_tensor`` of segment-allocated  # use_count > 0，这使段分配张量的weak_ref_tensor
    tensors safe across subsequent replays.  # 在后续重放中安全。
    """

    def __init__(  # 初始化捕获上下文管理器
        self,
        cuda_graph: BreakableCUDAGraph,  # 目标可中断CUDA图容器
        pool=None,  # 共享内存池ID，默认为(0,0)
        stream: torch.cuda.Stream | None = None,  # 捕获使用的CUDA流
        capture_error_mode: str = "global",  # 捕获错误模式
    ):
        assert isinstance(  # 断言类型检查
            cuda_graph, BreakableCUDAGraph  # 检查是否为BreakableCUDAGraph实例
        ), "cuda_graph must be a BreakableCUDAGraph"  # 错误提示
        self.cuda_graph = cuda_graph  # 保存CUDA图容器引用
        self._pool = pool if pool is not None else (0, 0)  # 设置内存池，默认为(0,0)
        self._stream = stream  # 保存CUDA流
        self._capture_error_mode = capture_error_mode  # 保存捕获错误模式
        self._stream_ctx = None  # 流上下文管理器
        self._capture_token = None  # 捕获上下文变量令牌
        self._stream_token = None  # 流上下文变量令牌
        self._forked_token = None  # 分叉流上下文变量令牌

    def __enter__(self):  # 进入上下文，初始化捕获环境
        _install_wait_stream_hook()  # 安装wait_stream钩子
        if self._stream is not None:  # 如果指定了捕获流
            self._stream_ctx = torch.cuda.stream(self._stream)  # 创建流上下文
            self._stream_ctx.__enter__()  # 进入流上下文
        self._capture_token = _current_capture_var.set(self)  # 设置当前捕获上下文
        self._stream_token = _current_stream_var.set(  # 设置当前流上下文
            self._stream or torch.cuda.current_stream()  # 使用指定流或当前默认流
        )
        self._forked_token = _forked_streams_var.set(set())  # 初始化分叉流集合
        self._begin_new_segment()  # 开始第一个段
        return self  # 返回自身

    def __exit__(self, *args: object):  # 退出上下文，结束捕获并恢复环境
        try:  # 尝试执行
            self._end_current_segment()  # 结束当前段
        finally:  # 无论如何执行清理
            _forked_streams_var.reset(self._forked_token)  # 恢复分叉流上下文
            _current_stream_var.reset(self._stream_token)  # 恢复流上下文
            _current_capture_var.reset(self._capture_token)  # 恢复捕获上下文
            if self._stream_ctx is not None:  # 如果有流上下文
                self._stream_ctx.__exit__(*args)  # 退出流上下文
                self._stream_ctx = None  # 清空流上下文
            _uninstall_wait_stream_hook()  # 卸载wait_stream钩子
        return False  # 不抑制异常

    def _begin_new_segment(self) -> None:  # 开始新的CUDA图段
        """Begin capturing a new CUDAGraph segment."""  # 开始捕获新的CUDAGraph段
        graph = torch.cuda.CUDAGraph()  # 创建新的CUDAGraph
        graph.capture_begin(  # 开始捕获
            pool=self._pool, capture_error_mode=self._capture_error_mode  # 使用共享内存池和错误模式
        )
        self.cuda_graph._segments.append(graph)  # 将新段添加到段列表

    def _end_current_segment(self) -> None:  # 结束当前CUDA图段
        """End the current CUDAGraph segment, auto-joining forked side streams."""  # 结束当前CUDAGraph段，自动汇合分叉的侧流
        # Auto-join any side streams forked during this segment but not joined.  # 自动汇合在本段中分叉但未汇合的侧流。
        main_stream = get_current_stream()  # 获取主捕获流
        forked = _forked_streams_var.get()  # 获取分叉流集合
        if forked:  # 如果有未汇合的分叉流
            assert _original_wait_stream is not None  # 断言原始wait_stream已保存
            for side in list(forked):  # 遍历所有分叉流
                if _is_capturing(side.cuda_stream):  # 如果侧流仍在捕获中
                    _original_wait_stream(main_stream, side)  # 使用原始wait_stream汇合
            forked.clear()  # 清空分叉流集合
        self.cuda_graph._segments[-1].capture_end()  # 结束最后一个段的捕获


@eager_on_graph(True)  # 使用eager_on_graph装饰器，启用图断点
def break_graph() -> None:  # 插入图断点，在CUDA图捕获期间分割为两个段
    """Insert a graph break. The @eager_on_graph decorator does the actual  # 插入图断点。@eager_on_graph装饰器执行实际的
    segment split; this function body intentionally does nothing."""  # 段分割；此函数体故意不做任何事情。
    pass  # 空操作，实际工作由装饰器完成
