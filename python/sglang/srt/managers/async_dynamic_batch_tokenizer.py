# 异步动态批量分词器模块
# 本模块为SGLang提供带有动态批量处理能力的异步分词器，
# 当多个请求并发到达时，通过动态合并编码请求来减少分词开销。
# 使用单线程ThreadPoolExecutor以保持事件循环的响应性。

"""
Asynchronous dynamic batch tokenizer for SGLang.
异步动态批量分词器，用于SGLang。

This module provides an async tokenizer with dynamic batching capabilities
to reduce tokenization overhead when multiple requests arrive concurrently.
本模块提供带有动态批量处理能力的异步分词器，以减少并发请求时的分词开销。
"""

import asyncio  # 导入异步IO模块 # 导入异步IO模块
import logging  # 导入日志模块 # 导入日志模块
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器 # 导入线程池执行器
from functools import partial  # 导入偏函数工具 # 导入偏函数工具
from typing import Any, Dict, List, Optional  # 导入类型注解 # 导入类型注解

logger = logging.getLogger(__name__)  # 创建日志记录器 # 创建日志记录器


class AsyncDynamicbatchTokenizer:  # 异步动态批量分词器类 # 异步动态批量分词器类
    """Asynchronous tokenizer with dynamic batching for single string prompts.
    用于单个字符串提示的异步动态批量分词器。

    Dynamically batches pending encode requests from a queue to reduce overhead.
    动态地从队列中批量处理待编码请求以减少开销。
    Only handles single string prompts - regular batch processing of multiple
    strings per request should be handled at a higher level.
    仅处理单个字符串提示 - 每个请求多个字符串的常规批量处理应在更上层处理。
    A single-thread ThreadPoolExecutor is used so the event loop stays responsive.
    使用单线程ThreadPoolExecutor以保持事件循环的响应性。

    Note: Uses lazy initialization for asyncio components because this class
    is instantiated in TokenizerManager.__init__() before the event loop starts.
    注意：对asyncio组件使用延迟初始化，因为此类在事件循环启动前
    在TokenizerManager.__init__()中被实例化。
    """

    def __init__(  # 初始化方法 # 初始化方法
        self,
        tokenizer,  # 分词器实例 # 分词器实例
        max_batch_size: int = 32,  # 最大批量大小，默认32 # 最大批量大小，默认32
        batch_wait_timeout_s: float = 0.002,  # 批量等待超时时间（秒），默认0.002 # 批量等待超时时间（秒），默认0.002
    ) -> None:
        self.tokenizer = tokenizer  # 保存分词器引用 # 保存分词器引用
        self.max_batch_size = max_batch_size  # 保存最大批量大小 # 保存最大批量大小
        self.batch_wait_timeout_s = batch_wait_timeout_s  # 保存批量等待超时时间 # 保存批量等待超时时间

        # Single queue for all encode requests - initialized lazily
        # 所有编码请求的单一队列 - 延迟初始化
        self._queue: Optional[asyncio.Queue] = None  # 异步队列，延迟初始化 # 异步队列，延迟初始化
        self._batcher_task: Optional[asyncio.Task] = None  # 批量处理任务，延迟初始化 # 批量处理任务，延迟初始化

        # Single-thread executor for blocking tokenizer calls
        # 用于阻塞式分词器调用的单线程执行器
        self._executor = ThreadPoolExecutor(max_workers=1)  # 创建单线程执行器 # 创建单线程执行器
        self._initialized = False  # 初始化标志 # 初始化标志

    def _ensure_initialized(self):  # 确保已初始化 # 确保已初始化
        """Lazy initialization of event loop dependent components.
        事件循环依赖组件的延迟初始化。"""
        if not self._initialized:  # 如果尚未初始化 # 如果尚未初始化
            self._queue = asyncio.Queue()  # 创建异步队列 # 创建异步队列
            self._batcher_task = asyncio.create_task(self._dynamic_batch_loop())  # 创建批量处理任务 # 创建批量处理任务
            self._initialized = True  # 设置初始化标志 # 设置初始化标志

    async def __call__(self, prompt: str, **kwargs) -> Any:  # 可调用方法 # 可调用方法
        """Encode a single prompt.
        编码单个提示。"""
        return await self.encode(prompt, **kwargs)  # 调用encode方法 # 调用encode方法

    async def encode(self, prompt: str, **kwargs) -> Any:  # 编码方法 # 编码方法
        """Encode a single prompt.
        编码单个提示。"""
        self._ensure_initialized()  # 确保已初始化 # 确保已初始化
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()  # 创建未来结果对象 # 创建未来结果对象
        await self._queue.put((prompt, kwargs, result_future))  # 将请求放入队列 # 将请求放入队列
        return await result_future  # 等待并返回结果 # 等待并返回结果

    async def _dynamic_batch_loop(self):  # 动态批量处理循环 # 动态批量处理循环
        """Dynamically batch incoming encode requests for efficiency.
        动态批量处理传入的编码请求以提高效率。"""
        while True:  # 无限循环 # 无限循环
            try:  # 异常处理 # 异常处理
                # Get the first request
                # 获取第一个请求
                prompt, kwargs, result_future = await self._queue.get()  # 从队列获取请求 # 从队列获取请求

                # Collect requests into dynamic batch
                # 收集请求到动态批量中
                prompts = [prompt]  # 初始化提示列表 # 初始化提示列表
                kwargs_list = [kwargs]  # 初始化参数列表 # 初始化参数列表
                result_futures = [result_future]  # 初始化未来结果列表 # 初始化未来结果列表

                # Check if there are more items immediately available in the queue
                # If queue is empty, process single item immediately without timeout
                # 检查队列中是否有更多立即可用的项
                # 如果队列为空，立即处理单项，不等待超时
                if self._queue.empty():  # 如果队列为空 # 如果队列为空
                    # No other requests waiting, process immediately
                    # 没有其他请求等待，立即处理
                    pass  # 直接通过 # 直接通过
                else:  # 否则 # 否则
                    # There might be more requests, wait for dynamic batching opportunity
                    # 可能还有更多请求，等待动态批处理机会
                    start_time = asyncio.get_running_loop().time()  # 记录开始时间 # 记录开始时间

                    # Collect more requests up to max_batch_size or batch_wait_timeout_s
                    # 收集更多请求直到达到max_batch_size或batch_wait_timeout_s
                    while len(prompts) < self.max_batch_size:  # 未达到最大批量大小 # 未达到最大批量大小
                        elapsed = asyncio.get_running_loop().time() - start_time  # 计算已用时间 # 计算已用时间
                        if elapsed >= self.batch_wait_timeout_s:  # 如果超过等待超时 # 如果超过等待超时
                            break  # 跳出循环 # 跳出循环

                        remaining_time = self.batch_wait_timeout_s - elapsed  # 计算剩余等待时间 # 计算剩余等待时间
                        try:  # 尝试获取更多请求 # 尝试获取更多请求
                            prompt, kwargs, result_future = await asyncio.wait_for(  # 带超时等待获取请求 # 带超时等待获取请求
                                self._queue.get(), remaining_time
                            )
                            prompts.append(prompt)  # 添加提示到列表 # 添加提示到列表
                            kwargs_list.append(kwargs)  # 添加参数到列表 # 添加参数到列表
                            result_futures.append(result_future)  # 添加未来结果到列表 # 添加未来结果到列表
                        except asyncio.TimeoutError:  # 超时则停止收集 # 超时则停止收集
                            break  # 跳出循环 # 跳出循环

                # Log dynamic batch information
                # 记录动态批量信息
                logger.debug(
                    f"AsyncDynamicbatchTokenizer: Processing dynamic batch of size {len(prompts)}"
                )

                # Process the dynamic batch
                # 处理动态批量
                await self._process_dynamic_batch(prompts, kwargs_list, result_futures)  # 调用批量处理方法 # 调用批量处理方法

            except Exception as e:  # 捕获异常 # 捕获异常
                logger.error(f"Error in dynamic batch loop: {e}")  # 记录错误日志 # 记录错误日志
                # Continue the loop to handle other requests
                # 继续循环以处理其他请求

    async def _process_dynamic_batch(  # 处理动态批量 # 处理动态批量
        self,
        prompts: List[str],  # 提示列表 # 提示列表
        kwargs_list: List[Dict],  # 参数列表 # 参数列表
        result_futures: List[asyncio.Future],  # 未来结果列表 # 未来结果列表
    ) -> None:  # 无返回值 # 无返回值
        """Process a dynamic batch of encode requests for single string prompts.
        处理单个字符串提示的动态批量编码请求。"""
        # Check if all kwargs are identical for efficient batch processing
        # 检查所有参数是否相同，以便高效批量处理
        first_kw = kwargs_list[0]  # 获取第一个请求的参数 # 获取第一个请求的参数
        can_batch = all(kw == first_kw for kw in kwargs_list[1:])  # 检查是否所有参数相同 # 检查是否所有参数相同
        kwargs = first_kw if can_batch else None  # 如果可以批量则使用第一个参数，否则为None # 如果可以批量则使用第一个参数，否则为None

        try:  # 异常处理 # 异常处理
            # If every request uses identical kwargs we can run a single
            # batch tokenizer call for a big speed-up.
            # 如果每个请求使用相同的参数，可以运行单次批量分词器调用以大幅提速。
            if can_batch and len(prompts) > 1:  # 如果可以批量且请求多于1个 # 如果可以批量且请求多于1个
                encode_fn = partial(self.tokenizer, prompts, **kwargs)  # 创建批量编码函数 # 创建批量编码函数
                results = await asyncio.get_running_loop().run_in_executor(  # 在执行器中运行 # 在执行器中运行
                    self._executor, encode_fn
                )

                for i, fut in enumerate(result_futures):  # 遍历未来结果 # 遍历未来结果
                    if not fut.done():  # 如果未来结果未完成 # 如果未来结果未完成
                        data = {k: v[i] for k, v in results.items()}  # 提取第i个结果 # 提取第i个结果
                        fut.set_result(data)  # 设置未来结果 # 设置未来结果
            else:  # 否则 # 否则
                # Process each request individually due to different kwargs
                # 由于参数不同，逐个处理请求
                if len(prompts) > 1 and not can_batch:  # 如果请求多于1个且不能批量 # 如果请求多于1个且不能批量
                    logger.warning(
                        f"AsyncDynamicbatchTokenizer: Dynamic batching disabled for batch of {len(prompts)} "
                        f"requests due to differing kwargs. This reduces performance benefits. "
                        f"Consider using consistent tokenization parameters across requests."
                    )  # 记录警告日志 # 记录警告日志

                encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [  # 创建逐个编码函数 # 创建逐个编码函数
                    self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs_list)
                ]
                results = await asyncio.get_running_loop().run_in_executor(  # 在执行器中运行 # 在执行器中运行
                    self._executor, encode_fn
                )

                for fut, res in zip(result_futures, results):  # 遍历结果 # 遍历结果
                    if not fut.done():  # 如果未来结果未完成 # 如果未来结果未完成
                        fut.set_result(res)  # 设置未来结果 # 设置未来结果
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"Error in dynamic batch processing: {e}")  # 记录错误日志 # 记录错误日志
            for fut in result_futures:  # 遍历所有未来结果 # 遍历所有未来结果
                if not fut.done():  # 如果未来结果未完成 # 如果未来结果未完成
                    fut.set_exception(e)  # 设置异常 # 设置异常

    def __del__(self):  # 析构方法 # 析构方法
        """Clean up background tasks.
        清理后台任务。"""
        if hasattr(self, "_batcher_task") and self._batcher_task:  # 如果存在批量处理任务 # 如果存在批量处理任务
            if not self._batcher_task.done():  # 如果任务未完成 # 如果任务未完成
                self._batcher_task.cancel()  # 取消任务 # 取消任务
        if hasattr(self, "_executor"):  # 如果存在执行器 # 如果存在执行器
            self._executor.shutdown(wait=False)  # 关闭执行器，不等待 # 关闭执行器，不等待
