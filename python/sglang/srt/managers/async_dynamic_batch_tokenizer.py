"""
Asynchronous dynamic batch tokenizer for SGLang.

This module provides an async tokenizer with dynamic batching capabilities
to reduce tokenization overhead when multiple requests arrive concurrently.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AsyncDynamicbatchTokenizer:
    """Asynchronous tokenizer with dynamic batching for single string prompts.

    Dynamically batches pending encode requests from a queue to reduce overhead.
    Only handles single string prompts - regular batch processing of multiple
    strings per request should be handled at a higher level.
    A single-thread ThreadPoolExecutor is used so the event loop stays responsive.

    Note: Uses lazy initialization for asyncio components because this class
    is instantiated in TokenizerManager.__init__() before the event loop starts.
    """

    def __init__(
        self,
        tokenizer,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.002,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s

        # Single queue for all encode requests - initialized lazily
        self._queue: Optional[asyncio.Queue] = None
        self._batcher_task: Optional[asyncio.Task] = None

        # Single-thread executor for blocking tokenizer calls
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of event loop dependent components."""
        if not self._initialized:
            self._queue = asyncio.Queue()
            self._batcher_task = asyncio.create_task(self._dynamic_batch_loop())
            self._initialized = True

    async def __call__(self, prompt: str, **kwargs) -> Any:
        """Encode a single prompt."""
        return await self.encode(prompt, **kwargs)

    async def encode(self, prompt: str, **kwargs) -> Any:
        """异步编码单条 prompt，并利用后台动态合批降低 tokenizer 开销。

        调用方看到的是单条 prompt 的 encode 接口；内部会把请求放入队列，
        由 `_dynamic_batch_loop()` 在很短时间窗口内收集并合并多个并发请求。
        当前协程通过 `result_future` 等待后台线程池中的 tokenizer 调用完成。
        """
        # 首次调用时再创建 asyncio.Queue 和后台合批任务，避免构造函数依赖事件循环。
        self._ensure_initialized()

        # 为本次 encode 创建一个 Future；后台合批处理完成后会写入结果或异常。
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()

        # 将 prompt、tokenizer 参数和 Future 一起入队，交给后台动态合批循环处理。
        await self._queue.put((prompt, kwargs, result_future))

        # 等待后台 tokenizer 处理完成；返回值与 tokenizer 单条调用结果保持一致。
        return await result_future

    async def _dynamic_batch_loop(self):
        """后台动态合批循环，持续从队列中收集 encode 请求并批量处理。

        该循环由 `encode()` 首次调用时启动。它先阻塞等待第一个请求，
        然后在 `batch_wait_timeout_s` 的短窗口内尽量收集更多请求，最多收集
        `max_batch_size` 条。收集到的一批请求会交给 `_process_dynamic_batch()`
        统一执行 tokenizer，并把结果写回各自的 Future。
        """
        while True:
            try:
                # 阻塞等待第一个请求；没有请求时不占用 CPU。
                prompt, kwargs, result_future = await self._queue.get()

                # 初始化当前动态 batch，每个请求都保留自己的参数和 Future。
                prompts = [prompt]
                kwargs_list = [kwargs]
                result_futures = [result_future]

                # 如果队列当前为空，说明没有并发请求在等待，直接处理单条请求以降低延迟。
                if self._queue.empty():
                    pass
                else:
                    # 队列里已经有请求，进入短暂等待窗口，尽量聚合更多请求提升吞吐。
                    start_time = asyncio.get_running_loop().time()

                    # 在最大 batch 大小或等待超时两者之一触达前，持续收集队列请求。
                    while len(prompts) < self.max_batch_size:
                        elapsed = asyncio.get_running_loop().time() - start_time
                        if elapsed >= self.batch_wait_timeout_s:
                            break

                        remaining_time = self.batch_wait_timeout_s - elapsed
                        try:
                            # 等待下一个请求，但不会超过剩余合批窗口。
                            prompt, kwargs, result_future = await asyncio.wait_for(
                                self._queue.get(), remaining_time
                            )
                            prompts.append(prompt)
                            kwargs_list.append(kwargs)
                            result_futures.append(result_future)
                        except asyncio.TimeoutError:
                            # 合批窗口结束，停止收集并处理当前 batch。
                            break

                # 记录本轮实际合批大小，便于观察动态合批是否生效。
                logger.debug(
                    f"AsyncDynamicbatchTokenizer: Processing dynamic batch of size {len(prompts)}"
                )

                # 执行本轮合批 tokenize，并将结果/异常写回各自 Future。
                await self._process_dynamic_batch(prompts, kwargs_list, result_futures)

            except Exception as e:
                logger.error(f"Error in dynamic batch loop: {e}")
                # 后台循环不能因为单轮异常退出，否则后续 encode 请求会永久等待。

    async def _process_dynamic_batch(  # 异步处理已经收集好的动态合批请求。
        self,  # 当前 AsyncDynamicbatchTokenizer 实例。
        prompts: List[str],  # 本批次中等待 tokenize 的原始字符串列表。
        kwargs_list: List[Dict],  # 与 prompts 一一对应的 tokenizer 参数列表。
        result_futures: List[asyncio.Future],  # 与 prompts 一一对应、用于回填结果的 Future 列表。
    ) -> None:  # 该方法只通过 Future 返回结果或异常，本身不返回值。
        """处理单字符串 prompt 的动态合批 encode 请求，并把结果写回对应 Future。"""
        first_kw = kwargs_list[0]  # 取第一个请求的 tokenizer 参数作为合批兼容性基准。
        can_batch = all(kw == first_kw for kw in kwargs_list[1:])  # 判断除首个请求外的参数是否都与基准一致。
        kwargs = first_kw if can_batch else None  # 只有所有参数一致时才保留可用于批量调用的公共参数。

        try:  # 捕获整个批处理过程中的异常，避免调用方 Future 永久等待。
            if can_batch and len(prompts) > 1:  # 多个请求且参数完全一致时，走 tokenizer 的批量路径。
                encode_fn = partial(self.tokenizer, prompts, **kwargs)  # 绑定批量 prompts 和公共参数，延迟到线程池执行。
                results = await asyncio.get_running_loop().run_in_executor(  # 在线程池中执行同步 tokenizer，避免阻塞事件循环。
                    self._executor, encode_fn  # 指定专用执行器和待执行的批量 tokenize 函数。
                )

                for i, fut in enumerate(result_futures):  # 遍历每个请求对应的 Future，并保留其批内下标。
                    if not fut.done():  # 仅在 Future 尚未完成时写入结果，避免覆盖取消或已设置的状态。
                        data = {k: v[i] for k, v in results.items()}  # 从批量结果中抽取当前下标对应的单请求结果。
                        fut.set_result(data)  # 将当前请求的 tokenize 结果写回对应 Future。
            else:  # 单请求或参数不一致时，逐条调用 tokenizer。
                if len(prompts) > 1 and not can_batch:  # 多请求但参数不同会失去真正批量 tokenize 的性能收益。
                    logger.warning(  # 记录参数不一致导致无法批量处理的性能告警。
                        f"AsyncDynamicbatchTokenizer: Dynamic batching disabled for batch of {len(prompts)} "  # 说明本批大小和动态合批被禁用。
                        f"requests due to differing kwargs. This reduces performance benefits. "  # 说明禁用原因和性能影响。
                        f"Consider using consistent tokenization parameters across requests."  # 提示调用方尽量使用一致参数。
                    )

                encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [  # 构造逐条 tokenize 的闭包，并冻结当前批次数据。
                    self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs_list)  # 按 prompt 与参数一一配对执行 tokenizer。
                ]
                results = await asyncio.get_running_loop().run_in_executor(  # 在线程池中执行逐条 tokenize，避免阻塞事件循环。
                    self._executor, encode_fn  # 指定专用执行器和待执行的逐条 tokenize 函数。
                )

                for fut, res in zip(result_futures, results):  # 将逐条 tokenize 的结果与 Future 一一配对。
                    if not fut.done():  # 仅在 Future 尚未完成时写入结果，避免覆盖取消或已设置的状态。
                        fut.set_result(res)  # 将当前请求的 tokenize 结果写回对应 Future。
        except Exception as e:  # 捕获 tokenizer 或线程池执行中的任意异常。
            logger.error(f"Error in dynamic batch processing: {e}")  # 记录动态批处理失败原因，便于排查后台任务错误。
            for fut in result_futures:  # 遍历本批次所有等待结果的 Future。
                if not fut.done():  # 仅对尚未完成的 Future 设置异常。
                    fut.set_exception(e)  # 将同一个异常传播给对应调用方。

    def __del__(self):
        """Clean up background tasks."""
        if hasattr(self, "_batcher_task") and self._batcher_task:
            if not self._batcher_task.done():
                self._batcher_task.cancel()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
