"""The interpreter that executes SGL programs"""
# 本文件实现了SGL程序解释器，包含流式执行器（StreamExecutor）、程序状态（ProgramState）、
# 程序状态组（ProgramStateGroup）以及运行程序的相关函数（run_program, run_program_batch, cache_program）。
# StreamExecutor负责在后台线程中执行SGL表达式，ProgramState是对执行器的状态封装，
# ProgramStateGroup管理fork/join操作，支持分支并行执行和结果合并。

import asyncio  # 异步IO库，用于异步迭代器
import contextvars  # 上下文变量库，用于线程间上下文传递
import copy  # 深拷贝库，用于复制采样参数
import multiprocessing  # 多进程库，用于获取CPU核心数
import queue  # 队列库，用于线程间通信
import threading  # 线程库，用于后台执行和同步
import uuid  # UUID库，用于生成唯一的会话ID
import warnings  # 警告库，用于输出错误警告
from concurrent.futures import ThreadPoolExecutor  # 线程池执行器，用于批量并行执行
from contextlib import contextmanager  # 上下文管理器装饰器，用于角色作用域
from typing import Any, Callable, Dict, List, Optional  # 类型提示库

import tqdm  # 进度条库，用于显示批量执行进度

from sglang.global_config import global_config  # 全局配置，控制冗余度等设置
from sglang.lang.ir import (  # 导入SGL中间表示（IR）节点类型
    SglCommitLazy,  # 延迟提交表达式
    SglConcateAndAppend,  # 拼接追加表达式（用于fork/join合并）
    SglConstantText,  # 常量文本表达式
    SglExpr,  # SGL表达式基类
    SglExprList,  # 表达式列表
    SglGen,  # 生成表达式
    SglImage,  # 图像表达式
    SglRoleBegin,  # 角色开始表达式
    SglRoleEnd,  # 角色结束表达式
    SglSelect,  # 选择表达式
    SglSeparateReasoning,  # 分离推理表达式
    SglVariable,  # 变量引用表达式
    SglVarScopeBegin,  # 变量作用域开始表达式
    SglVarScopeEnd,  # 变量作用域结束表达式
    SglVideo,  # 视频表达式
)
from sglang.utils import (  # 导入工具函数
    encode_image_base64,  # 将图像编码为base64
    encode_video_base64,  # 将视频编码为base64
    get_exception_traceback,  # 获取异常的回溯信息
)


def run_internal(state, program, func_args, func_kwargs, sync):
    """内部运行函数，执行SGL程序并将返回值存储到状态中"""
    try:  # 尝试执行程序函数
        state.ret_value = program.func(state, *func_args, **func_kwargs)  # 调用程序函数并存储返回值
    except Exception as e:  # 捕获异常
        raise e  # 重新抛出异常
    finally:  # 无论是否异常都执行
        state.stream_executor.end()  # 结束流式执行器

    if sync:  # 如果需要同步等待
        state.stream_executor.sync()  # 等待所有任务完成

    if global_config.verbosity >= 2:  # 如果冗余度大于等于2
        print(state.text())  # 打印状态文本


def run_program(
    program,
    backend,
    func_args,
    func_kwargs,
    default_sampling_para,
    stream,
    sync=False,
    use_thread=True,
):
    """运行单个SGL程序，创建流式执行器并执行程序函数"""
    if hasattr(backend, "endpoint"):  # 如果后端有endpoint属性
        backend = backend.endpoint  # 使用endpoint作为后端
    assert backend is not None, "Please specify a backend"  # 确保后端不为空
    func_kwargs.update(program.bind_arguments)  # 将程序的绑定参数合并到函数参数中
    stream_executor = StreamExecutor(  # 创建流式执行器
        backend,  # 后端
        func_kwargs,  # 函数关键字参数
        default_sampling_para,  # 默认采样参数
        chat_template=None,  # 聊天模板（稍后从后端获取）
        stream=stream,  # 是否流式输出
        num_api_spec_tokens=program.num_api_spec_tokens,  # API推测执行的token数
        use_thread=use_thread,  # 是否使用后台线程
    )
    state = ProgramState(stream_executor)  # 用流式执行器创建程序状态

    if stream:  # 如果是流式模式
        t = threading.Thread(  # 创建新线程
            target=run_internal, args=(state, program, func_args, func_kwargs, sync)  # 线程目标函数和参数
        )
        t.start()  # 启动线程
        return state  # 立即返回状态（后台执行中）
    else:  # 非流式模式
        run_internal(state, program, func_args, func_kwargs, sync)  # 直接在当前线程中运行
        return state  # 返回状态


def run_program_batch(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    num_threads,
    progress_bar,
    generator_style=False,
):
    """批量运行SGL程序，支持多线程并行执行和进度条显示"""
    if hasattr(backend, "endpoint"):  # 如果后端有endpoint属性
        backend = backend.endpoint  # 使用endpoint作为后端

    # Pre-cache the common prefix for a batch. The prefix is extracted by tracing the program.
    # 预缓存批次的公共前缀。前缀通过追踪程序提取。
    if global_config.enable_precache_with_tracing and len(batch_arguments) > 1:  # 如果启用预缓存且批次大小大于1
        cache_program(program, backend)  # 缓存程序的公共前缀

    # Run all programs
    # 运行所有程序
    if num_threads == "auto":  # 如果线程数为自动
        num_threads = max(96, multiprocessing.cpu_count() * 16)  # 根据CPU核心数自动设置线程数
    num_threads = min(num_threads, len(batch_arguments))  # 线程数不超过批次参数数量

    if generator_style:  # 如果使用生成器风格
        return _run_program_batch_generator(  # 返回生成器
            program,  # 程序
            backend,  # 后端
            batch_arguments,  # 批次参数
            default_sampling_para,  # 默认采样参数
            num_threads,  # 线程数
            progress_bar,  # 是否显示进度条
        )

    # Original code path when generator_style=False
    # 生成器风格为False时的原始代码路径
    if num_threads == 1:  # 单线程执行
        rets = []  # 结果列表
        if progress_bar:  # 如果显示进度条
            for arguments in tqdm.tqdm(batch_arguments):  # 带进度条遍历批次参数
                rets.append(  # 添加运行结果
                    run_program(  # 运行单个程序
                        program,  # 程序
                        backend,  # 后端
                        (),  # 无位置参数
                        arguments,  # 关键字参数
                        default_sampling_para,  # 默认采样参数
                        False,  # 非流式
                        True,  # 同步
                    )
                )
        else:  # 不显示进度条
            for arguments in batch_arguments:  # 遍历批次参数
                rets.append(  # 添加运行结果
                    run_program(  # 运行单个程序
                        program,  # 程序
                        backend,  # 后端
                        (),  # 无位置参数
                        arguments,  # 关键字参数
                        default_sampling_para,  # 默认采样参数
                        False,  # 非流式
                        True,  # 同步
                    )
                )
    else:  # 多线程执行
        if progress_bar:  # 如果显示进度条
            pbar = tqdm.tqdm(total=len(batch_arguments))  # 创建进度条

        with ThreadPoolExecutor(num_threads) as executor:  # 创建线程池
            futures = []  # 未来结果列表
            for arguments in batch_arguments:  # 遍历批次参数
                futures.append(  # 添加未来结果
                    executor.submit(  # 提交任务到线程池
                        run_program,  # 运行程序函数
                        program,  # 程序
                        backend,  # 后端
                        (),  # 无位置参数
                        arguments,  # 关键字参数
                        default_sampling_para,  # 默认采样参数
                        False,  # 非流式
                        True,  # 同步
                    )
                )
                if progress_bar:  # 如果显示进度条
                    futures[-1].add_done_callback(lambda _: pbar.update())  # 任务完成时更新进度条

            rets = [f.result() for f in futures]  # 获取所有结果
        rets[-1].sync()  # 同步最后一个结果，确保所有后端操作完成

        if progress_bar:  # 如果显示进度条
            pbar.close()  # 关闭进度条

    return rets  # 返回结果列表


def _run_program_batch_generator(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    num_threads,
    progress_bar,
):
    """Helper function that yields results one by one using chunking to avoid overwhelming ThreadPoolExecutor."""
    """辅助函数，使用分块方式逐个产生结果，避免ThreadPoolExecutor过载"""
    if num_threads == 1:  # 单线程执行
        iterator = tqdm.tqdm(batch_arguments) if progress_bar else batch_arguments  # 根据是否显示进度条选择迭代器
        for arguments in iterator:  # 遍历参数
            yield run_program(  # 生成运行结果
                program,  # 程序
                backend,  # 后端
                (),  # 无位置参数
                arguments,  # 关键字参数
                default_sampling_para,  # 默认采样参数
                False,  # 非流式
                True,  # 同步
            )
    else:  # 多线程执行
        pbar = tqdm.tqdm(total=len(batch_arguments)) if progress_bar else None  # 创建进度条

        # Process in chunks to avoid overwhelming ThreadPoolExecutor
        # 分块处理，避免ThreadPoolExecutor过载
        # Otherwise, ThreadPoolExecutor.submit will block after adding certain number of tasks
        # 否则，ThreadPoolExecutor.submit在添加一定数量的任务后会阻塞
        # so we will never reach "yield" until all tasks are done
        # 因此在所有任务完成之前，我们永远无法到达"yield"
        chunk_size = 200  # 每块大小为200个任务

        with ThreadPoolExecutor(num_threads) as executor:  # 创建线程池
            for chunk_start in range(0, len(batch_arguments), chunk_size):  # 遍历每个块的起始位置
                chunk_end = min(chunk_start + chunk_size, len(batch_arguments))  # 计算块的结束位置
                chunk_futures = []  # 当前块的未来结果列表

                # Submit chunk of tasks
                # 提交一块任务
                for i in range(chunk_start, chunk_end):  # 遍历当前块中的每个任务
                    future = executor.submit(  # 提交任务
                        run_program,  # 运行程序函数
                        program,  # 程序
                        backend,  # 后端
                        (),  # 无位置参数
                        batch_arguments[i],  # 当前参数
                        default_sampling_para,  # 默认采样参数
                        False,  # 非流式
                        True,  # 同步
                    )
                    if pbar:  # 如果显示进度条
                        future.add_done_callback(lambda _: pbar.update())  # 任务完成时更新进度条
                    chunk_futures.append(future)  # 添加到当前块的未来结果列表

                # Yield results from this chunk as they complete
                # 从当前块中逐个产生结果
                for future in chunk_futures:  # 遍历当前块的未来结果
                    yield future.result()  # 生成结果

        if pbar:  # 如果显示进度条
            pbar.close()  # 关闭进度条


def cache_program(program, backend):
    """缓存程序的公共前缀，通过追踪程序提取前缀并提交到后端"""
    from sglang.lang.tracer import extract_prefix_by_tracing  # 导入追踪提取前缀函数

    prefix = extract_prefix_by_tracing(program, backend)  # 通过追踪提取公共前缀
    if prefix and len(prefix) > 64:  # 如果前缀存在且长度大于64
        backend.cache_prefix(prefix)  # 在后端缓存前缀


_INCREMENTAL_STREAMING_META_INFO_KEYS = (  # 增量流式元信息键名元组
    "output_token_logprobs",  # 输出token的对数概率
    "output_top_logprobs",  # 输出top对数概率
    "output_token_ids_logprobs",  # 输出token ID的对数概率
)


def _merge_stream_meta_info(
    pending_meta_info: dict[str, Any] | None,
    meta_info: dict[str, Any],
) -> dict[str, Any]:
    """合并流式元信息，将待处理的元信息与新的元信息拼接"""
    if pending_meta_info is None:  # 如果没有待处理的元信息
        return meta_info  # 直接返回新元信息

    merged_meta_info = dict(meta_info)  # 复制新元信息
    for key in _INCREMENTAL_STREAMING_META_INFO_KEYS:  # 遍历增量流式元信息键
        if key not in meta_info and key not in pending_meta_info:  # 如果两边都没有该键
            continue  # 跳过
        merged_meta_info[key] = list(pending_meta_info.get(key, [])) + list(  # 拼接待处理和新元信息中的列表
            meta_info.get(key, [])  # 新元信息中的列表
        )
    return merged_meta_info  # 返回合并后的元信息


class StreamExecutor:
    """A stream executor that executes SGL expressions in a background thread."""
    """流式执行器，在后台线程中执行SGL表达式"""

    def __init__(
        self,
        backend,
        arguments,
        default_sampling_para,
        chat_template,
        stream,
        num_api_spec_tokens=None,
        use_thread=True,
    ):
        """初始化流式执行器，设置后端、参数、采样参数和线程"""
        from sglang.lang.backend.base_backend import BaseBackend  # 导入基后端类

        self.sid = uuid.uuid4().hex  # 生成唯一的会话ID
        self.backend: BaseBackend = backend  # 后端实例
        self.arguments: Dict[str, Any] = arguments  # 程序参数字典
        self.default_sampling_para = default_sampling_para  # 默认采样参数
        self.stream = stream  # 是否流式输出

        self.variables = {}  # Dict[name: str -> value: str]  # 变量字典，名称到值的映射
        self.variable_event = {}  # Dict[name: str -> event: threading.Event]  # 变量事件字典，名称到事件的映射
        self.meta_info = {}  # Dict[name: str -> info: str]  # 元信息字典，名称到信息的映射
        self.is_finished = False  # 是否已完成
        self.error_ = None  # 错误信息

        # For completion
        # 用于补全模式
        self.text_ = ""  # The full text  # 完整文本

        # For chat
        # 用于聊天模式
        self.messages_ = []  # The messages in the OpenAI API format  # OpenAI API格式的消息列表
        self.chat_template = chat_template or self.backend.get_chat_template()  # 聊天模板
        self.cur_role = None  # 当前角色
        self.cur_role_begin_pos = None  # 当前角色开始位置

        # For vision
        # 用于视觉（图像/视频）
        self.images_ = []  # 所有图像列表
        self.cur_images = []  # 当前角色中的图像列表

        # For fork/join
        # 用于分支/合并
        self.fork_start_text_pos = None  # 分支开始时的文本位置

        # For speculative execution
        # 用于推测执行
        self.num_api_spec_tokens = num_api_spec_tokens  # API推测执行的token数
        self.speculated_text = ""  # 推测生成的文本

        # Worker thread
        # 工作线程
        self.use_thread = use_thread  # 是否使用后台线程
        if self.use_thread:  # 如果使用后台线程
            self.queue = queue.Queue()  # 创建任务队列

            def _run_worker_in_context():  # 在上下文中运行工作线程
                self._thread_worker_func()  # 调用工作线程函数

            self.worker = threading.Thread(  # 创建工作线程
                target=contextvars.copy_context().run, args=(_run_worker_in_context,)  # 复制上下文并运行
            )
            self.worker.start()  # 启动工作线程

        # For streaming
        # 用于流式输出
        if stream:  # 如果是流式模式
            self.stream_text_event = threading.Event()  # 文本流式事件
            self.stream_var_event = {}  # 变量流式事件字典
        else:  # 非流式模式
            self.stream_text_event = None  # 无文本流式事件
            self.stream_var_event = None  # 无变量流式事件

    def submit(self, expr: SglExpr):
        """提交一个SGL表达式到执行队列"""
        self._init_var_event(expr)  # 初始化表达式中的变量事件

        if self.use_thread:  # 如果使用后台线程
            self.queue.put(expr)  # 将表达式放入队列
        else:  # 不使用后台线程
            self._execute(expr)  # 直接执行表达式

    def sync(self):
        """同步等待所有已提交的表达式执行完成"""
        if self.use_thread:  # 如果使用后台线程
            self.queue.join()  # 等待队列中所有任务完成

    def get_var(self, name):
        """获取变量的值，如果变量未就绪则等待"""
        if name in self.variable_event:  # 如果变量有对应的事件
            self.variable_event[name].wait()  # 等待变量就绪
        return self.variables[name]  # 返回变量值

    def set_var(self, name, value):
        """设置变量的值"""
        self.variables[name] = value  # 设置变量值

    def get_meta_info(self, name, timeout=None):
        """获取变量的元信息，支持超时等待"""
        if name in self.variable_event:  # 如果变量有对应的事件
            got = self.variable_event[name].wait(timeout)  # 等待变量就绪，带超时
            if not got:  # 如果超时
                raise TimeoutError(f"Timeout while waiting for event '{name}'")  # 抛出超时错误
        ret = self.meta_info.get(name, None)  # 获取元信息
        return ret  # 返回元信息

    def fork(
        self,
        size: int = 1,
        position_ids_offset: Optional[List[int]] = None,
    ):
        """创建多个分支执行器，复制当前状态"""
        if size > 1 and str(self.text_):  # 如果分支数大于1且有文本
            self.submit(SglCommitLazy())  # 提交延迟提交操作

        self.sync()  # 等待所有操作完成
        size = int(size)  # 确保分支数为整数

        exes = [  # 创建分支执行器列表
            StreamExecutor(  # 创建新的流式执行器
                self.backend,  # 相同的后端
                self.arguments,  # 相同的参数
                self.default_sampling_para,  # 相同的默认采样参数
                self.chat_template,  # 相同的聊天模板
                self.stream,  # 相同的流式模式
            )
            for _ in range(size)  # 创建size个执行器
        ]
        for i in range(size):  # 遍历每个分支执行器
            exes[i].variables = dict(self.variables)  # 复制变量字典
            exes[i].text_ = str(self.text_)  # 复制文本
            exes[i].messages_ = list(self.messages_)  # 复制消息列表
            exes[i].cur_role = self.cur_role  # 复制当前角色
            exes[i].cur_role_begin_pos = self.cur_role_begin_pos  # 复制当前角色开始位置
            exes[i].fork_start_text_pos = len(self.text_)  # 记录分支开始的文本位置
            exes[i].images_ = list(self.images_)  # 复制图像列表

            # TODO(ying): handle API speculative execution
            # TODO(ying): 处理API推测执行

        return exes  # 返回分支执行器列表

    def text(self):
        """获取完整文本，同步等待后返回"""
        self.sync()  # 等待所有操作完成
        return self.text_  # 返回完整文本

    def messages(self):
        """获取消息列表，同步等待后返回"""
        self.sync()  # 等待所有操作完成
        return self.messages_  # 返回消息列表

    def error(self):
        """获取错误信息，同步等待后返回"""
        self.sync()  # 等待所有操作完成
        return self.error_  # 返回错误信息

    def end(self):
        """结束流式执行器，停止工作线程并通知后端"""
        if self.use_thread:  # 如果使用后台线程
            if self.worker.is_alive():  # 如果工作线程还活着
                self.queue.put(None)  # 放入None信号以停止线程
        self.backend.end_program(self)  # 通知后端程序结束

    def _thread_worker_func(self):
        """工作线程的主循环，从队列中获取并执行表达式"""
        error = None  # 错误信息

        while True:  # 无限循环
            expr = self.queue.get()  # 从队列获取表达式
            if expr is None:  # 如果收到停止信号
                self.queue.task_done()  # 标记任务完成
                break  # 跳出循环

            try:  # 尝试执行表达式
                self._execute(expr)  # 执行表达式
            except Exception as e:  # 捕获异常
                warnings.warn(f"Error in stream_executor: {get_exception_traceback()}")  # 输出错误警告
                error = e  # 记录错误
                break  # 跳出循环
            self.queue.task_done()  # 标记任务完成
            if self.stream_text_event:  # 如果有流式文本事件
                self.stream_text_event.set()  # 通知文本已更新

        # Clean the queue and events
        # 清理队列和事件
        if error is not None:  # 如果有错误
            try:  # 尝试清理队列
                while True:  # 循环清理
                    self.queue.task_done()  # 标记任务完成
                    self.queue.get_nowait()  # 立即获取下一个任务
            except queue.Empty:  # 队列为空时退出
                pass  # 忽略异常
            for name in self.variable_event:  # 遍历所有变量事件
                self.variable_event[name].set()  # 设置事件，解除等待
            if self.stream_var_event:  # 如果有流式变量事件
                for name in self.stream_var_event:  # 遍历所有流式变量事件
                    self.stream_var_event[name].set()  # 设置事件，解除等待
            self.error_ = error  # 记录错误

        if self.stream_text_event:  # 如果有流式文本事件
            self.stream_text_event.set()  # 最终通知文本更新

        self.is_finished = True  # 标记为已完成

    def _execute(self, other):
        """根据表达式类型分发执行对应的处理方法"""
        if isinstance(other, str):  # 如果是字符串
            other = SglConstantText(other)  # 转换为常量文本表达式

        assert isinstance(other, SglExpr), f"{other}"  # 确保是SGL表达式

        if isinstance(other, SglConstantText):  # 常量文本
            self._execute_fill(other.value)  # 执行填充文本
        elif isinstance(other, SglGen):  # 生成表达式
            self._execute_gen(other)  # 执行生成
        elif isinstance(other, SglSelect):  # 选择表达式
            self._execute_select(other)  # 执行选择
        elif isinstance(other, SglExprList):  # 表达式列表
            for x in other.expr_list:  # 遍历表达式列表
                self._execute(x)  # 递归执行每个表达式
        elif isinstance(other, SglRoleBegin):  # 角色开始
            self._execute_role_begin(other)  # 执行角色开始
        elif isinstance(other, SglRoleEnd):  # 角色结束
            self._execute_role_end(other)  # 执行角色结束
        elif isinstance(other, SglImage):  # 图像表达式
            self._execute_image(other)  # 执行图像处理
        elif isinstance(other, SglVideo):  # 视频表达式
            self._execute_video(other)  # 执行视频处理
        elif isinstance(other, SglVariable):  # 变量引用
            self._execute_variable(other)  # 执行变量引用
        elif isinstance(other, SglVarScopeBegin):  # 变量作用域开始
            self._execute_var_scope_begin(other)  # 执行变量作用域开始
        elif isinstance(other, SglVarScopeEnd):  # 变量作用域结束
            self._execute_var_scope_end(other)  # 执行变量作用域结束
        elif isinstance(other, SglCommitLazy):  # 延迟提交
            self._execute_commit_lazy_operations(other)  # 执行延迟提交操作
        elif isinstance(other, SglConcateAndAppend):  # 拼接追加
            if (  # 如果启用并行编码且后端支持拼接追加
                global_config.enable_parallel_encoding
                and self.backend.support_concate_and_append
            ):
                self._execute_concatenate_and_append_kv_cache(other)  # 执行KV缓存拼接追加
            else:  # 否则
                self._execute_concatenate_and_append_text(other)  # 执行文本拼接追加
        elif isinstance(other, SglSeparateReasoning):  # 分离推理
            self._execute_separate_reasoning(other)  # 执行分离推理
        else:  # 未知类型
            raise ValueError(f"Unknown type: {type(other)}")  # 抛出值错误

    def _execute_fill(self, value: str, prefix=False):
        """执行文本填充操作，将文本追加到当前文本中"""
        value = str(value)  # 确保值是字符串

        if (  # 如果当前角色是助手，且有API推测token，且后端是聊天模型，且不是前缀填充
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
            and not prefix
        ):
            self.backend.spec_fill(value)  # 调用后端的推测填充
            return  # 直接返回

        if self.speculated_text.startswith(value):  # 如果推测文本以当前值开头
            self.speculated_text = self.speculated_text[len(value) :]  # 移除已匹配的推测文本
        else:  # 推测文本不匹配
            self.speculated_text = ""  # 清空推测文本

        self.text_ += value  # 追加文本

    def _execute_image(self, expr: SglImage):
        """执行图像表达式，将图像编码为base64并添加到当前状态"""
        path = expr.path  # 获取图像路径

        base64_data = encode_image_base64(path)  # 将图像编码为base64

        self.images_.append((path, base64_data))  # 添加到图像列表
        self.cur_images.append((path, base64_data))  # 添加到当前角色图像列表
        self.text_ += self.chat_template.image_token  # 在文本中添加图像token

    def _execute_video(self, expr: SglVideo):
        """执行视频表达式，将视频编码为base64并添加到当前状态"""
        path = expr.path  # 获取视频路径
        num_frames = expr.num_frames  # 获取帧数

        base64_data = encode_video_base64(path, num_frames)  # 将视频编码为base64

        self.images_.append((path, base64_data))  # 添加到图像列表
        self.cur_images.append((path, base64_data))  # 添加到当前角色图像列表
        self.text_ += self.chat_template.image_token  # 在文本中添加图像token

    def _spec_gen(self, sampling_params):
        """推测生成，利用之前推测的文本来加速生成"""
        stop = sampling_params.stop  # 获取停止条件
        max_new_tokens = sampling_params.max_new_tokens  # 获取最大新token数
        meta_info = {}  # 初始化元信息

        def regen():  # 重新生成推测文本
            nonlocal meta_info  # 声明元信息为外部变量

            sampling_params.max_new_tokens = max(  # 确保最大token数足够大
                sampling_params.max_new_tokens, self.num_api_spec_tokens
            )
            sampling_params.stop = None  # 临时移除停止条件
            self.speculated_text, meta_info = self.backend.generate(  # 调用后端生成
                self, sampling_params=sampling_params  # 传入采样参数
            )

        def find_stop():  # 在推测文本中查找停止位置
            if isinstance(stop, str):  # 如果停止条件是字符串
                return self.speculated_text.find(stop)  # 返回字符串位置
            elif isinstance(stop, (tuple, list)):  # 如果停止条件是列表或元组
                pos = -1  # 初始化位置为-1
                for stop_str in stop:  # 遍历所有停止字符串
                    stop_pos = self.speculated_text.find(stop_str)  # 查找位置
                    if stop_pos != -1 and (pos == -1 or stop_pos < pos):  # 如果找到更近的停止位置
                        pos = stop_pos  # 更新位置
                return pos  # 返回最近的停止位置
            else:  # 其他类型
                raise Exception("Wrong type of stop in sampling parameters.")  # 抛出异常

        if stop is None:  # 如果没有停止条件
            if len(self.speculated_text) < max_new_tokens:  # 如果推测文本不够长
                regen()  # 重新生成
            comp = self.speculated_text[:max_new_tokens]  # 截取需要的部分
            self.speculated_text = self.speculated_text[max_new_tokens:]  # 移除已使用的部分
        elif isinstance(stop, (str, list, tuple)):  # 如果有停止条件
            if self.speculated_text == "":  # 如果推测文本为空
                regen()  # 重新生成
            stop_pos = find_stop()  # 查找停止位置
            if stop_pos == -1:  # 如果没有找到停止位置
                stop_pos = min(  # 使用最大token数或推测文本长度的较小值
                    sampling_params.max_new_tokens,
                    len(self.speculated_text),
                )
            comp = self.speculated_text[:stop_pos]  # 截取停止位置前的文本
            self.speculated_text = self.speculated_text[stop_pos:]  # 移除已使用的部分
        else:  # 其他类型
            raise ValueError("Wrong type of stop in sampling parameters.")  # 抛出值错误

        return comp, meta_info  # 返回生成的文本和元信息

    def _execute_gen(self, expr: SglGen):
        """执行生成表达式，调用后端生成文本或流式生成"""
        sampling_params = self._resolve_sampling_params(expr.sampling_params)  # 解析采样参数
        name = expr.name  # 获取变量名
        if not self.stream:  # 非流式模式
            if self.num_api_spec_tokens is None:  # 如果没有API推测
                comp, meta_info = self.backend.generate(  # 直接调用后端生成
                    self,  # 传入执行器自身
                    sampling_params=sampling_params,  # 传入采样参数
                )

            else:  # 有API推测
                if self.backend.is_chat_model:  # 如果是聊天模型
                    # Speculative execution on models with only chat interface.
                    # 仅聊天接口模型上的推测执行。
                    # Store the calls into a temporary list.
                    # 将调用存储到临时列表中。
                    # They will be lazily executed later.
                    # 它们稍后将延迟执行。
                    comp, meta_info = self.backend.generate(  # 调用后端生成
                        self,  # 传入执行器自身
                        sampling_params=sampling_params,  # 传入采样参数
                        spec_var_name=name,  # 传入推测变量名
                    )
                    return  # 延迟执行，直接返回

                else:  # Speculative execution on models with completion interface  # 补全接口模型上的推测执行
                    comp, meta_info = self._spec_gen(sampling_params)  # 调用推测生成
            if isinstance(comp, list):  # 如果结果是列表
                self.text_ += comp[0]  # 追加第一个元素到文本
            else:  # 结果是字符串
                assert isinstance(comp, str)  # 确保是字符串
                self.text_ += comp  # 追加到文本

            self.variables[name] = comp  # 存储变量值
            self.meta_info[name] = meta_info  # 存储元信息
            self.variable_event[name].set()  # 通知变量已就绪
        else:  # 流式模式
            assert (  # 确保流式模式不支持API推测
                self.num_api_spec_tokens is None
            ), "stream is not supported with api speculative execution"  # 流式不支持API推测执行
            generator = self.backend.generate_stream(  # 获取流式生成器
                self, sampling_params=sampling_params  # 传入执行器和采样参数
            )

            self.variables[name] = ""  # 初始化变量为空字符串
            self.stream_var_event[name].set()  # 通知流式变量已创建

            for comp, meta_info in generator:  # 遍历流式生成结果
                self.text_ += comp  # 追加文本
                self.variables[name] += comp  # 追加变量值
                self.meta_info[name] = meta_info  # 更新元信息
                self.stream_var_event[name].set()  # 通知流式变量已更新
                self.stream_text_event.set()  # 通知文本已更新

            self.variable_event[name].set()  # 通知变量已完成
            self.stream_var_event[name].set()  # 通知流式变量已完成

    def _execute_select(self, expr: SglSelect):
        """执行选择表达式，从候选选项中选择一个"""
        choices_decision = self.backend.select(  # 调用后端选择
            self, expr.choices, expr.temperature, expr.choices_method  # 传入选项、温度和选择方法
        )
        if expr.name is not None:  # 如果有变量名
            name = expr.name  # 获取变量名
            self.variables[name] = choices_decision.decision  # 存储选择结果
            self.meta_info[name] = choices_decision.meta_info  # 存储元信息
            self.variable_event[name].set()  # 通知变量已就绪
            if self.stream_var_event:  # 如果有流式变量事件
                self.stream_var_event[name].set()  # 通知流式变量已更新
        self.text_ += choices_decision.decision  # 追加选择结果到文本

    def _execute_variable(self, expr: SglVariable):
        """执行变量引用表达式，从源执行器获取变量值并填充"""
        src_executor = expr.source_stream_executor  # 获取源执行器
        value = src_executor.get_var(expr.name)  # 从源执行器获取变量值
        self._execute_fill(value)  # 填充变量值

    def _execute_role_begin(self, expr: SglRoleBegin):
        """执行角色开始表达式，设置当前角色并添加前缀"""
        assert self.cur_role is None, "Nested roles are not allowed."  # 不允许嵌套角色

        if len(self.messages_) == 0 and expr.role != "system":  # 如果没有消息且角色不是system
            # Insert the default system message
            # 插入默认系统消息
            default_system = self.chat_template.default_system_prompt  # 获取默认系统提示
            if default_system:  # 如果有默认系统提示
                self._execute_role_begin(SglRoleBegin("system"))  # 递归执行system角色开始
                self._execute_fill(default_system)  # 填充默认系统提示
                self._execute_role_end(SglRoleEnd("system"))  # 执行system角色结束

        self.cur_role = expr.role  # 设置当前角色

        prefix, _ = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)  # 获取角色前缀

        self._execute_fill(prefix, prefix=True)  # 填充前缀
        self.cur_role_begin_pos = len(self.text_)  # 记录角色内容开始位置

    def _execute_role_end(self, expr: SglRoleEnd):
        """执行角色结束表达式，添加后缀并构建消息"""
        if (  # 如果当前角色是助手，且有API推测，且是聊天模型
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
        ):
            # Execute the stored lazy generation calls
            # 执行存储的延迟生成调用
            self.backend.role_end_generate(self)  # 调用后端的角色结束生成
        self.cur_role = None  # 清除当前角色

        new_text = self.text_[self.cur_role_begin_pos :].lstrip()  # 获取角色内容并去除前导空白

        _, suffix = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)  # 获取角色后缀
        self._execute_fill(suffix)  # 填充后缀

        if self.cur_images:  # 如果当前角色有图像
            # OpenAI vision API format
            # OpenAI视觉API格式
            last_msg = {  # 构建消息字典
                "role": expr.role,  # 角色名
                "content": [{"type": "text", "text": new_text}],  # 文本内容
            }
            for image_path, image_base64_data in self.cur_images:  # 遍历当前图像
                last_msg["content"].append(  # 追加图像内容
                    {
                        "type": "image_url",  # 图像URL类型
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64_data}"  # base64编码的图像URL
                        },
                    }
                )
            self.messages_.append(last_msg)  # 添加消息到消息列表
            self.cur_images = []  # 清空当前图像列表
        else:  # 没有图像
            # OpenAI chat API format
            # OpenAI聊天API格式
            self.messages_.append({"role": expr.role, "content": new_text})  # 添加纯文本消息

    def _execute_var_scope_begin(self, expr: SglVarScopeBegin):
        """执行变量作用域开始，记录当前文本位置作为变量值"""
        self.variables[expr.name] = int(len(self.text_))  # 将当前文本长度存为变量值

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        """执行变量作用域结束，截取从开始位置到当前的文本作为变量值"""
        self.variables[expr.name] = self.text_[self.variables[expr.name] :]  # 截取文本作为变量值
        self.variable_event[expr.name].set()  # 通知变量已就绪

    def _execute_commit_lazy_operations(self, expr: SglCommitLazy):
        """执行延迟提交操作，将延迟的操作提交到后端"""
        self.backend.commit_lazy_operations(self)  # 调用后端提交延迟操作

    def _execute_concatenate_and_append_text(self, expr: SglConcateAndAppend):
        """执行文本拼接追加，将分支状态的文本拼接到当前状态"""
        new_text = ""  # 初始化新文本
        for s in expr.states:  # 遍历分支状态
            exe = s.stream_executor  # 获取流式执行器
            exe.sync()  # 等待执行完成
            new_text += exe.text_[exe.fork_start_text_pos :]  # 拼接分支后的文本

        self._execute_fill(new_text)  # 填充拼接后的文本

    def _execute_concatenate_and_append_kv_cache(self, expr: SglConcateAndAppend):
        """执行KV缓存拼接追加，将分支状态的KV缓存合并到当前状态"""
        self_len = len(self.text_)  # 获取当前文本长度

        for i, s in enumerate(expr.states):  # 遍历分支状态
            exe = s.stream_executor  # 获取流式执行器
            exe.submit(SglCommitLazy())  # 提交延迟提交操作

        for i, s in enumerate(expr.states):  # 遍历分支状态
            exe = s.stream_executor  # 获取流式执行器
            exe.sync()  # 等待执行完成
            assert exe.fork_start_text_pos == self_len  # 确保分支开始位置正确
            self.text_ += exe.text_[exe.fork_start_text_pos :]  # 追加分支文本

        src_rids = [state.stream_executor.sid for state in expr.states]  # 获取所有分支的会话ID
        self.backend.concatenate_and_append(src_rids, self.sid)  # 调用后端合并KV缓存

    def _execute_separate_reasoning(self, expr: SglSeparateReasoning):
        """执行分离推理，将推理过程从生成文本中分离出来"""
        if self.stream:  # 如果是流式模式
            # separate reasoning for stream is not supported
            # 流式模式不支持分离推理
            return  # 直接返回

        if (  # 如果当前角色是助手，且有API推测，且是聊天模型
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
        ):
            # Execute the stored lazy generation calls
            # 执行存储的延迟生成调用
            self.backend.role_end_generate(self)  # 调用后端的角色结束生成

        from sglang.srt.parser.reasoning_parser import ReasoningParser  # 导入推理解析器

        reasoning_parser = ReasoningParser(expr.model_type)  # 创建推理解析器
        other = expr.expr  # 获取内部表达式
        if not other:  # 如果没有内部表达式
            return  # 直接返回
        elif isinstance(other, SglGen) or isinstance(other, SglSelect):  # 如果是生成或选择表达式
            cur_text = self.get_var(other.name)  # 获取变量值
            reasoning, normal_text = reasoning_parser.parse_non_stream(cur_text)  # 解析推理和普通文本
            reasoning_name = expr.process_name_for_reasoning(other.name)  # 获取推理变量名
            self.set_var(other.name, normal_text)  # 设置普通文本变量
            self.set_var(reasoning_name, reasoning)  # 设置推理变量
            # the variable is ready to be used
            # 变量已就绪可以使用
            self.variable_event[reasoning_name].set()  # 通知推理变量已就绪
            self.text_ = self.text_[: self.cur_role_begin_pos] + normal_text  # 更新文本，移除推理部分
        elif isinstance(other, SglExprList):  # 如果是表达式列表
            for x in other.expr_list:  # 遍历表达式列表
                self._execute_separate_reasoning(  # 递归执行分离推理
                    SglSeparateReasoning(expr.model_type, x)  # 创建新的分离推理表达式
                )

    def _init_var_event(self, expr):
        """初始化表达式中变量的等待事件"""
        if isinstance(  # 如果是生成、选择、变量作用域开始或分离推理表达式
            expr, (SglGen, SglSelect, SglVarScopeBegin, SglSeparateReasoning)
        ):
            self.variable_event[expr.name] = threading.Event()  # 创建变量事件
            if self.stream:  # 如果是流式模式
                self.stream_var_event[expr.name] = threading.Event()  # 创建流式变量事件
        elif isinstance(expr, SglExprList):  # 如果是表达式列表
            for e in expr.expr_list:  # 遍历表达式列表
                self._init_var_event(e)  # 递归初始化变量事件

    def _resolve_sampling_params(self, sampling_params):
        """
        Construct sampling param based on default + override values
        根据默认值和覆盖值构建采样参数

        The default values of sampling are populated in `default_sampling_para` via sgl.function.run(...sampling_args)
        , and `sampling_params` contains the override values from sgl.gen().
        采样参数的默认值通过sgl.function.run(...sampling_args)填充到default_sampling_para中，
        sampling_params包含来自sgl.gen()的覆盖值。

        Here we use default_sampling_para as the base and override the values if they exist in `sampling_params`.
        这里我们以default_sampling_para为基础，如果sampling_params中存在值则覆盖。
        It also extends the stop tokens based on the chat template.
        同时还根据聊天模板扩展停止token。
        """

        # deepcopy is required because the dict has lists inside
        # 需要深拷贝，因为字典内含有列表
        clone = copy.deepcopy(self.default_sampling_para)  # 深拷贝默认采样参数

        for item in [  # 遍历所有可覆盖的参数
            "max_new_tokens",  # 最大新token数
            "min_new_tokens",  # 最小新token数
            "n",  # 生成数量
            "stop",  # 停止字符串
            "stop_token_ids",  # 停止token ID
            "stop_regex",  # 停止正则
            "temperature",  # 温度
            "top_p",  # top_p采样
            "top_k",  # top_k采样
            "min_p",  # min_p采样
            "frequency_penalty",  # 频率惩罚
            "presence_penalty",  # 存在惩罚
            "ignore_eos",  # 忽略结束符
            "return_logprob",  # 返回对数概率
            "logprob_start_len",  # 对数概率起始长度
            "top_logprobs_num",  # top对数概率数量
            "return_text_in_logprobs",  # 在对数概率中返回文本
            "dtype",  # 数据类型
            "regex",  # 正则表达式
            "json_schema",  # JSON模式
        ]:
            value = getattr(sampling_params, item, None)  # 获取覆盖值
            if value is not None:  # 如果覆盖值存在
                setattr(clone, item, value)  # 设置到克隆的参数中

        if self.chat_template.stop_str:  # 如果聊天模板有停止字符串
            if clone.stop == ():  # 如果当前停止条件是空元组
                clone.stop = []  # 转换为空列表
            elif isinstance(clone.stop, str):  # 如果当前停止条件是字符串
                clone.stop = [clone.stop]  # 转换为列表
            clone.stop += self.chat_template.stop_str  # 追加聊天模板的停止字符串

        return clone  # 返回解析后的采样参数

    def __del__(self):
        """析构函数，结束时调用end方法"""
        self.end()  # 结束流式执行器


class ProgramState:
    """The state of an SGL program."""
    """SGL程序的状态封装类，提供对流式执行器的高级接口"""

    def __init__(self, stream_executor: StreamExecutor):
        """初始化程序状态，关联流式执行器"""
        self.stream_executor = stream_executor  # 关联的流式执行器

    def _role_common(self, name: str, expr: Optional[SglExpr] = None):
        """角色操作的通用方法，支持表达式或上下文管理器两种模式"""
        if expr is not None:  # 如果提供了表达式
            role_expr = SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])  # 构建角色表达式列表
            self.stream_executor.submit(role_expr)  # 提交角色表达式
            return role_expr  # 返回角色表达式
        else:  # 没有提供表达式，使用上下文管理器

            @contextmanager  # 上下文管理器装饰器
            def role_scope():  # 定义角色作用域
                self.stream_executor.submit(SglRoleBegin(name))  # 提交角色开始
                yield  # 生成中间内容
                self.stream_executor.submit(SglRoleEnd(name))  # 提交角色结束

            return role_scope()  # 返回上下文管理器

    def system(self, expr: Optional[SglExpr] = None):
        """设置系统角色"""
        return self._role_common("system", expr)  # 调用通用角色方法

    def user(self, expr: Optional[SglExpr] = None):
        """设置用户角色"""
        return self._role_common("user", expr)  # 调用通用角色方法

    def assistant(self, expr: Optional[SglExpr] = None):
        """设置助手角色"""
        return self._role_common("assistant", expr)  # 调用通用角色方法

    @contextmanager
    def var_scope(self, name: str):
        """变量作用域上下文管理器，记录作用域内生成的文本"""
        self.stream_executor.submit(SglVarScopeBegin(name))  # 提交变量作用域开始
        yield  # 生成中间内容
        self.stream_executor.submit(SglVarScopeEnd(name))  # 提交变量作用域结束

    def fork(
        self,
        size: int = 1,
        position_ids_offset: Optional[List[int]] = None,
    ):
        """创建分支状态组，复制当前状态到多个并行分支"""
        stream_executors = self.stream_executor.fork(size, position_ids_offset)  # 创建分支执行器
        states = [ProgramState(x) for x in stream_executors]  # 用执行器创建状态列表
        state_group = ProgramStateGroup(states, self)  # 创建状态组
        return state_group  # 返回状态组

    @contextmanager
    def copy(self, position_ids_offset: Optional[List[int]] = None):
        """复制当前状态的上下文管理器，用于临时分支"""
        state_group = self.fork(1, position_ids_offset)  # 创建单个分支
        try:  # 尝试执行
            yield state_group[0]  # 生成分支状态
        finally:  # 最终执行
            state_group.join()  # 合并分支

    def text(self):
        """获取完整文本"""
        return self.stream_executor.text()  # 委托给流式执行器

    def messages(self):
        """获取消息列表"""
        return self.stream_executor.messages()  # 委托给流式执行器

    def sync(self):
        """同步等待所有操作完成"""
        return self.stream_executor.sync()  # 委托给流式执行器

    def error(self):
        """获取错误信息"""
        return self.stream_executor.error()  # 委托给流式执行器

    def text_iter(self, var_name: Optional[str] = None):
        """同步文本迭代器，逐步产生生成的文本"""
        if self.stream_executor.stream:  # 如果是流式模式
            prev = 0  # 上次读取的位置
            if var_name is None:  # 如果没有指定变量名
                event = self.stream_executor.stream_text_event  # 使用文本流式事件
                while True:  # 无限循环
                    event.wait()  # 等待事件
                    event.clear()  # 清除事件
                    out = str(self.stream_executor.text_[prev:])  # 获取新增文本
                    prev += len(out)  # 更新位置
                    if out:  # 如果有新文本
                        yield out  # 产生新文本
                    if self.stream_executor.is_finished:  # 如果已完成
                        break  # 跳出循环
            else:  # 指定了变量名
                event = None  # 初始化事件为None
                while not event:  # 等待事件创建
                    if var_name in self.stream_executor.stream_var_event:  # 如果变量有流式事件
                        event = self.stream_executor.stream_var_event[var_name]  # 获取事件
                    if self.stream_executor.is_finished:  # 如果已完成
                        yield ""  # 产生空字符串
                        return  # 返回

                while True:  # 无限循环
                    event.wait()  # 等待事件
                    event.clear()  # 清除事件
                    out = str(self.stream_executor.variables[var_name][prev:])  # 获取新增变量值
                    prev += len(out)  # 更新位置
                    if out:  # 如果有新值
                        yield out  # 产生新值
                    if self.stream_executor.variable_event[var_name].is_set():  # 如果变量已完成
                        break  # 跳出循环
        else:  # 非流式模式
            if var_name is None:  # 如果没有指定变量名
                yield self.text()  # 产生完整文本
            else:  # 指定了变量名
                yield self.get_var(var_name)  # 产生变量值

    async def text_async_iter(
        self, var_name: Optional[str] = None, return_meta_data: bool = False
    ):
        """异步文本迭代器，逐步产生生成的文本，支持返回元数据"""
        loop = asyncio.get_running_loop()  # 获取当前事件循环

        if self.stream_executor.stream:  # 如果是流式模式
            prev = 0  # 上次读取的位置
            if var_name is None:  # 如果没有指定变量名
                event = self.stream_executor.stream_text_event  # 使用文本流式事件
                while True:  # 无限循环
                    await loop.run_in_executor(None, event.wait)  # 异步等待事件
                    event.clear()  # 清除事件
                    out = str(self.stream_executor.text_[prev:])  # 获取新增文本
                    prev += len(out)  # 更新位置
                    if out:  # 如果有新文本
                        yield out  # 产生新文本
                    if self.stream_executor.is_finished:  # 如果已完成
                        break  # 跳出循环
            else:  # 指定了变量名
                event = None  # 初始化事件为None
                pending_meta_info = None  # 待处理的元信息
                while not event:  # 等待事件创建
                    if var_name in self.stream_executor.stream_var_event:  # 如果变量有流式事件
                        event = self.stream_executor.stream_var_event[var_name]  # 获取事件
                    if self.stream_executor.is_finished:  # 如果已完成
                        yield ""  # 产生空字符串
                        return  # 返回

                while True:  # 无限循环
                    await loop.run_in_executor(None, event.wait)  # 异步等待事件
                    event.clear()  # 清除事件
                    out = str(self.stream_executor.variables[var_name][prev:])  # 获取新增变量值
                    meta_info = self.stream_executor.meta_info.get(var_name)  # 获取元信息
                    prev += len(out)  # 更新位置
                    if out:  # 如果有新值
                        if return_meta_data:  # 如果需要返回元数据
                            assert meta_info is not None  # 确保元信息不为空
                            merged_meta_info = _merge_stream_meta_info(  # 合并流式元信息
                                pending_meta_info,  # 待处理的元信息
                                meta_info,  # 当前元信息
                            )
                            pending_meta_info = None  # 清空待处理元信息
                            yield out, merged_meta_info  # 产生值和元信息
                        else:  # 不需要返回元数据
                            yield out  # 产生新值
                    elif return_meta_data and meta_info is not None:  # 如果没有新值但有元数据
                        pending_meta_info = _merge_stream_meta_info(  # 合并元信息
                            pending_meta_info,  # 待处理的元信息
                            meta_info,  # 当前元信息
                        )
                    if self.stream_executor.variable_event[var_name].is_set():  # 如果变量已完成
                        break  # 跳出循环
        else:  # 非流式模式
            if var_name is None:  # 如果没有指定变量名
                yield self.text()  # 产生完整文本
            else:  # 指定了变量名
                yield self.get_var(var_name)  # 产生变量值

    def get_var(self, name):
        """获取变量值"""
        return self.stream_executor.get_var(name)  # 委托给流式执行器

    def set_var(self, name, value):
        """设置变量值"""
        return self.stream_executor.set_var(name, value)  # 委托给流式执行器

    def get_meta_info(self, name):
        """获取变量的元信息"""
        return self.stream_executor.get_meta_info(name)  # 委托给流式执行器

    def __iadd__(self, other):
        """+=运算符重载，用于提交表达式"""
        if other is None:  # 如果添加None
            raise ValueError("Tried to append None to state.")  # 抛出值错误
        self.stream_executor.submit(other)  # 提交表达式
        return self  # 返回自身

    def __getitem__(self, name):
        """[]运算符重载，用于获取变量值"""
        return self.get_var(name)  # 获取变量值

    def __setitem__(self, name, value):
        """[]=运算符重载，用于设置变量值"""
        self.set_var(name, value)  # 设置变量值

    def __contains__(self, name):
        """in运算符重载，用于检查变量是否存在"""
        return name in self.stream_executor.variables  # 检查变量是否在字典中

    def __del__(self):
        """析构函数，结束时调用流式执行器的end方法"""
        self.stream_executor.end()  # 结束流式执行器

    def __repr__(self) -> str:
        """字符串表示，显示程序状态的文本"""
        return f"ProgramState({self.text()})"  # 返回程序状态的字符串表示


class ProgramStateGroup:
    """程序状态组，管理fork产生的多个并行分支状态"""
    def __init__(
        self, states: List[ProgramState], src_state: Optional[ProgramState] = None
    ):
        """初始化程序状态组，关联分支状态列表和源状态"""
        self.states = states  # 分支状态列表
        self.src_state = src_state  # 源状态（fork的来源）

    def join(self, mode: str = "gather_variable"):
        """合并分支状态到源状态，支持变量收集和KV缓存拼接两种模式"""
        if mode == "gather_variable":  # 变量收集模式
            # Copy variables back
            # 将变量复制回源状态
            src_vars = self.src_state.stream_executor.variables  # 获取源状态变量字典
            src_var_set = set(src_vars.keys())  # 获取源状态变量名集合
            for child_state in self.states:  # 遍历每个分支状态
                child_state.stream_executor.sync()  # 等待分支执行完成
                child_vars = child_state.stream_executor.variables  # 获取分支变量字典
                new_vars = set(child_vars.keys()) - src_var_set  # 获取新增的变量名

                for k in new_vars:  # 遍历新增变量
                    if k in src_vars:  # 如果源状态已有该变量
                        src_vars[k].append(child_vars[k])  # 将分支值追加到列表
                    else:  # 源状态没有该变量
                        src_vars[k] = [child_vars[k]]  # 创建新的列表
        elif mode == "concate_and_append":  # KV缓存拼接模式
            # Concatenate and append KV cache
            # 拼接追加KV缓存
            self.src_state += SglConcateAndAppend(self.states)  # 提交拼接追加表达式
            # Need a sync here. Otherwise, `states` can be deleted.
            # 需要在此同步。否则，`states`可能被删除。
            self.src_state.stream_executor.sync()  # 同步等待拼接完成
        else:  # 无效模式
            raise ValueError(f"Invalid join mode: {mode}")  # 抛出值错误

        for s in self.states:  # 遍历每个分支状态
            s.stream_executor.end()  # 结束分支的流式执行器

    def __getitem__(self, i: int):
        """[]运算符重载，用于获取指定索引的分支状态"""
        return self.states[i]  # 返回指定索引的状态

    def __setitem__(self, i: int, value):
        """[]=运算符重载，用于设置指定索引的分支状态（需相等）"""
        assert self.states[i] == value  # 确保值相等

    def __iadd__(self, other):
        """+=运算符重载，向所有分支状态添加表达式"""
        if isinstance(other, Callable):  # 如果是可调用对象
            # lambda function
            # lambda函数
            for i in range(len(self.states)):  # 遍历每个分支
                self.states[i] += other(i)  # 调用lambda并添加结果
        elif isinstance(other, SglExpr):  # 如果是SGL表达式
            for i in range(len(self.states)):  # 遍历每个分支
                self.states[i] += other  # 向每个分支添加相同的表达式
        elif isinstance(other, (list, tuple)):  # 如果是列表或元组
            for i in range(len(self.states)):  # 遍历每个分支
                self.states[i] += other[i]  # 向每个分支添加对应的表达式
        else:  # 无效类型
            raise ValueError(f"Invalid value: {other}")  # 抛出值错误

        return self  # 返回自身
