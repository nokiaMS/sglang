# 外部语料库管理器模块
# 管理ngram投机解码的SAM外部语料库，包括添加、删除、列表操作
# 以及异步后台加载。由调度器使用，独立的管理器对象。
"""Manages external SAM corpora for ngram speculative decoding.

Handles add/remove/list operations and async background loading.
Used by the Scheduler — not a mixin, a standalone manager object.
"""

import logging  # 导入日志模块
import threading  # 导入线程模块
from typing import Callable, Optional, Tuple  # 导入类型注解

from sglang.srt.managers.io_struct import (  # 导入IO结构
    AddExternalCorpusReqInput,  # 添加外部语料库请求输入
    AddExternalCorpusReqOutput,  # 添加外部语料库请求输出
    ListExternalCorporaReqInput,  # 列出外部语料库请求输入
    ListExternalCorporaReqOutput,  # 列出外部语料库请求输出
    RemoveExternalCorpusReqInput,  # 删除外部语料库请求输入
    RemoveExternalCorpusReqOutput,  # 删除外部语料库请求输出
)

logger = logging.getLogger(__name__)  # 获取日志记录器


class ExternalCorpusManager:
    """Manages external SAM corpus lifecycle for a single scheduler.
    # 管理单个调度器的外部SAM语料库生命周期。

    Args:
    # 参数：
        draft_worker: the NGRAMWorker instance (must have add_external_corpus,
            remove_external_corpus, list_external_corpora methods).
        # draft_worker：NGRAMWorker实例（必须有add_external_corpus、
        # remove_external_corpus、list_external_corpora方法）。
        send_response: callable(output, recv_req) to send deferred responses
            back to the tokenizer manager.
        # send_response：可调用对象(output, recv_req)，用于将延迟响应发送回tokenizer管理器。
    """

    def __init__(self, draft_worker, send_response: Callable):
        # 初始化外部语料库管理器
        self._worker = draft_worker  # 草稿Worker
        self._send_response = send_response  # 发送响应的回调
        self._pending_load: Optional[  # 待处理的加载任务
            Tuple[AddExternalCorpusReqInput, threading.Thread]
        ] = None
        self._load_result: Optional[AddExternalCorpusReqOutput] = None  # 加载结果

    def check_pending_load(self):
        # 从调度器事件循环中轮询，完成时发送响应
        """Poll from the scheduler event loop. Sends response when done."""
        if self._pending_load is None:  # 无待处理加载
            return  # 直接返回
        recv_req, thread = self._pending_load  # 获取请求和线程
        if thread.is_alive():  # 线程仍在运行
            return  # 等待下次轮询
        self._pending_load = None  # 清除待处理
        thread.join()  # formal happens-before for _load_result visibility 正式确保_load_result可见性
        result = self._load_result  # 获取加载结果
        self._load_result = None  # 清除结果
        if result.success:  # 加载成功
            self._worker.commit_corpus_load(result.corpus_id, result.loaded_token_count)  # 提交加载
        self._send_response(result, recv_req)  # 发送响应

    def add(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        # 异步添加外部语料库
        if self._pending_load is not None:  # 已有加载任务在进行
            return AddExternalCorpusReqOutput(
                success=False,
                message="Another corpus load is already in progress.",  # 另一个语料库加载正在进行
            )

        def _build():  # 后台构建函数
            try:
                loaded = self._worker.add_external_corpus(  # 添加外部语料库
                    recv_req.corpus_id, recv_req.token_chunks
                )
                self._load_result = AddExternalCorpusReqOutput(  # 设置成功结果
                    success=True,
                    corpus_id=recv_req.corpus_id,
                    message=f"Loaded corpus '{recv_req.corpus_id}' with {loaded} tokens.",
                    loaded_token_count=loaded,
                )
            except Exception as e:  # 添加失败
                self._load_result = AddExternalCorpusReqOutput(
                    success=False, message=str(e)
                )

        thread = threading.Thread(target=_build, daemon=True)  # 创建守护线程
        self._pending_load = (recv_req, thread)  # 记录待处理任务
        thread.start()  # 启动线程
        return None  # response sent later by check_pending_load 稍后由check_pending_load发送响应

    # FIXME(kpham-sgl): remove a corpus during a pending load is an undefined behaviour
    # and should be explicitly prevented.
    # FIXME(kpham-sgl): 在待处理加载期间删除语料库是未定义行为，应该显式阻止。
    def remove(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        # 同步删除外部语料库
        try:
            self._worker.remove_external_corpus(recv_req.corpus_id)  # 删除语料库
            return RemoveExternalCorpusReqOutput(
                success=True,
                message=f"Removed corpus '{recv_req.corpus_id}'.",  # 删除成功
            )
        except Exception as e:  # 删除失败
            return RemoveExternalCorpusReqOutput(success=False, message=str(e))

    def list(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        # 列出所有外部语料库
        try:
            token_counts = self._worker.list_external_corpora()  # 获取语料库列表
            return ListExternalCorporaReqOutput(
                success=True,
                corpus_token_counts=token_counts,
            )
        except Exception as e:  # 获取失败
            return ListExternalCorporaReqOutput(success=False, message=str(e))
