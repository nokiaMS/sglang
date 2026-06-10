# 多HTTP工作器模式下的混入类和工具函数
# 本文件实现多进程处理请求和分词，以减少Python和HTTP服务器的开销

from __future__ import annotations  # 启用延迟类型注解求值

# Copyright 2023-2024 SGLang Team
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

"""
Mixin classes and utils for multi-http-worker mode
This file uses multiple processes to handle requests and tokenization, reducing the overhead of python and http server.
混入类和工具函数，用于多HTTP工作器模式。
本文件使用多进程来处理请求和分词，减少Python和HTTP服务器的开销。
"""

import asyncio  # 异步IO库
import logging  # 日志库
import multiprocessing as multiprocessing  # 多进程库
import os  # 操作系统接口
import pickle  # 序列化库
import signal  # 信号处理
import sys  # 系统相关
import threading  # 线程库
import zlib  # 压缩库，用于CRC32哈希
from multiprocessing import shared_memory  # 共享内存
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union  # 类型提示

import psutil  # 进程工具库
import setproctitle  # 设置进程标题
import zmq  # ZeroMQ消息队列
import zmq.asyncio  # ZeroMQ异步版本

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend  # 解聚合模式和传输后端
from sglang.srt.managers.disagg_service import start_disagg_service  # 启动解聚合服务
from sglang.srt.managers.io_struct import (  # 导入IO结构体
    BaseBatchReq,  # 批量请求基类
    BaseReq,  # 请求基类
    BatchEmbeddingOutput,  # 批量嵌入输出
    BatchStrOutput,  # 批量字符串输出
    BatchTokenIDOutput,  # 批量Token ID输出
    ContinueGenerationReqInput,  # 继续生成请求输入
    FreezeGCReq,  # 冻结GC请求
    PauseContinueBroadcast,  # 暂停/继续广播
    PauseGenerationReqInput,  # 暂停生成请求输入
    TokenizerWorkerRegistration,  # 分词工作器注册
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # 分词管理器
from sglang.srt.server_args import PortArgs, ServerArgs  # 服务器参数
from sglang.srt.utils import (  # 导入工具函数
    configure_logger,  # 配置日志
    kill_itself_when_parent_died,  # 父进程死亡时自杀
    kill_process_tree,  # 杀死进程树
)
from sglang.srt.utils.network import get_zmq_socket  # 获取ZMQ套接字
from sglang.utils import get_exception_traceback  # 获取异常回溯

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # 反分词管理器

logger = logging.getLogger(__name__)  # 获取日志记录器


class SocketMapping:  # IPC套接字映射类，管理ZMQ套接字的注册和发送
    def __init__(self):  # 初始化套接字映射
        self._zmq_context = zmq.Context()  # 创建ZMQ上下文
        self._mapping: Dict[str, zmq.Socket] = {}  # IPC名称到套接字的映射字典

    def clear_all_sockets(self):  # 关闭并清除所有已注册的套接字
        for socket in self._mapping.values():  # 遍历所有套接字
            socket.close()  # 关闭套接字
        self._mapping.clear()  # 清空映射字典

    def _register_ipc_mapping(self, ipc_name: str, is_tokenizer: bool):  # 注册新的IPC套接字映射
        type_str = "tokenizer" if is_tokenizer else "detokenizer"  # 根据是否为分词器确定类型字符串
        if ipc_name in self._mapping:  # 如果IPC名称已注册
            logger.warning(f"{type_str} already registered {ipc_name=}, skipping...")  # 已注册则跳过
            return  # 返回
        logger.info(f"Registering {type_str} {ipc_name=} in SocketMapping...")  # 记录注册信息
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)  # 创建PUSH类型ZMQ套接字
        self._mapping[ipc_name] = socket  # 将套接字存入映射

    def send_output(self, ipc_name: str, output: Any, is_tokenizer: bool = False):  # 通过IPC名称发送输出对象
        if ipc_name is None:  # 如果IPC名称为空
            # Some unhandled cases  某些未处理的情况
            logger.warning(f"IPC name is None, output type={type(output)}, skipping...")  # IPC名称为空，跳过
            return  # 返回

        if ipc_name not in self._mapping:  # 如果IPC名称未注册
            self._register_ipc_mapping(ipc_name, is_tokenizer=is_tokenizer)  # 动态注册IPC映射
        self._mapping[ipc_name].send_pyobj(output)  # 通过套接字发送Python对象


def _extract_field_by_index(
    output: Any, field_name: str, index: int, check_length: bool = True  # 是否检查长度
) -> Any:  # 根据索引从输出对象中提取字段值
    """Extract a field value from output by index, handling None and length checks.
    根据索引从输出对象中提取字段值，处理None和长度检查。

    Args:
        output: The output object containing the field  包含字段的输出对象
        field_name: The name of the field to extract  要提取的字段名称
        index: The index to access in the field list  字段列表中要访问的索引
        check_length: If True, check both field existence and length. If False, only check field existence.  如果为True，同时检查字段存在性和长度；如果为False，仅检查字段存在性。

    Returns:
        A list containing the field value at index, or None if not available.  包含指定索引处字段值的列表，如果不可用则返回None。
    """
    field = getattr(output, field_name, None)  # 获取输出对象的指定字段
    if field is None:  # 如果字段为None
        return None  # 返回None

    if isinstance(field, dict):  # 如果字段是字典类型
        new_field = {}  # 创建新字典
        for k, v in field.items():  # 遍历字典项
            new_field[k] = v[index] if len(v) > index else None  # 按索引提取值，越界则为None
        return new_field  # 返回新字典

    if check_length:  # 如果需要检查长度
        if len(field) <= index:  # 如果字段长度不足
            return None  # 返回None

    return [field[index]]  # 返回包含索引处值的列表


def _handle_output_by_index(output, i):  # 按索引拆分批量输出，提取第i个请求的结果
    """NOTE: A maintainable method is better here.  注意：这里最好使用更可维护的方法。"""
    if isinstance(output, BatchTokenIDOutput):  # 如果是批量TokenID输出
        new_output = BatchTokenIDOutput(  # 创建新的批量TokenID输出对象
            rids=[output.rids[i]],  # 请求ID列表，取第i个
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),  # 推测验证计数
            spec_num_correct_drafts=_extract_field_by_index(  # 推测正确草稿数
                output, "spec_num_correct_drafts", i
            ),
            spec_correct_drafts_histogram=_extract_field_by_index(  # 推测正确草稿直方图
                output, "spec_correct_drafts_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),  # 时间统计
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),  # 完成原因
            decoded_texts=_extract_field_by_index(output, "decoded_texts", i),  # 解码后的文本
            decode_ids=_extract_field_by_index(output, "decode_ids", i),  # 解码ID
            read_offsets=_extract_field_by_index(output, "read_offsets", i),  # 读取偏移量
            output_ids=_extract_field_by_index(output, "output_ids", i),  # 输出ID
            skip_special_tokens=_extract_field_by_index(  # 是否跳过特殊Token
                output, "skip_special_tokens", i
            ),
            spaces_between_special_tokens=_extract_field_by_index(  # 特殊Token之间是否加空格
                output, "spaces_between_special_tokens", i
            ),
            no_stop_trim=_extract_field_by_index(output, "no_stop_trim", i),  # 是否不裁剪停止词
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),  # 提示Token数
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),  # 补全Token数
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),  # 推理Token数
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),  # 缓存Token数
            cached_tokens_details=_extract_field_by_index(  # 缓存Token详情
                output, "cached_tokens_details", i
            ),
            input_token_logprobs_val=_extract_field_by_index(  # 输入Token对数概率值
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(  # 输入Token对数概率索引
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(  # 输出Token对数概率值
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(  # 输出Token对数概率索引
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(  # 输入Top对数概率值
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(  # 输入Top对数概率索引
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(  # 输出Top对数概率值
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(  # 输出Top对数概率索引
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(  # 输入Token ID对数概率值
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(  # 输入Token ID对数概率索引
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(  # 输出Token ID对数概率值
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(  # 输出Token ID对数概率索引
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(  # 输出Token熵值
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(  # 输出隐藏状态
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(  # 路由专家
                output, "routed_experts", i, check_length=False
            ),
            indexer_topk=_extract_field_by_index(  # 索引器TopK
                output, "indexer_topk", i, check_length=False
            ),
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),  # 回缩计数
            placeholder_tokens_idx=None,  # 占位Token索引，设为None
            placeholder_tokens_val=None,  # 占位Token值，设为None
            token_steps=_extract_field_by_index(  # Token步数
                output, "token_steps", i, check_length=False
            ),
            customized_info=_extract_field_by_index(  # 自定义信息
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),  # 数据并行排名
        )
    elif isinstance(output, BatchEmbeddingOutput):  # 如果是批量嵌入输出
        new_output = BatchEmbeddingOutput(  # 创建新的批量嵌入输出对象
            rids=[output.rids[i]],  # 请求ID列表，取第i个
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),  # 完成原因
            embeddings=_extract_field_by_index(output, "embeddings", i),  # 嵌入向量
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),  # 提示Token数
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),  # 缓存Token数
            placeholder_tokens_idx=None,  # 占位Token索引，设为None
            placeholder_tokens_val=None,  # 占位Token值，设为None
        )
    elif isinstance(output, BatchStrOutput):  # 如果是批量字符串输出
        new_output = BatchStrOutput(  # 创建新的批量字符串输出对象
            rids=[output.rids[i]],  # 请求ID列表，取第i个
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),  # 推测验证计数
            spec_num_correct_drafts=_extract_field_by_index(  # 推测正确草稿数
                output, "spec_num_correct_drafts", i
            ),
            spec_correct_drafts_histogram=_extract_field_by_index(  # 推测正确草稿直方图
                output, "spec_correct_drafts_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),  # 时间统计
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),  # 完成原因
            output_strs=_extract_field_by_index(output, "output_strs", i),  # 输出字符串
            output_ids=_extract_field_by_index(output, "output_ids", i),  # 输出ID
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),  # 提示Token数
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),  # 补全Token数
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),  # 推理Token数
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),  # 缓存Token数
            cached_tokens_details=_extract_field_by_index(  # 缓存Token详情
                output, "cached_tokens_details", i
            ),
            input_token_logprobs_val=_extract_field_by_index(  # 输入Token对数概率值
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(  # 输入Token对数概率索引
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(  # 输出Token对数概率值
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(  # 输出Token对数概率索引
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(  # 输入Top对数概率值
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(  # 输入Top对数概率索引
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(  # 输出Top对数概率值
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(  # 输出Top对数概率索引
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(  # 输入Token ID对数概率值
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(  # 输入Token ID对数概率索引
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(  # 输出Token ID对数概率值
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(  # 输出Token ID对数概率索引
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(  # 输出Token熵值
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(  # 输出隐藏状态
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(  # 路由专家
                output, "routed_experts", i, check_length=False
            ),
            indexer_topk=_extract_field_by_index(  # 索引器TopK
                output, "indexer_topk", i, check_length=False
            ),
            customized_info=_extract_field_by_index(  # 自定义信息
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),  # 数据并行排名
            placeholder_tokens_idx=None,  # 占位Token索引，设为None
            placeholder_tokens_val=None,  # 占位Token值，设为None
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),  # 回缩计数
            token_steps=_extract_field_by_index(  # Token步数
                output, "token_steps", i, check_length=False
            ),
        )
    else:  # 其他类型的输出
        new_output = output  # 直接使用原输出
    return new_output  # 返回处理后的输出


class MultiHttpWorkerDetokenizerMixin:  # 多HTTP工作器反分词混入类，用于反分词管理器
    """Mixin class for DetokenizerManager  DetokenizerManager的混入类"""

    def maybe_clear_socket_mapping(self: DetokenizerManager):  # 如果存在套接字映射则清除
        if hasattr(self, "socket_mapping"):  # 如果有socket_mapping属性
            self.socket_mapping.clear_all_sockets()  # 清除所有套接字

    def multi_http_worker_event_loop(self: DetokenizerManager):  # 多HTTP工作器模式下的事件循环
        """The event loop that handles requests, for multi multi-http-worker mode  处理请求的事件循环，用于多多HTTP工作器模式"""
        self.socket_mapping = SocketMapping()  # 创建新的套接字映射
        while True:  # 无限循环
            recv_obj = self.recv_from_scheduler.recv_pyobj()  # 从调度器接收Python对象
            output = self._request_dispatcher(recv_obj)  # 请求分发器处理
            if output is None:  # 如果输出为空
                continue  # 跳过本次循环

            # Fan out the output back to the originating tokenizer worker(s).
            # In multi-detokenizer mode the upstream MultiDetokenizerRouter may
            # forward either batched or single requests, so handle both shapes.
            # 将输出扇出回原始的分词工作器。
            # 在多反分词模式下，上游MultiDetokenizerRouter可能转发批量或单个请求，因此处理两种形式。
            if isinstance(recv_obj, BaseBatchReq):  # 如果是批量请求
                for i, ipc_name in enumerate(recv_obj.http_worker_ipcs):  # 遍历每个HTTP工作器IPC名称
                    new_output = _handle_output_by_index(output, i)  # 按索引提取单个输出
                    self.socket_mapping.send_output(  # 发送输出到对应的分词工作器
                        ipc_name, new_output, is_tokenizer=True
                    )
            elif isinstance(recv_obj, BaseReq):  # 如果是单个请求
                self.socket_mapping.send_output(  # 直接发送输出到对应的分词工作器
                    recv_obj.http_worker_ipc, output, is_tokenizer=True
                )
            else:  # 其他类型则报错
                raise ValueError(  # 抛出值错误
                    f"multi_http_worker_event_loop got unexpected req type {type(recv_obj)}"
                )


class MultiTokenizerRouter:  # 多分词器路由器，在分词管理器和调度器/反分词管理器之间路由消息
    """A router between tokenizer managers and the scheduler/detokenizer manager.
    分词管理器和调度器/反分词管理器之间的路由器。

    Forward: tokenizer managers → router → scheduler.  正向：分词管理器 → 路由器 → 调度器
    Backward: detokenizer manager → router → tokenizer managers.  反向：反分词管理器 → 路由器 → 分词管理器
    Also broadcasts pause/continue to all tokenizer managers for consistent is_pause state.
    同时向所有分词管理器广播暂停/继续，以保持一致的is_pause状态。
    """

    def __init__(  # 初始化多分词器路由器
        self,
        server_args: ServerArgs,  # 服务器参数
        port_args: PortArgs,  # 端口参数
    ):
        self.server_args = server_args  # 保存服务器参数
        context = zmq.asyncio.Context(3)  # 创建异步ZMQ上下文，3个IO线程
        self.recv_from_detokenizer = get_zmq_socket(  # 从反分词器接收消息的套接字
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(  # 发送到调度器的套接字
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.receive_from_worker = get_zmq_socket(  # 从工作器接收消息的套接字
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, True
        )
        self._loop = asyncio.new_event_loop()  # 创建新的事件循环
        self._thread = threading.Thread(target=self._run_loop, daemon=True)  # 创建守护线程运行事件循环
        self._thread.start()  # 启动线程
        self._task = asyncio.run_coroutine_threadsafe(  # 在事件循环中运行路由工作协程
            self.router_worker_obj(), self._loop
        )
        self._handle_task = asyncio.run_coroutine_threadsafe(  # 在事件循环中运行处理循环协程
            print_exception_wrapper(self.handle_loop), self._loop
        )
        self.disaggregation_bootstrap_server = start_disagg_service(self.server_args)  # 启动解聚合引导服务

        # Worker IPC names for pause/continue broadcasting  工作器IPC名称，用于暂停/继续广播
        self.all_worker_ipcs: set[str] = set()  # 所有已注册工作器的IPC名称集合
        # Shared socket mapping (both coroutines run on self._loop, so safe)  共享套接字映射（两个协程都在self._loop上运行，因此线程安全）
        self.socket_mapping = SocketMapping()  # 创建套接字映射

    def _run_loop(self):  # 运行事件循环
        self._loop.run_forever()  # 永久运行事件循环

    async def router_worker_obj(self):  # 正向路径：工作器 → 调度器，并处理暂停/继续广播
        """Forward path: workers → scheduler, with pause/continue broadcast.  正向路径：工作器 → 调度器，带暂停/继续广播。"""
        while True:  # 无限循环
            recv_obj = await self.receive_from_worker.recv_pyobj()  # 异步接收来自工作器的对象

            if isinstance(recv_obj, TokenizerWorkerRegistration):  # 如果是工作器注册消息
                if recv_obj.worker_ipc_name not in self.all_worker_ipcs:  # 如果工作器IPC名称未注册
                    self.all_worker_ipcs.add(recv_obj.worker_ipc_name)  # 添加到已注册集合
                    logger.info(  # 记录注册信息
                        f"Router registered worker IPC: {recv_obj.worker_ipc_name} "
                        f"(total: {len(self.all_worker_ipcs)})"
                    )
                continue  # 跳过后续处理

            if isinstance(  # 如果是暂停/继续生成请求
                recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
            ):
                # Broadcast to ALL workers so every worker's is_pause is set  广播到所有工作器，以设置每个工作器的is_pause状态
                is_pause = isinstance(recv_obj, PauseGenerationReqInput)  # 判断是否为暂停请求
                broadcast = PauseContinueBroadcast(is_pause=is_pause)  # 创建暂停/继续广播对象
                for ipc_name in self.all_worker_ipcs:  # 遍历所有工作器
                    self.socket_mapping.send_output(ipc_name, broadcast)  # 发送广播
                # Forward to scheduler rank 0 (it broadcasts to all TP/PP/DP
                # ranks internally). Skip for abort mode which drains via polling.
                # 转发到调度器rank 0（它在内部广播到所有TP/PP/DP rank）。跳过中止模式，该模式通过轮询排空。
                if not (  # 如果不是中止模式的暂停请求
                    isinstance(recv_obj, PauseGenerationReqInput)
                    and recv_obj.mode == "abort"
                ):
                    await self.send_to_scheduler.send_pyobj(recv_obj)  # 转发到调度器
                continue  # 跳过后续处理

            await self.send_to_scheduler.send_pyobj(recv_obj)  # 其他消息直接转发到调度器

    async def handle_loop(self):  # 反向路径：反分词器 → 路由结果到正确的工作器
        """Backward path: detokenizer → route results to correct worker.  反向路径：反分词器 → 将结果路由到正确的工作器。"""
        while True:  # 无限循环
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()  # 异步接收来自反分词器的对象
            await self._distribute_result_to_workers(recv_obj)  # 将结果分发到对应的工作器

    async def _distribute_result_to_workers(self, recv_obj):  # 将接收到的结果分发到对应的工作器
        if isinstance(recv_obj, BaseReq):  # 如果是单个请求
            ipc_names = [recv_obj.http_worker_ipc]  # 获取对应的IPC名称
        elif isinstance(recv_obj, BaseBatchReq):  # 如果是批量请求
            ipc_names = recv_obj.http_worker_ipcs  # 获取所有IPC名称列表
        else:  # 其他类型则报错
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):  # 遍历每个IPC名称
            new_recv_obj = _handle_output_by_index(recv_obj, i)  # 按索引提取单个结果
            self.socket_mapping.send_output(ipc_name, new_recv_obj)  # 发送到对应工作器


class MultiDetokenizerRouter:  # 多反分词器路由器，将调度器输出路由到N个反分词管理器工作器之一
    """Route scheduler outputs to one of N DetokenizerManager workers.
    将调度器输出路由到N个DetokenizerManager工作器之一。

    Each request is pinned to a worker by hashing its ``http_worker_ipc`` with
    ``zlib.crc32`` (deterministic across runs), so all outputs of the same rid
    always land on the same detokenizer and ``decode_status`` stays consistent.
    每个请求通过使用``zlib.crc32``对其``http_worker_ipc``进行哈希（跨运行确定性）来固定到某个工作器，
    因此同一rid的所有输出总是落在同一个反分词器上，``decode_status``保持一致。
    """

    def __init__(self, ipc_name_list: List[str], port_args: PortArgs):  # 初始化多反分词器路由器
        self.ipc_name_list = ipc_name_list  # 保存IPC名称列表
        self.num_workers = len(ipc_name_list)  # 工作器数量
        self.socket_mapping = SocketMapping()  # 创建套接字映射
        context = zmq.Context(2)  # 创建ZMQ上下文，2个IO线程
        self.recv_from_scheduler = get_zmq_socket(  # 从调度器接收消息的套接字
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )

    def _pick(self, key: str) -> str:  # 根据键的CRC32哈希值选择目标工作器IPC名称
        return self.ipc_name_list[zlib.crc32(key.encode()) % self.num_workers]  # 哈希取模选择

    def _send(self, ipc_name: str, obj: Any) -> None:  # 发送对象到指定的反分词器工作器
        self.socket_mapping.send_output(ipc_name, obj, is_tokenizer=False)  # 标记为非分词器

    def event_loop(self):  # 事件循环，处理从调度器接收到的消息
        while True:  # 无限循环
            recv_obj = self.recv_from_scheduler.recv_pyobj()  # 从调度器接收Python对象

            # FreezeGCReq must freeze every detokenizer process.  FreezeGCReq必须冻结每个反分词器进程。
            if isinstance(recv_obj, FreezeGCReq):  # 如果是冻结GC请求
                for ipc in self.ipc_name_list:  # 遍历所有工作器
                    self._send(ipc, recv_obj)  # 广播冻结GC请求
                continue  # 跳过后续处理

            # Single request: route by its own http_worker_ipc.  单个请求：按其自身的http_worker_ipc路由。
            if isinstance(recv_obj, BaseReq):  # 如果是单个请求
                assert (  # 断言http_worker_ipc不为空
                    recv_obj.http_worker_ipc is not None
                ), f"Single req {recv_obj.rid=} missing http_worker_ipc"
                self._send(self._pick(recv_obj.http_worker_ipc), recv_obj)  # 按哈希路由
                continue  # 跳过后续处理

            # Batch request.  批量请求。
            if isinstance(recv_obj, BaseBatchReq):  # 如果是批量请求
                # Idle/no-op batch (rids=[]): broadcast to all detokenizers  空闲/空操作批量（rids=[]）：广播到所有反分词器
                if not recv_obj.rids:  # 如果没有请求ID
                    for ipc in self.ipc_name_list:  # 遍历所有工作器
                        self._send(ipc, recv_obj)  # 广播到所有工作器
                    continue  # 跳过后续处理

                ipcs = recv_obj.http_worker_ipcs  # 获取HTTP工作器IPC列表
                assert (  # 断言IPC列表有效
                    ipcs is not None
                    and len(ipcs) == len(recv_obj.rids)
                    and all(x is not None for x in ipcs)
                ), f"Batch req {recv_obj.rids=} has invalid http_worker_ipcs"

                # Split per-item and route each by its own ipc.  按项拆分，按各自的IPC路由。
                for i, ipc_key in enumerate(ipcs):  # 遍历每个IPC键
                    one = _handle_output_by_index(recv_obj, i)  # 按索引提取单个请求
                    if one is recv_obj:  # 如果未能拆分
                        raise TypeError(f"Cannot split {type(recv_obj)}")  # 抛出类型错误
                    one.http_worker_ipcs = [ipc_key]  # 设置单个请求的IPC列表
                    self._send(self._pick(ipc_key), one)  # 按哈希路由
                continue  # 跳过后续处理

            raise ValueError(  # 不支持的类型则抛出值错误
                f"MultiDetokenizerRouter got unsupported type {type(recv_obj)}"
            )


def run_multi_detokenizer_router_process(  # 运行多反分词器路由器进程
    ipc_name_list: List[str],  # IPC名称列表
    server_args: ServerArgs,  # 服务器参数
    port_args: PortArgs,  # 端口参数
):
    kill_itself_when_parent_died()  # 父进程死亡时自动终止
    setproctitle.setproctitle("sglang::detokenizer_router")  # 设置进程标题
    configure_logger(server_args)  # 配置日志
    parent_process = psutil.Process().parent()  # 获取父进程

    router = None  # 路由器初始化为None
    try:  # 尝试运行路由器
        router = MultiDetokenizerRouter(ipc_name_list, port_args)  # 创建多反分词器路由器
        router.event_loop()  # 运行事件循环
    except Exception:  # 捕获异常
        traceback = get_exception_traceback()  # 获取异常回溯
        logger.error(f"MultiDetokenizerRouter hit an exception: {traceback}")  # 记录错误日志
        if router is not None:  # 如果路由器已创建
            router.socket_mapping.clear_all_sockets()  # 清除所有套接字
        parent_process.send_signal(signal.SIGQUIT)  # 向父进程发送SIGQUIT信号


class TokenizerWorker(TokenizerManager):  # 分词工作器，继承自分词管理器，用于多HTTP工作器模式
    """Tokenizer Worker in multi-http-worker mode  多HTTP工作器模式下的分词工作器"""

    def __init__(  # 初始化分词工作器
        self,
        server_args: ServerArgs,  # 服务器参数
        port_args: PortArgs,  # 端口参数
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")  # 设置进程标题
        # prevent init prefill bootstrapserver again  防止再次初始化预填充引导服务器
        disaggregation_mode = server_args.disaggregation_mode  # 保存解聚合模式
        server_args.disaggregation_mode = "null"  # 临时设为null以跳过引导服务器初始化
        super().__init__(server_args, port_args)  # 调用父类初始化

        self.worker_id = os.getpid()  # 工作器ID为当前进程ID
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name  # 保存分词器IPC名称

        # For PD disaggregtion  用于PD解聚合
        self.server_args.disaggregation_mode = disaggregation_mode  # 恢复解聚合模式
        self.disaggregation_mode = DisaggregationMode(  # 设置解聚合模式枚举
            self.server_args.disaggregation_mode
        )
        self.disaggregation_transfer_backend = TransferBackend(  # 设置解聚合传输后端枚举
            self.server_args.disaggregation_transfer_backend
        )

        # Register this worker with the router for pause/continue broadcasting  向路由器注册此工作器，用于暂停/继续广播
        reg = TokenizerWorkerRegistration(worker_ipc_name=self.tokenizer_ipc_name)  # 创建注册消息
        self.send_to_scheduler.send_pyobj(reg)  # 发送注册消息到调度器

        # Future for awaiting pause/continue broadcast confirmation  用于等待暂停/继续广播确认的Future
        self._pause_continue_future: Optional[asyncio.Future] = None  # 暂停/继续Future，初始为None

        # Register PauseContinueBroadcast in the result dispatcher so
        # handle_loop routes it to _handle_pause_continue_broadcast
        # 在结果分发器中注册PauseContinueBroadcast，以便handle_loop将其路由到_handle_pause_continue_broadcast
        from sglang.utils import TypeBasedDispatcher  # 导入类型分发器

        self._result_dispatcher += TypeBasedDispatcher(  # 添加暂停/继续广播处理器到结果分发器
            [(PauseContinueBroadcast, self._handle_pause_continue_broadcast)]
        )

    async def pause_generation(self, obj: PauseGenerationReqInput):  # 暂停生成
        loop = asyncio.get_event_loop()  # 获取当前事件循环
        self._pause_continue_future = loop.create_future()  # 创建Future用于等待广播确认
        # Send to router which will broadcast to all workers  发送到路由器，路由器将广播到所有工作器
        # (router also handles forwarding to scheduler for non-abort modes)  （路由器还负责在非中止模式下转发到调度器）
        self.send_to_scheduler.send_pyobj(obj)  # 发送暂停生成请求
        await self._pause_continue_future  # 等待广播确认

        if obj.mode == "abort":  # 如果是中止模式
            # Abort polling: only the originator checks its own lock state  中止轮询：只有发起者检查自己的锁状态
            while True:  # 循环检查
                self.abort_request(abort_all=True)  # 中止所有请求
                is_locked = await self.model_update_lock.is_locked()  # 检查模型更新锁是否锁定
                if not is_locked:  # 如果未锁定
                    break  # 跳出循环
                await asyncio.sleep(1.0)  # 等待1秒后重试

    async def continue_generation(self, obj: ContinueGenerationReqInput):  # 继续生成
        loop = asyncio.get_event_loop()  # 获取当前事件循环
        self._pause_continue_future = loop.create_future()  # 创建Future用于等待广播确认
        self.send_to_scheduler.send_pyobj(obj)  # 发送继续生成请求
        await self._pause_continue_future  # 等待广播确认

    def _handle_pause_continue_broadcast(self, obj: PauseContinueBroadcast):  # 处理从路由器收到的暂停/继续广播
        """Called from handle_loop when a broadcast arrives from the router.  当从路由器收到广播时，由handle_loop调用。"""
        loop = asyncio.get_event_loop()  # 获取当前事件循环
        loop.create_task(self._apply_pause_continue_broadcast(obj))  # 创建异步任务应用广播

    async def _apply_pause_continue_broadcast(self, obj: PauseContinueBroadcast):  # 在条件锁下应用暂停/继续状态
        """Apply pause/continue state under the condition lock.  在条件锁下应用暂停/继续状态。"""
        async with self.is_pause_cond:  # 获取条件锁
            if obj.is_pause:  # 如果是暂停
                self.is_pause = True  # 设置暂停标志
            else:  # 如果是继续
                self.is_pause = False  # 清除暂停标志
                self.is_pause_cond.notify_all()  # 通知所有等待的协程

        # Resolve the pending future if this worker initiated the pause/continue  如果此工作器发起了暂停/继续，则解决挂起的Future
        if self._pause_continue_future and not self._pause_continue_future.done():  # 如果Future存在且未完成
            self._pause_continue_future.set_result(True)  # 设置Future结果
            self._pause_continue_future = None  # 清除Future引用

    def _attach_multi_http_worker_info(self, req: Union[BaseReq, BaseBatchReq]):  # 为请求附加多HTTP工作器信息

        if isinstance(req, BaseReq):  # 如果是单个请求
            req.http_worker_ipc = self.tokenizer_ipc_name  # 设置HTTP工作器IPC名称
        elif isinstance(req, BaseBatchReq):  # 如果是批量请求
            req.http_worker_ipcs = [self.tokenizer_ipc_name] * len(req.rids)  # 为每个请求ID设置相同的IPC名称
        else:  # 其他类型则报错
            raise ValueError(f"Unknown req type: {type(req)}")


async def print_exception_wrapper(func):  # 异步异常包装器，捕获并打印协程中的异常
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    有时异步函数不会打印异常。
    我们使用另一个包装器来处理异常。
    """
    try:  # 尝试执行
        await func()  # 执行被包装的协程
    except Exception:  # 捕获异常
        traceback = get_exception_traceback()  # 获取异常回溯
        logger.error(f"MultiTokenizerRouter hit an exception: {traceback}")  # 记录错误日志
        if hasattr(func, "__self__") and isinstance(  # 如果函数有__self__属性且是MultiTokenizerRouter实例
            func.__self__, MultiTokenizerRouter
        ):
            func.__self__.dump_requests_before_crash()  # 崩溃前转储请求
        kill_process_tree(os.getpid(), include_parent=True)  # 杀死进程树
        sys.exit(1)  # 退出程序


def get_main_process_id() -> int:  # 获取主进程ID
    """Get the main process ID.  获取主进程ID。

    Supports override via SGLANG_GRANIAN_PARENT_PID for workers whose
    multiprocessing parent PID differs from the shared-memory owner.
    支持通过SGLANG_GRANIAN_PARENT_PID覆盖，用于多进程父PID与共享内存所有者不同的工作器。
    """
    from sglang.srt.environ import envs  # 导入环境变量

    override = envs.SGLANG_GRANIAN_PARENT_PID.get()  # 获取覆盖值
    if override is not None:  # 如果有覆盖值
        return override  # 返回覆盖值
    return multiprocessing.current_process()._parent_pid  # 返回多进程父PID


def write_to_shared_memory(obj, name: str) -> shared_memory.SharedMemory:  # 将数据写入共享内存
    """Write data to shared memory  将数据写入共享内存"""
    serialized = pickle.dumps(obj)  # 序列化对象
    size = len(serialized)  # 获取序列化后的长度
    try:  # 尝试打开已有共享内存
        # Try to open existing shared memory  尝试打开已有的共享内存
        shm = shared_memory.SharedMemory(name=name)  # 打开已有共享内存
        # If size is insufficient, close and recreate  如果大小不足，关闭并重新创建
        if shm.size < size:  # 如果共享内存大小不足
            shm.close()  # 关闭共享内存
            shm.unlink()  # 释放共享内存
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)  # 创建新的共享内存
    except FileNotFoundError:  # 如果共享内存不存在
        # If not present, create new shared memory  如果不存在，创建新的共享内存
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)  # 创建新的共享内存

    shm.buf[:size] = serialized  # 将序列化数据写入共享内存缓冲区
    return shm  # 返回共享内存对象


def read_from_shared_memory(name: str) -> Any:  # 从共享内存读取数据
    """Read data from shared memory  从共享内存读取数据"""
    try:  # 尝试读取
        shm = shared_memory.SharedMemory(name=name)  # 打开共享内存
        data = pickle.loads(bytes(shm.buf))  # 从缓冲区反序列化数据
        shm.close()  # 关闭共享内存
        return data  # 返回数据
    except FileNotFoundError:  # 如果共享内存不存在
        raise FileNotFoundError(f"Shared memory {name} not found")  # 抛出文件未找到错误


def write_data_for_multi_tokenizer(  # 将参数信息写入共享内存，供多分词器使用
    port_args: PortArgs, server_args: ServerArgs, scheduler_info: Dict  # 端口参数、服务器参数、调度器信息
):
    """Write args information to share memory for multi-tokenizer  将参数信息写入共享内存，供多分词器使用"""
    # get main process ID  获取主进程ID
    main_pid = get_main_process_id()  # 获取主进程ID
    current_pid = os.getpid()  # 获取当前进程ID
    logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")  # 记录进程ID信息
    args = (port_args, server_args, scheduler_info)  # 打包参数元组
    args_shm = write_to_shared_memory(args, f"multi_tokenizer_args_{current_pid}")  # 写入共享内存
    args_shm.close()  # 关闭共享内存

    return args_shm  # 返回共享内存对象


class SenderWrapper:  # 发送器包装类，为发送到调度器的消息附加HTTP工作器IPC信息
    def __init__(self, port_args: PortArgs, send_to_scheduler: zmq.Socket):  # 初始化发送器包装器
        self.port_args = port_args  # 保存端口参数
        self.send_to_scheduler = send_to_scheduler  # 保存发送到调度器的套接字

    def send_pyobj(self, obj):  # 发送Python对象，自动附加HTTP工作器IPC名称
        if isinstance(obj, BaseReq):  # 如果是单个请求
            obj.http_worker_ipc = self.port_args.tokenizer_ipc_name  # 设置HTTP工作器IPC名称
        self.send_to_scheduler.send_pyobj(obj)  # 通过套接字发送Python对象
