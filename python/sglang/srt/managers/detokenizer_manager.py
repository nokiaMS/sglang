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
"""DetokenizerManager is a process that detokenizes the token ids."""
# 反Tokenizer管理器：负责将模型生成的token ID序列解码（反Tokenize）为可读文本的独立进程。
# 该管理器从调度器接收token ID输出，执行增量解码，并将解码后的字符串结果发送回Tokenizer工作进程。

import dataclasses
import logging
import os
import signal
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
# 以上为标准库和类型提示导入

import psutil
import pybase64
import setproctitle
import torch
import zmq

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ConfigureLoggingReq,
    FreezeGCReq,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiHttpWorkerDetokenizerMixin
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, freeze_gc, kill_itself_when_parent_died
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.patch_tokenizer import decode_without_hf_kwargs
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
# 反Tokenizer最大状态数：超过此限制时，最早的请求状态将被驱逐
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""
    # 存储增量解码状态的类，用于跟踪每个请求的解码进度

    decoded_text: str  # 已确认完成的解码文本
    decode_ids: List[int]  # 当前累计的所有token ID列表
    surr_offset: int  # 增量解码中已发送给tokenizer的起始偏移量
    read_offset: int  # 已读取（确认可输出）的token偏移量
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0  # 已发送给客户端的文本偏移量，用于增量输出


class DetokenizerManager(MultiHttpWorkerDetokenizerMixin):
    """DetokenizerManager is a process that detokenizes the token ids."""
    # 反Tokenizer管理器：独立进程，负责将token ID反编码为文本字符串
    # 支持增量解码、停止词裁剪、批量解码优化等功能

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init inter-process communication
        # 初始化进程间通信通道
        self.init_ipc_channels(port_args, server_args)

        # Init tokenizer
        # 初始化分词器
        self.init_tokenizer(server_args)

        # Init running status
        # 初始化运行状态（解码状态字典、看门狗等）
        self.init_running_status(server_args)

        # Init dispatcher
        # 初始化请求分发器，根据消息类型路由到对应处理函数
        self.init_request_dispatcher()

    def init_ipc_channels(self, port_args: PortArgs, server_args: ServerArgs):
        # 初始化ZMQ进程间通信通道
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )  # 从调度器接收token ID输出的ZMQ PULL套接字
        # In multi-tokenizer mode, results are pushed back to each TokenizerWorker
        # directly via SocketMapping inside multi_http_worker_event_loop, so the
        # single send_to_tokenizer socket is unused.
        # 多Tokenizer模式下，结果通过SocketMapping直接推送到各TokenizerWorker，不需要单一socket
        if server_args.tokenizer_worker_num == 1:
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )  # 向TokenizerWorker发送解码结果的ZMQ PUSH套接字

    def init_tokenizer(self, server_args: ServerArgs):
        # 初始化HuggingFace分词器，用于将token ID解码为文本
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )

    def init_running_status(self, server_args: ServerArgs):
        # 初始化运行时状态
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)  # 有限容量的解码状态字典，超容量时淘汰最早的请求
        self.disable_tokenizer_batch_decode = server_args.disable_tokenizer_batch_decode  # 是否禁用批量解码
        self.is_tool_call_parser_gpt_oss = server_args.tool_call_parser == "gpt-oss"  # 是否为gpt-oss工具调用解析器

        # 初始化软看门狗，用于检测反Tokenizer进程是否卡住
        self.soft_watchdog = Watchdog.create(
            debug_name="DetokenizerManager",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DETOKENIZER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("detokenizer")  # 启动CPU监控线程，收集指标

    def init_request_dispatcher(self):
        # 初始化基于类型的请求分发器，将不同类型的消息路由到对应的处理函数
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),  # 嵌入模型输出，无需反编码
                (BatchTokenIDOutput, self.handle_batch_token_id_out),  # token ID输出，需要反编码为文本
                (FreezeGCReq, self.handle_freeze_gc_req),  # 冻结垃圾回收请求
                (ConfigureLoggingReq, self.handle_configure_logging_req),  # 配置日志级别请求
            ]
        )

    def event_loop(self):
        """The event loop that handles requests"""
        # 主事件循环：持续接收调度器消息并分发处理
        while True:
            with self.soft_watchdog.disable():
                recv_obj = self.recv_from_scheduler.recv_pyobj()  # 从调度器接收消息
            output = self._request_dispatcher(recv_obj)  # 根据消息类型分发到对应处理函数
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)  # 将处理结果发送回TokenizerWorker
            self.soft_watchdog.feed()  # 喂狗，表示进程正常运行

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        # 裁剪输出中匹配到的停止字符串或停止token
        if no_stop_trim or not finished_reason:
            return output  # 如果禁止裁剪或未结束，直接返回

        matched = finished_reason.get("matched", None)  # 获取匹配到的停止条件
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        # 裁剪停止字符串：从输出文本中移除匹配到的停止字符串及其后内容
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token.
        # 裁剪停止token：从输出token ID列表中移除最后一个匹配到的停止token
        if isinstance(matched, int) and isinstance(output, list):
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            # gpt-oss模型的工具调用token需要特殊处理，不裁剪
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]  # 移除最后一个停止token
        return output

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOutput):
        # If it is embedding model, no detokenization is needed.
        # 嵌入模型输出无需反编码，直接透传
        return recv_obj

    def _grouped_batch_decode(
        self,
        ids_list: List[List[int]],
        skip_list: List[bool],
        space_list: List[bool],
    ) -> List[str]:
        """Batch decode with grouping by (skip_special_tokens, spaces_between_special_tokens)."""
        # 按(skip_special_tokens, spaces_between_special_tokens)分组批量解码，提高解码效率

        if not getattr(self.tokenizer, "is_fast", False):
            # 非fast tokenizer时逐个解码
            return [
                decode_without_hf_kwargs(self.tokenizer, ids, skip)
                for ids, skip in zip(ids_list, skip_list)
            ]

        # fast path
        # 快速路径：如果所有请求的skip和space参数相同，直接批量解码
        first_skip, first_space = skip_list[0], space_list[0]
        if all(
            s == first_skip and sp == first_space
            for s, sp in zip(skip_list, space_list)
        ):
            return self.tokenizer.batch_decode(
                ids_list,
                skip_special_tokens=first_skip,
                spaces_between_special_tokens=first_space,
            )

        # Group indices by (skip, space) tuple
        # 按参数分组：将相同(skip, space)的请求分组后批量解码
        groups: Dict[Tuple[bool, bool], List[int]]
        groups = defaultdict(list)
        for idx, (skip, space) in enumerate(zip(skip_list, space_list)):
            groups[(skip, space)].append(idx)

        # Decode each group and collect results
        # 对每组分别批量解码，然后将结果按原始顺序组合
        results: List[str] = [""] * len(ids_list)
        for (skip, space), indices in groups.items():
            decoded = self.tokenizer.batch_decode(
                [ids_list[idx] for idx in indices],
                skip_special_tokens=skip,
                spaces_between_special_tokens=space,
            )
            for idx, text in zip(indices, decoded):
                results[idx] = text

        return results

    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
        # 批量解码token ID输出的核心方法：执行增量解码并生成输出字符串
        bs = len(recv_obj.rids)  # 批次大小

        # Initialize decode status
        # 初始化或更新每个请求的解码状态
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                # 新请求：创建初始解码状态
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=list(recv_obj.decode_ids[i]),
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                # 已有请求：追加新的token ID
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            # 裁剪停止token后，获取需要解码的token ID范围
            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])  # 上次已确认的token ID范围

        # Decode token ids to strings
        # 将token ID列表解码为文本字符串
        if not self.disable_tokenizer_batch_decode:
            # 启用批量解码：使用分组批量解码提高效率
            surr_texts = self._grouped_batch_decode(
                surr_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
            read_texts = self._grouped_batch_decode(
                read_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
        else:
            # Do not use batch decode to prevent some detokenization edge cases (e.g., gpt-oss).
            # 禁用批量解码：逐个解码以避免某些反编码边界情况（如gpt-oss模型）
            surr_texts = [
                self.tokenizer.decode(
                    surr, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for surr, skip, space in zip(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
            read_texts = [
                self.tokenizer.decode(
                    read, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for read, skip, space in zip(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]

        # Incremental decoding
        # 增量解码：计算新增的文本内容并更新解码状态
        output_strs = []
        for i in range(bs):
            rid = recv_obj.rids[i]
            try:
                s = self.decode_status[rid]
            except KeyError:
                raise RuntimeError(
                    f"Decode status not found for request {rid}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            # 计算新增文本：用当前完整解码结果减去上次已确认的部分
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                # 流式输出且请求未结束：更新解码状态
                if new_text and not new_text.endswith("�"):
                    # 新文本不以乱码结尾，说明可以安全确认这部分文本
                    s.decoded_text += new_text  # 追加到已确认的解码文本
                    s.surr_offset = s.read_offset  # 更新增量解码偏移量
                    s.read_offset = len(s.decode_ids)  # 更新已读取偏移量
                    new_text = ""  # 新文本已合并到decoded_text，清空增量部分
                else:
                    # 文本以乱码（不完整UTF-8字符）结尾，只输出可打印部分，等待更多token
                    new_text = find_printable_text(new_text)
            else:
                # 请求已完成：从解码状态字典中删除
                if rid in self.decode_status:
                    del self.decode_status[rid]

            # 裁剪停止字符串后，计算本次增量输出
            output_str = self.trim_matched_stop(
                s.decoded_text + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )

            # Incrementally send text.
            # 计算增量输出：只发送自上次以来新增的文本
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)  # 更新已发送偏移量
            output_strs.append(incremental_output)

        return output_strs

    @staticmethod
    def _b64_encode_per_request(
        data_list: Optional[List[Optional[torch.Tensor]]],
    ) -> Optional[List[Optional[str]]]:
        """Encode a per-request list of tensors as base64 strings, off the
        tokenizer hot path. Returns None when the input is None; per-item None
        stays None.
        """
        # 将每个请求的张量数据编码为base64字符串，避免在tokenizer热路径上进行序列化
        if data_list is None:
            return None
        return [
            (
                pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")  # 将张量转为numpy再编码为base64
                if item is not None
                else None
            )
            for item in data_list
        ]

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # 处理批量token ID输出：解码为文本并构造返回结果
        # If handling idle batch, set output_strs to [].
        # 如果是空闲批次（无请求），输出为空列表
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            if len(recv_obj.rids) > 0
            else []
        )
        routed_experts = self._b64_encode_per_request(recv_obj.routed_experts)  # 编码路由专家信息
        indexer_topk = self._b64_encode_per_request(recv_obj.indexer_topk)  # 编码索引topk信息
        # 构造BatchStrOutput，将token ID输出转换为字符串输出
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            reasoning_tokens=recv_obj.reasoning_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            cached_tokens_details=recv_obj.cached_tokens_details,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_num_correct_drafts=recv_obj.spec_num_correct_drafts,
            spec_correct_drafts_histogram=recv_obj.spec_correct_drafts_histogram,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_token_entropy_val=recv_obj.output_token_entropy_val,
            output_hidden_states=recv_obj.output_hidden_states,
            routed_experts=routed_experts,
            indexer_topk=indexer_topk,
            customized_info=recv_obj.customized_info,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=recv_obj.retraction_counts,
            token_steps=recv_obj.token_steps,
            dp_ranks=recv_obj.dp_ranks,
            time_stats=recv_obj.time_stats,
        )

    def handle_freeze_gc_req(self, recv_req: FreezeGCReq):
        # 处理冻结垃圾回收请求
        freeze_gc("Detokenizer Manager")
        return None

    def handle_configure_logging_req(self, recv_req: ConfigureLoggingReq):
        # 处理配置日志级别请求
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())  # 动态调整日志级别


def is_health_check_request(rid: Optional[str]) -> bool:
    # 判断请求ID是否为健康检查请求
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


class LimitedCapacityDict(OrderedDict):
    # 有限容量字典：当元素数量超过容量限制时，自动淘汰最早插入的元素（FIFO策略）
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity  # 最大容量

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)  # 容量已满，移除最早插入的元素
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    # 反Tokenizer进程入口函数：启动并运行反Tokenizer管理器
    kill_itself_when_parent_died()  # 父进程死亡时自动退出
    setproctitle.setproctitle("sglang::detokenizer")  # 设置进程标题
    configure_logger(server_args)  # 配置日志
    parent_process = psutil.Process().parent()  # 获取父进程引用，用于异常时发送信号

    manager = None
    try:
        manager = detokenizer_manager_class(server_args, port_args)  # 创建反Tokenizer管理器实例
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()  # 单Tokenizer模式：运行标准事件循环
        else:
            manager.multi_http_worker_event_loop()  # 多Tokenizer模式：运行多HTTP工作进程事件循环
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        if manager is not None:
            manager.maybe_clear_socket_mapping()  # 清理socket映射
        parent_process.send_signal(signal.SIGQUIT)  # 通知父进程异常退出
