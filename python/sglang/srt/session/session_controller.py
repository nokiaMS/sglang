# 该文件实现了会话控制器，管理多轮对话会话的生命周期
# 包含会话请求节点、会话类和会话控制器类
# 支持流式和非流式会话模式，处理请求的追加、替换和偏移操作
# 提供会话的打开、关闭、超时回收和多媒体偏移调整等功能

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations  # 启用延迟注解评估

import logging  # 日志模块
import time  # 时间模块
import uuid  # UUID生成模块
from typing import TYPE_CHECKING, Dict, Optional  # 类型注解

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,  # 关闭会话请求输入
    OpenSessionReqInput,  # 打开会话请求输入
    OpenSessionReqOutput,  # 打开会话请求输出
    TokenizedGenerateReqInput,  # 分词后的生成请求输入
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req  # 完成中止标志和请求类
from sglang.srt.utils.common import log_info_on_rank0  # 在rank0上记录信息的工具

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache  # 基础前缀缓存类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class SessionReqNode:  # 会话请求节点类，用于构建请求树
    def __init__(
        self,
        req: Req,  # 请求对象
        parent: Optional["SessionReqNode"] = None,  # 父节点，可选
        children=None,  # 子节点列表，可选
    ):  # 初始化会话请求节点
        """初始化会话请求节点，构建请求的树形结构"""
        self.req = req  # 请求对象
        self.parent = parent  # 父节点
        if parent is not None:  # 如果有父节点
            parent.children.append(self)  # 将自身添加到父节点的子节点列表
        self.children = [] if not children else children  # 子节点列表

    def clear_children(self, req_dict):  # 清除所有子节点
        """递归清除所有子节点的请求"""
        for req_node in self.children:  # 遍历子节点
            req_node.clear(req_dict)  # 递归清除子节点
        self.children = []  # 清空子节点列表

    def clear(self, req_dict):  # 清除当前节点及其所有子节点
        """递归清除当前节点和所有子节点的请求"""
        for req_node in self.children:  # 遍历子节点
            req_node.clear(req_dict)  # 递归清除子节点

        if self.req.finished_reason is None:  # 如果请求未完成
            self.req.to_finish = FINISH_ABORT()  # 设置为中止完成
        del req_dict[self.req.rid]  # 从请求字典中删除

    def abort(self):  # 中止当前节点的请求
        """中止当前节点的请求"""
        if self.req.finished_reason is None:  # 如果请求未完成
            self.req.to_finish = FINISH_ABORT()  # 设置为中止完成

    def __str__(self):  # 字符串表示
        """返回节点的字符串表示"""
        return self._str_helper(self.req.rid)  # 使用辅助方法生成字符串

    def _str_helper(self, prefix=""):  # 递归生成树形字符串
        """递归生成请求树的字符串表示"""
        if len(self.children) == 0:  # 如果没有子节点
            return prefix + "\n"  # 返回前缀加换行
        else:
            origin_prefix = prefix  # 保存原始前缀
            prefix += " -- " + self.children[0].req.rid  # 第一个子节点用"--"
            ret = self.children[0]._str_helper(prefix)  # 递归生成第一个子节点的字符串
            for child in self.children[1:]:  # 遍历其余子节点
                prefix = " " * len(origin_prefix) + " \\- " + child.req.rid  # 其余子节点用"\\-"
                ret += child._str_helper(prefix)  # 递归生成子节点的字符串
            return ret  # 返回完整字符串


class Session:  # 会话类，管理一个对话会话
    def __init__(
        self,
        capacity_of_str_len: int,  # 字符串长度容量
        session_id: Optional[str] = None,  # 会话ID，可选
        streaming: bool = False,  # 是否为流式会话
        timeout: Optional[float] = None,  # 超时时间（秒），可选
    ):  # 初始化会话
        """初始化会话，设置ID、容量、流式模式和超时时间"""
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex  # 生成或使用指定的会话ID
        self.capacity_of_str_len = capacity_of_str_len  # 字符串长度容量
        self.streaming = streaming  # 是否为流式会话
        self.timeout = timeout  # 超时时间
        self.last_active_time: float = time.monotonic()  # 最后活跃时间
        self.req_nodes: Dict[str, SessionReqNode] = {}  # 请求节点字典
        self.close_on_finish: bool = False  # 是否在完成后关闭
        self._inflight: bool = False  # 是否有正在处理的请求

    def is_timed_out(self) -> bool:  # 检查会话是否超时
        """检查会话是否已超时"""
        if self.timeout is None:  # 如果没有设置超时
            return False  # 不超时
        return time.monotonic() - self.last_active_time > self.timeout  # 检查是否超过超时时间

    def create_req(
        self,
        req: TokenizedGenerateReqInput,  # 分词后的生成请求输入
        tokenizer,  # 分词器
        vocab_size: int,  # 词表大小
        eos_token_ids=None,  # 结束token ID列表，可选
    ):  # 创建新的请求
        """根据会话参数创建新请求，处理追加、替换和偏移操作"""
        assert req.session_params is not None  # 断言会话参数存在
        self.last_active_time = time.monotonic()  # 更新最后活跃时间
        session_params = req.session_params  # 获取会话参数

        last_req_node = None  # 上一个请求节点
        last_req = None  # 上一个请求
        abort = False  # 是否中止
        abort_message = ""  # 中止消息
        if self.streaming:  # 如果是流式会话
            # Streaming sessions: only simple appends allowed; reject otherwise.  # 流式会话：仅允许简单追加，否则拒绝
            if self._inflight:  # 如果已有进行中的请求
                abort = True  # 标记中止
                abort_message = "Streaming session already has an active request."  # 设置中止消息
            elif session_params.replace:  # 如果尝试替换
                abort = True  # 标记中止
                abort_message = "Streaming sessions do not support replace."  # 设置中止消息
            elif session_params.drop_previous_output:  # 如果尝试丢弃之前的输出
                abort = True  # 标记中止
                abort_message = (
                    "Streaming sessions do not support drop_previous_output."  # 设置中止消息
                )
            elif session_params.offset and session_params.offset != 0:  # 如果有非零偏移
                abort = True  # 标记中止
                abort_message = "Streaming sessions do not support offset."  # 设置中止消息
            elif self.req_nodes:  # 如果有已有的请求节点
                assert len(self.req_nodes) == 1  # 断言只有一个请求节点
                # Peek (don't pop) the single req_node. req_nodes is updated  # 查看（不弹出）唯一的请求节点
                # only in finish_req after the request completes successfully.  # 仅在请求成功完成后在finish_req中更新
                [last_req_node] = self.req_nodes.values()  # 获取唯一的请求节点
                last_req = last_req_node.req  # 获取其请求
        elif session_params.replace:  # 如果是替换模式
            if session_params.rid is None:  # 如果没有指定要替换的请求ID
                for _, req_node in self.req_nodes.items():  # 遍历所有请求节点
                    req_node.clear(self.req_nodes)  # 清除所有请求
            else:
                if session_params.rid not in self.req_nodes:  # 如果指定的请求ID不存在
                    abort = True  # 标记中止
                    abort_message = "Invalid request session id"  # 设置中止消息
                else:
                    last_req_node = self.req_nodes[session_params.rid]  # 获取指定的请求节点
                    last_req_node.abort()  # 中止该请求
                    last_req = last_req_node.req  # 获取其请求
                    last_req_node.clear_children(self.req_nodes)  # 清除其子节点
        else:  # 非替换模式（追加模式）
            if session_params.rid is not None:  # 如果指定了请求ID
                if session_params.rid not in self.req_nodes:  # 如果指定的请求ID不存在
                    abort = True  # 标记中止
                    abort_message = "Invalid request session id"  # 设置中止消息
                else:
                    last_req_node = self.req_nodes[session_params.rid]  # 获取指定的请求节点
                    last_req = last_req_node.req  # 获取其请求
                    if not last_req.finished():  # 如果上一个请求未完成
                        abort = True  # 标记中止
                        abort_message = "Session request is appending to a request that hasn't finished."  # 设置中止消息
                        logging.warning(abort_message)  # 记录警告

        if last_req is not None:  # 如果有上一个请求
            # trim bos token if it is an append  # 如果是追加操作，去除BOS token
            if (
                tokenizer is not None  # 如果有分词器
                and req.input_ids  # 如果有输入ID
                and req.input_ids[0] == tokenizer.bos_token_id  # 如果第一个token是BOS
            ):
                req.input_ids = req.input_ids[1:]  # 去除BOS token
                # Adjust mm_item offsets since they were computed on  # 调整多媒体项的偏移量，因为它们是在
                # the pre-strip sequence (with BOS at position 0)  # 去除BOS之前的序列上计算的（BOS在位置0）
                if req.mm_inputs:  # 如果有多媒体输入
                    for item in req.mm_inputs.mm_items:  # 遍历多媒体项
                        if item.offsets:  # 如果有偏移量
                            if any(s == 0 for s, _ in item.offsets):  # 如果有起始位置为0的偏移
                                logging.warning(
                                    "mm_item offset starts at 0 (BOS position), "
                                    "clamping to 0 after BOS strip"  # 记录警告
                                )
                            item.offsets = [
                                (max(0, s - 1), max(0, e - 1)) for s, e in item.offsets  # 偏移量减1
                            ]

            input_ids = (  # 构建输入ID序列
                last_req.origin_input_ids  # 上一个请求的原始输入ID
                + last_req.output_ids[: last_req.sampling_params.max_new_tokens]  # 上一个请求的输出ID（截取到最大新token数）
            )

            if session_params.drop_previous_output:  # 如果需要丢弃之前的输出
                input_ids = last_req.origin_input_ids[:]  # 仅使用原始输入ID

            if session_params.offset and session_params.offset != 0:  # 如果有偏移
                input_ids = input_ids[: session_params.offset] + req.input_ids  # 截取到偏移位置后追加新输入
            else:
                input_ids += req.input_ids  # 直接追加新输入

            input_ids_unpadded = (  # 构建未填充的输入ID序列
                last_req.origin_input_ids_unpadded  # 上一个请求的未填充原始输入ID
                + last_req.output_ids[: last_req.sampling_params.max_new_tokens]  # 上一个请求的输出ID
            )
            if session_params.drop_previous_output:  # 如果需要丢弃之前的输出
                input_ids_unpadded = last_req.origin_input_ids_unpadded[:]  # 仅使用未填充的原始输入ID

            if session_params.offset and session_params.offset != 0:  # 如果有偏移
                input_ids_unpadded = (
                    input_ids_unpadded[: session_params.offset] + req.input_ids  # 截取到偏移位置后追加新输入
                )
            else:
                input_ids_unpadded += req.input_ids  # 直接追加新输入
        else:  # 没有上一个请求
            input_ids = req.input_ids  # 使用请求的输入ID
            input_ids_unpadded = req.input_ids  # 使用请求的输入ID

        new_req = Req(  # 创建新请求
            rid=req.rid,  # 请求ID
            origin_input_text=None,  # 原始输入文本（无）
            origin_input_ids=input_ids,  # 原始输入ID
            origin_input_ids_unpadded=input_ids_unpadded,  # 未填充的原始输入ID
            sampling_params=req.sampling_params,  # 采样参数
            lora_id=req.lora_id,  # LoRA ID
            session=self,  # 所属会话
            custom_logit_processor=req.custom_logit_processor,  # 自定义logit处理器
            stream=req.stream,  # 是否流式输出
            return_logprob=req.return_logprob,  # 是否返回logprob
            top_logprobs_num=req.top_logprobs_num,  # top logprob数量
            token_ids_logprob=req.token_ids_logprob,  # 指定token的logprob
            vocab_size=vocab_size,  # 词表大小
            eos_token_ids=eos_token_ids,  # 结束token ID列表
            require_reasoning=req.require_reasoning,  # 是否需要推理
            return_hidden_states=req.return_hidden_states,  # 是否返回隐藏状态
            return_routed_experts=req.return_routed_experts,  # 是否返回路由专家
            routed_experts_start_len=req.routed_experts_start_len,  # 路由专家起始长度
            priority=req.priority,  # 优先级
            routing_key=req.routing_key,  # 路由键
            extra_key=req.extra_key,  # 额外键
            http_worker_ipc=req.http_worker_ipc,  # HTTP工作进程IPC
            time_stats=req.time_stats,  # 时间统计
        )
        if last_req is not None:  # 如果有上一个请求
            new_req.multimodal_inputs = last_req.multimodal_inputs  # 继承上一个请求的多媒体输入
        new_req.tokenizer = tokenizer  # 设置分词器

        if abort:  # 如果需要中止
            new_req.set_finish_with_abort(abort_message)  # 设置中止完成
        elif self.streaming:  # 如果是流式会话
            # req_nodes is NOT updated here — finish_req() handles it.  # req_nodes不在此更新——由finish_req()处理
            self._inflight = True  # 标记为有进行中的请求
        else:  # 非流式会话
            new_req_node = SessionReqNode(new_req, last_req_node)  # 创建新的请求节点
            self.req_nodes[req.rid] = new_req_node  # 添加到请求节点字典

        return new_req  # 返回新请求

    def finish_req(self, req):  # 完成请求
        """Update req_nodes after a streaming request finishes successfully."""  # 流式请求成功完成后更新请求节点
        self._inflight = False  # 清除进行中标志
        if self.req_nodes:  # 如果有请求节点
            [prev_node] = self.req_nodes.values()  # 获取上一个请求节点
            prev_node.req.session = None  # 断开上一个请求与会话的关联
            self.req_nodes.clear()  # 清空请求节点
        self.req_nodes[req.rid] = SessionReqNode(req)  # 添加新的请求节点

    def abort_req(self):  # 中止请求
        """Clear inflight flag on abort (req_nodes stays unchanged)."""  # 中止时清除进行中标志（req_nodes保持不变）
        self._inflight = False  # 清除进行中标志


class SessionController:  # 会话控制器类，管理所有会话
    def __init__(self, tree_cache: BasePrefixCache):  # 初始化会话控制器
        """初始化会话控制器，设置会话字典和树缓存"""
        self.sessions: Dict[str, Session] = {}  # 会话字典
        self._last_reap_time: float = 0.0  # 上次回收时间
        self.tree_cache = tree_cache  # 树缓存

    def __contains__(self, session_id: str) -> bool:  # 检查会话是否存在
        """检查指定ID的会话是否存在"""
        return session_id in self.sessions  # 返回是否包含

    def get(self, session_id: str) -> Optional[Session]:  # 获取会话
        """根据会话ID获取会话对象"""
        return self.sessions.get(session_id)  # 返回会话对象或None

    def open(self, recv_req: OpenSessionReqInput) -> OpenSessionReqOutput:  # 打开会话
        """打开新的会话，创建会话对象并添加到字典中"""
        session_id = recv_req.session_id  # 获取会话ID
        if session_id in self.sessions:  # 如果会话已存在
            logger.warning(f"session id {session_id} already exist, cannot open.")  # 记录警告
            return OpenSessionReqOutput(session_id, False)  # 返回失败
        elif session_id is None:  # 如果会话ID为None
            logger.warning("session id is None, cannot open.")  # 记录警告
            return OpenSessionReqOutput(session_id, False)  # 返回失败
        else:  # 创建新会话
            self.sessions[session_id] = Session(  # 创建会话对象
                recv_req.capacity_of_str_len,  # 字符串长度容量
                session_id,  # 会话ID
                streaming=bool(recv_req.streaming),  # 是否流式
                timeout=recv_req.timeout,  # 超时时间
            )
            log_info_on_rank0(
                logger, f"Session opened: {session_id} (active={len(self.sessions)})"  # 记录信息
            )
            return OpenSessionReqOutput(session_id, True)  # 返回成功

    def close(self, recv_req: CloseSessionReqInput):  # 关闭会话
        """关闭指定ID的会话"""
        session_id = recv_req.session_id  # 获取会话ID
        if session_id not in self.sessions:  # 如果会话不存在
            logger.warning(f"session id {session_id} does not exist, cannot delete.")  # 记录警告
        else:  # 会话存在
            self._close(session_id)  # 执行关闭操作

    def _close(self, session_id: str):  # 内部关闭会话方法
        """执行关闭会话操作，处理未完成请求、释放多媒体特征和树缓存"""
        session = self.sessions[session_id]  # 获取会话对象
        req = None  # 请求对象
        has_unfinished_request = False  # 是否有未完成的请求
        if session.streaming and session._inflight:  # 流式会话且有进行中的请求
            has_unfinished_request = True  # 标记有未完成请求
        elif session.streaming and session.req_nodes:  # 流式会话且有请求节点
            assert len(session.req_nodes) == 1  # 断言只有一个请求节点
            [last_node] = session.req_nodes.values()  # 获取请求节点
            req = last_node.req  # 获取请求
            if not req.finished():  # 如果请求未完成
                has_unfinished_request = True  # 标记有未完成请求

        if has_unfinished_request:  # 如果有未完成的请求
            # An in-flight request is still decoding on this session's KV  # 一个进行中的请求仍在使用此会话的KV进行解码
            # memory. Freeing now would corrupt the scheduler. Mark the  # 现在释放会损坏调度器
            # session for deferred cleanup: the request keeps its session  # 标记会话为延迟清理：请求保持其会话
            # reference so cache_finished_req takes the streaming path,  # 引用，以便cache_finished_req走流式路径
            # and we schedule release_session for after it completes.  # 我们安排在完成后执行release_session
            session.close_on_finish = True  # 标记完成后关闭
            logger.info(
                "Deferring session close for %s (unfinished request)",
                session_id,  # 记录延迟关闭信息
            )
            return  # 直接返回

        # No owning request -- safe to release immediately.  # 没有拥有请求——可以安全地立即释放
        if session.streaming and session.req_nodes:  # 流式会话且有请求节点
            req = next(iter(session.req_nodes.values())).req  # 获取请求
            req.session = None  # 断开请求与会话的关联

        # Release multimodal features held by session requests.  # 释放会话请求持有的多媒体特征
        # Session reqs skip the normal mm cleanup path (scheduler and  # 会话请求跳过正常的多媒体清理路径（调度器和
        # output_processor) so features stay alive until the session closes.  # output_processor），因此特征会保持存活直到会话关闭
        seen_mm = set()  # 已见到的多媒体对象集合
        for node in session.req_nodes.values():  # 遍历请求节点
            mm = node.req.multimodal_inputs  # 获取多媒体输入
            if mm is not None and id(mm) not in seen_mm:  # 如果有多媒体输入且未处理过
                seen_mm.add(id(mm))  # 添加到已处理集合
                mm.release_features()  # 释放多媒体特征
            node.req.multimodal_inputs = None  # 清空多媒体输入引用

        self.tree_cache.release_session(session_id)  # 释放树缓存中的会话
        del self.sessions[session_id]  # 从会话字典中删除
        log_info_on_rank0(
            logger, f"Session closed: {session_id} (active={len(self.sessions)})"  # 记录关闭信息
        )

    def maybe_reap(self, now: float, interval: float = 1.0):  # 可能回收过期会话
        """定期检查并回收已超时的会话和延迟关闭的会话"""
        # reap sessions every second  # 每秒回收一次会话
        if now - self._last_reap_time > interval:  # 如果超过回收间隔
            self._last_reap_time = now  # 更新上次回收时间

            # Finish deferred closes for sessions whose requests completed.  # 完成请求已完成会话的延迟关闭
            pending = [
                sid
                for sid, session in self.sessions.items()
                if session.close_on_finish and self._all_requests_finished(session)  # 等待关闭且所有请求已完成
            ]
            for sid in pending:  # 遍历待关闭的会话
                log_info_on_rank0(
                    logger, f"Deferred close ready for session {sid}, releasing."  # 记录释放信息
                )
                # Reset close_on_finish so _close proceeds with the release.  # 重置close_on_finish以便_close执行释放
                self.sessions[sid].close_on_finish = False  # 重置关闭标志
                self._close(sid)  # 执行关闭

            timed_out = [
                sid for sid, session in self.sessions.items() if session.is_timed_out()  # 已超时的会话
            ]
            for sid in timed_out:  # 遍历超时的会话
                log_info_on_rank0(logger, f"Session {sid} timed out, closing.")  # 记录超时信息
                self._close(sid)  # 关闭超时会话

    @staticmethod
    def _all_requests_finished(session: Session) -> bool:  # 检查会话中所有请求是否已完成
        """检查会话中所有请求是否都已完成"""
        if not session.req_nodes:  # 如果没有请求节点
            return True  # 视为已完成
        return all(node.req.finished() for node in session.req_nodes.values())  # 检查所有请求是否已完成

    @staticmethod
    def adjust_mm_offsets(recv_req: TokenizedGenerateReqInput, req: Req, image_inputs):  # 调整多媒体偏移量
        """For session requests, adjust mm_inputs offsets by the prefix length.  # 对于会话请求，根据前缀长度调整多媒体输入偏移量
        Session.create_req prepends previous context to origin_input_ids,  # Session.create_req将之前的上下文添加到origin_input_ids前面
        so offsets from the new prompt need to be shifted."""  # 因此来自新提示的偏移量需要被偏移
        if len(recv_req.input_ids) >= len(req.origin_input_ids):  # 如果输入ID没有增长
            return  # 无需调整
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)  # 计算前缀长度
        for mm_item in image_inputs.mm_items:  # 遍历多媒体项
            if mm_item.offsets:  # 如果有偏移量
                mm_item.offsets = [
                    (start + prefix_len, end + prefix_len)
                    for start, end in mm_item.offsets  # 所有偏移加上前缀长度
                ]
