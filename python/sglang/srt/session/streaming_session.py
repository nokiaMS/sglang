# 该文件实现了流式会话的KV缓存保存和恢复机制
# 通过在BasePrefixCache之上包装流式会话功能，支持会话间KV状态的持久化
# 包含虚拟节点、会话槽、流式会话缓存等核心组件
# 处理会话的匹配前缀、缓存完成/未完成请求、释放会话等操作

from __future__ import annotations  # 启用延迟注解评估

import logging  # 日志模块
from dataclasses import dataclass, field  # 数据类和字段装饰器
from typing import TYPE_CHECKING, Any, Dict, Optional  # 类型注解

import torch  # PyTorch张量库

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,  # 基础前缀缓存类
    DecLockRefParams,  # 减少锁引用参数
    DecLockRefResult,  # 减少锁引用结果
    EvictParams,  # 驱逐参数
    EvictResult,  # 驱逐结果
    IncLockRefResult,  # 增加锁引用结果
    InitLoadBackParams,  # 初始化加载回写参数
    MatchPrefixParams,  # 匹配前缀参数
    MatchResult,  # 匹配结果
)
from sglang.srt.utils.common import ceil_align  # 向上对齐工具函数

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.managers.schedule_batch import Req  # 请求类


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class _VirtualNode:  # 虚拟节点类，用于流式会话请求的哨兵节点
    """Sentinel node for streaming session requests.

    Passed to inc_lock_ref / dec_lock_ref so the cache can distinguish
    streaming-session locks (no-op) from real radix-tree locks (forwarded).
    """  # 流式会话请求的哨兵节点，传递给inc_lock_ref/dec_lock_ref以便缓存区分流式会话锁（无操作）和真实的基数树锁（转发）

    pass


@dataclass
class SessionSlot:  # 会话槽数据类，保存流式会话轮次之间的KV状态
    """Holds KV state between streaming session turns."""  # 在流式会话轮次之间保存KV状态

    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)  # 虚拟节点

    # KV pool state (None means no KV is currently held by this slot)  # KV池状态（None表示此槽当前未持有KV）
    req_pool_idx: Optional[int] = None  # 请求池索引
    kv_committed_len: int = 0  # KV已提交长度
    kv_allocated_len: int = 0  # KV已分配长度

    # First req's radix tree node (for dec_lock_ref on session close)  # 第一个请求的基数树节点（用于会话关闭时的dec_lock_ref）
    last_node: Any = None  # 上一个节点
    cache_protected_len: int = 0  # 缓存保护长度
    swa_uuid_for_lock: Optional[str] = None  # 滑动窗口注意力锁的UUID

    # SWA state  # 滑动窗口注意力状态
    swa_evicted_seqlen: int = 0  # 滑动窗口注意力已驱逐的序列长度

    # Mamba states  # Mamba状态
    mamba_pool_idx: Any = None  # Mamba池索引
    mamba_ping_pong_track_buffer: Any = None  # Mamba乒乓追踪缓冲区
    mamba_next_track_idx: Any = None  # Mamba下一个追踪索引
    mamba_last_track_seqlen: Any = None  # Mamba最后追踪序列长度
    mamba_branching_seqlen: Any = None  # Mamba分支序列长度

    @property
    def is_holding_kv(self) -> bool:  # 检查是否持有KV资源
        """Whether this slot currently holds KV pool resources."""  # 此槽当前是否持有KV池资源
        return self.req_pool_idx is not None  # 请求池索引不为None则表示持有

    def save_from_req(self, req: Req, is_first: bool):  # 从完成的请求保存KV状态到槽
        """Save KV state from a finishing request into this slot."""  # 从完成的请求保存KV状态到此槽
        self.req_pool_idx = req.req_pool_idx  # 保存请求池索引
        self.kv_committed_len = req.kv_committed_len  # 保存KV已提交长度
        self.kv_allocated_len = req.kv_allocated_len  # 保存KV已分配长度
        self.swa_evicted_seqlen = req.swa_evicted_seqlen  # 保存SWA已驱逐序列长度

        if is_first:  # 如果是第一个请求
            self.last_node = req.last_node  # 保存上一个节点
            self.cache_protected_len = req.cache_protected_len  # 保存缓存保护长度
            self.swa_uuid_for_lock = req.swa_uuid_for_lock  # 保存SWA锁UUID

        self.mamba_pool_idx = req.mamba_pool_idx  # 保存Mamba池索引
        self.mamba_ping_pong_track_buffer = req.mamba_ping_pong_track_buffer  # 保存Mamba乒乓追踪缓冲区
        self.mamba_next_track_idx = req.mamba_next_track_idx  # 保存Mamba下一个追踪索引
        self.mamba_last_track_seqlen = req.mamba_last_track_seqlen  # 保存Mamba最后追踪序列长度
        self.mamba_branching_seqlen = req.mamba_branching_seqlen  # 保存Mamba分支序列长度

        req.req_pool_idx = None  # 清空请求的池索引
        req.mamba_pool_idx = None  # 清空请求的Mamba池索引

    def restore_to_req(self, req: Req):  # 从槽恢复KV状态到请求
        """Restore KV state from this slot into an incoming request."""  # 从此槽恢复KV状态到传入的请求
        req.req_pool_idx = self.req_pool_idx  # 恢复请求池索引
        req.kv_committed_len = self.kv_committed_len  # 恢复KV已提交长度
        req.kv_allocated_len = self.kv_allocated_len  # 恢复KV已分配长度
        req.swa_evicted_seqlen = self.swa_evicted_seqlen  # 恢复SWA已驱逐序列长度
        req.swa_uuid_for_lock = self.swa_uuid_for_lock  # 恢复SWA锁UUID

        req.mamba_pool_idx = self.mamba_pool_idx  # 恢复Mamba池索引
        req.mamba_ping_pong_track_buffer = self.mamba_ping_pong_track_buffer  # 恢复Mamba乒乓追踪缓冲区
        req.mamba_next_track_idx = self.mamba_next_track_idx  # 恢复Mamba下一个追踪索引
        req.mamba_last_track_seqlen = self.mamba_last_track_seqlen  # 恢复Mamba最后追踪序列长度
        req.mamba_branching_seqlen = self.mamba_branching_seqlen  # 恢复Mamba分支序列长度

        # NOTE: req_pool_idx and mamba_pool_idx are intentionally NOT cleared  # 注意：req_pool_idx和mamba_pool_idx故意不清除
        # from the slot. During chunked prefill, a request may be rejected by  # 从槽中。在分块预填充期间，请求可能被
        # the scheduler (e.g. budget exhausted) and retried in the next cycle.  # 调度器拒绝（例如预算耗尽）并在下一个周期重试
        # Each retry calls match_prefix -> restore_to_req again, so the slot  # 每次重试都调用match_prefix -> restore_to_req，所以槽
        # must remain intact for idempotent restoration.  # 必须保持完整以支持幂等恢复


def _is_streaming(req: Optional[Req]) -> bool:  # 检查请求是否为流式会话请求
    """检查请求是否为流式会话请求"""
    return req is not None and req.session is not None and req.session.streaming  # 请求非空且会话非空且为流式


class StreamingSession(BasePrefixCache):  # 流式会话缓存类，在BasePrefixCache之上添加流式会话KV保存/恢复
    """Adds streaming-session KV save/restore on top of any BasePrefixCache.

    Works both as an external wrapper (``StreamingSession(RadixCache(...))``)
    and in embedded composition (``StreamingSession(inner=self)``). For the
    embedded case, the composing cache must pre-check dispatch conditions
    (``_is_streaming`` / ``find_active_slot`` / ``has_slot``) so the internal
    fall-through to ``self.inner.xxx`` never fires -- otherwise it recurses.
    """  # 在任何BasePrefixCache之上添加流式会话KV保存/恢复，支持外部包装和内嵌组合两种模式

    def __init__(self, inner: BasePrefixCache):  # 初始化流式会话缓存
        """初始化流式会话缓存，设置内部缓存和会话槽字典"""
        self.inner = inner  # 内部前缀缓存
        self.slots: Dict[str, SessionSlot] = {}  # 会话槽字典

    # -- Forward PrefixCacheTrait properties to inner cache --  # 将PrefixCacheTrait属性转发到内部缓存

    @property
    def req_to_token_pool(self):  # 请求到token的映射池属性
        """获取请求到token的映射池"""
        return self.inner.req_to_token_pool  # 返回内部缓存的映射池

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):  # 设置请求到token的映射池
        """设置请求到token的映射池"""
        self.inner.req_to_token_pool = value  # 设置内部缓存的映射池

    @property
    def token_to_kv_pool_allocator(self):  # token到KV池分配器属性
        """获取token到KV池分配器"""
        return self.inner.token_to_kv_pool_allocator  # 返回内部缓存的分配器

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):  # 设置token到KV池分配器
        """设置token到KV池分配器"""
        self.inner.token_to_kv_pool_allocator = value  # 设置内部缓存的分配器

    @property
    def page_size(self):  # 页大小属性
        """获取页大小"""
        return self.inner.page_size  # 返回内部缓存的页大小

    @page_size.setter
    def page_size(self, value):  # 设置页大小
        """设置页大小"""
        self.inner.page_size = value  # 设置内部缓存的页大小

    @property
    def disable(self):  # 禁用标志属性
        """获取禁用标志"""
        return self.inner.disable  # 返回内部缓存的禁用标志

    @disable.setter
    def disable(self, value):  # 设置禁用标志
        """设置禁用标志"""
        self.inner.disable = value  # 设置内部缓存的禁用标志

    @property
    def metrics_collector(self):  # 指标收集器属性
        """获取指标收集器"""
        return self.inner.metrics_collector  # 返回内部缓存的指标收集器

    @metrics_collector.setter
    def metrics_collector(self, value):  # 设置指标收集器
        """设置指标收集器"""
        self.inner.metrics_collector = value  # 设置内部缓存的指标收集器

    # -- Condition helpers (used by embedded-mode callers for pre-dispatch) --  # 条件辅助方法（供内嵌模式调用者用于预分发）

    def has_slot(self, session_id: str) -> bool:  # 检查会话是否有槽
        """检查指定会话ID是否有对应的会话槽"""
        return session_id in self.slots  # 返回是否在槽字典中

    def any_holding_kv(self) -> bool:  # 检查是否有任何槽持有KV
        """检查是否有任何会话槽当前持有KV资源"""
        return any(s.is_holding_kv for s in self.slots.values())  # 检查所有槽的持有状态

    # -- Try-handle entries for composition (see class docstring) --  # 组合的尝试处理入口（见类文档字符串）

    def try_inc_lock_ref(self, node: Any) -> Optional[IncLockRefResult]:  # 尝试增加锁引用
        """No-op lock if ``node`` is a session-internal sentinel; returns
        None to tell the caller to run its raw tree lock path."""  # 如果节点是会话内部哨兵则无操作锁；返回None告诉调用者运行原始树锁路径
        if isinstance(node, _VirtualNode):  # 如果是虚拟节点
            return IncLockRefResult()  # 返回空结果（无操作）
        return None  # 返回None，调用者应继续原始路径

    def try_dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> Optional[DecLockRefResult]:  # 尝试减少锁引用
        """如果节点是虚拟节点则无操作，否则返回None让调用者走原始路径"""
        if isinstance(node, _VirtualNode):  # 如果是虚拟节点
            return DecLockRefResult()  # 返回空结果（无操作）
        return None  # 返回None，调用者应继续原始路径

    def find_active_slot(self, req: Req) -> Optional[SessionSlot]:  # 查找活跃的会话槽
        """Returns an active slot for this req, or None.

        Side effect: if req is pre-aborted (to_finish set, e.g. input too
        long), detach it from the session so cache_finished_req treats it
        as a normal req. The slot stays intact for the next request.
        """  # 返回此请求的活跃槽，或None。副作用：如果请求预中止，则断开与会话的关联
        if not _is_streaming(req):  # 如果不是流式请求
            return None  # 返回None
        slot = self.slots.get(req.session.session_id)  # 获取会话槽
        if slot is None or slot.req_pool_idx is None:  # 如果没有槽或槽未持有KV
            return None  # 返回None
        if req.to_finish is not None:  # 如果请求预中止
            req.session.abort_req()  # 中止请求
            req.session = None  # 断开与会话的关联
            return None  # 返回None
        return slot  # 返回活跃槽

    # -- BasePrefixCache abstract methods --  # BasePrefixCache抽象方法

    def reset(self):  # 重置流式会话缓存
        """重置所有会话槽和内部缓存"""
        self.slots.clear()  # 清空所有会话槽
        self.inner.reset()  # 重置内部缓存

    # -- Streaming entries: contract with embedded composers (e.g.  # 流式条目：与内嵌组合器（例如
    # UnifiedRadixCache) is a uniform "try_handle_*" pattern. Each method  # UnifiedRadixCache）的契约是统一的"try_handle_*"模式
    # executes the streaming body if applicable and signals whether the  # 每个方法在适用时执行流式体，并信号
    # caller still needs to run its raw path.  # 调用者是否仍需运行原始路径

    def try_match_prefix(self, params: MatchPrefixParams) -> Optional[MatchResult]:  # 尝试匹配前缀
        """Returns a MatchResult iff the request hits an active session slot;
        otherwise None (caller falls back to its raw match)."""  # 仅当请求命中活跃会话槽时返回MatchResult；否则返回None
        slot = self.find_active_slot(params.req)  # 查找活跃槽
        if slot is None:  # 如果没有活跃槽
            return None  # 返回None

        req = params.req  # 获取请求
        slot.restore_to_req(req)  # 从槽恢复KV状态到请求

        # token_ids = fill_ids[:input_len-1] (1-token logit reserve already  # token_ids = fill_ids[:input_len-1]（1-token的logit保留已
        # applied). min handles retract retry where committed_len can  # 应用）。min处理回退重试，其中committed_len可以
        # exceed len(token_ids) by 1.  # 超过len(token_ids) 1
        prefix_len = min(req.kv_committed_len, len(params.key.token_ids))  # 计算前缀长度

        # Streaming sessions are append-only (session_controller rollback  # 流式会话只允许追加（session_controller的回滚
        # ensures req_nodes always points to the last successful req).  # 确保req_nodes始终指向最后一个成功的请求）
        assert prefix_len >= slot.cache_protected_len, (
            f"streaming session prefix shrank: {prefix_len=} < "
            f"{slot.cache_protected_len=}"  # 断言前缀长度不小于缓存保护长度
        )

        # Free orphaned tail: alloc_for_extend will overwrite  # 释放孤立的尾部：alloc_for_extend将覆盖
        # req_to_token[prefix_len:] with new indices. The range  # req_to_token[prefix_len:]用新的索引。范围
        # [prefix_len, kv_allocated_len) has stale indices from the  # [prefix_len, kv_allocated_len)有来自
        # previous turn's decode (e.g. alloc-commit gap on retract,  # 上一轮解码的陈旧索引（例如回退时的分配-提交间隙，
        # or speculative draft tokens).  # 或推测性草稿token）
        self._free_tail(slot, req, prefix_len)  # 释放孤立的尾部KV

        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)  # 获取设备端索引

        return MatchResult(  # 返回匹配结果
            device_indices=device_indices,  # 设备端索引
            last_device_node=slot.virtual_node,  # 上一个设备节点（虚拟节点）
            last_host_node=slot.virtual_node,  # 上一个主机节点（虚拟节点）
            best_match_node=slot.virtual_node,  # 最佳匹配节点（虚拟节点）
            cache_protected_len=slot.cache_protected_len,  # 缓存保护长度
        )

    def try_cache_finished_req(
        self, req: Req, is_insert: bool = True, **kwargs
    ) -> bool:  # 尝试缓存已完成的请求
        """Handles a streaming-session finish (save slot / mid-abort nuke).
        Returns True if handled; False means caller runs its raw path."""  # 处理流式会话完成（保存槽/中止时清除），返回True表示已处理
        if not _is_streaming(req):  # 如果不是流式请求
            return False  # 返回False

        from sglang.srt.managers.schedule_batch import FINISH_ABORT  # 导入中止完成标志

        session_id = req.session.session_id  # 获取会话ID
        slot = self.slots.get(session_id)  # 获取会话槽
        is_first = slot is None  # 是否是第一个请求

        # Mid-processing abort only. Pre-aborted reqs have session=None  # 仅处理中途中止。预中止的请求session=None
        # (set in find_active_slot) and never reach here.  # （在find_active_slot中设置），不会到达这里
        # Nuke all KV via release_session, delete slot. Token IDs stay  # 通过release_session清除所有KV，删除槽。Token ID保留
        # in req_nodes (finish_req was never called -> last successful  # 在req_nodes中（finish_req从未被调用 -> 最后成功的
        # req). Next request re-prefills from scratch.  # 请求）。下一个请求从头重新预填充
        if isinstance(req.finished_reason, FINISH_ABORT):  # 如果是中止完成
            if slot is None:  # 如果没有槽
                # First-request mid-processing abort: create ephemeral  # 第一个请求中途中止：创建临时
                # slot from req state so release_session handles cleanup.  # 从请求状态创建槽以便release_session处理清理
                # Include last_node/cache_protected_len from the req so  # 包含请求的last_node/cache_protected_len以便
                # release_session calls dec_lock_ref on the tree lock.  # release_session在树锁上调用dec_lock_ref
                slot = SessionSlot(
                    req_pool_idx=req.req_pool_idx,  # 请求池索引
                    kv_allocated_len=req.kv_allocated_len,  # KV已分配长度
                    last_node=req.last_node,  # 上一个节点
                    cache_protected_len=req.cache_protected_len,  # 缓存保护长度
                    swa_uuid_for_lock=req.swa_uuid_for_lock,  # SWA锁UUID
                )
                self.slots[session_id] = slot  # 保存临时槽
            slot.kv_allocated_len = max(slot.kv_allocated_len, req.kv_allocated_len)  # 更新已分配长度为较大值
            self.release_session(session_id)  # 释放会话
            req.req_pool_idx = None  # 清空请求池索引
            req.session.abort_req()  # 中止请求
            self._mark_kv_freed(req)  # 标记KV已释放
            return True  # 返回已处理

        if is_first:  # 如果是第一个请求
            slot = SessionSlot()  # 创建新槽
            self.slots[session_id] = slot  # 保存到字典

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)  # 获取完成长度
        )
        self._trim_overshoot(req, finished_len)  # 修剪超出的KV

        slot.save_from_req(req, is_first=is_first)  # 从请求保存KV状态到槽

        # Update req_nodes to this successfully finished request.  # 更新req_nodes为此成功完成的请求
        req.session.finish_req(req)  # 通知会话请求已完成

        self._mark_kv_freed(req)  # 标记KV已释放
        return True  # 返回已处理

    def try_cache_unfinished_req(
        self, req: Req, chunked: bool = False, **kwargs
    ) -> bool:  # 尝试缓存未完成的请求
        """Handles a streaming-session mid-flight cache op:
          - chunked prefill: snapshot current KV as prefix, skip radix
          - subsequent turn: skip radix (slot already holds KV)
        Returns False for first-turn non-chunked (caller must run raw radix
        insert to set up the initial tree lock)."""  # 处理流式会话中途缓存操作：分块预填充时快照当前KV作为前缀；后续轮次跳过基数树
        if not _is_streaming(req):  # 如果不是流式请求
            return False  # 返回False
        if chunked:  # 如果是分块预填充
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)  # 获取已填充位置的KV索引
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)  # 复制为前缀索引
            return True  # 返回已处理
        if req.session.session_id in self.slots:  # 如果会话已有槽（后续轮次）
            return True  # 返回已处理，跳过基数树
        return False  # 返回False，调用者需运行原始基数树插入

    # -- BasePrefixCache abstract methods: thin adapters over try_handle_* --  # BasePrefixCache抽象方法：try_handle_*的薄适配器

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # 匹配前缀
        """匹配前缀，先尝试流式会话路径，失败则回退到内部缓存"""
        result = self.try_match_prefix(params)  # 尝试流式会话匹配
        if result is not None:  # 如果匹配成功
            return result  # 返回结果
        return self.inner.match_prefix(params)  # 回退到内部缓存匹配

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):  # 缓存已完成的请求
        """缓存已完成的请求，先尝试流式会话路径，失败则回退到内部缓存"""
        if self.try_cache_finished_req(req, is_insert=is_insert, **kwargs):  # 尝试流式会话缓存
            return  # 已处理
        self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)  # 回退到内部缓存

    def cache_unfinished_req(self, req: Req, **kwargs):  # 缓存未完成的请求
        """缓存未完成的请求，先尝试流式会话路径，失败则回退到内部缓存"""
        if self.try_cache_unfinished_req(req, **kwargs):  # 尝试流式会话缓存
            return  # 已处理
        self.inner.cache_unfinished_req(req, **kwargs)  # 回退到内部缓存

    def evict(self, params: EvictParams) -> EvictResult:  # 驱逐缓存
        """驱逐缓存，直接委托给内部缓存"""
        return self.inner.evict(params)  # 返回内部缓存的驱逐结果

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:  # 增加锁引用
        """增加锁引用，先尝试虚拟节点路径，失败则委托给内部缓存"""
        result = self.try_inc_lock_ref(node)  # 尝试虚拟节点路径
        if result is not None:  # 如果成功
            return result  # 返回结果
        return self.inner.inc_lock_ref(node)  # 回退到内部缓存

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:  # 减少锁引用
        """减少锁引用，先尝试虚拟节点路径，失败则委托给内部缓存"""
        result = self.try_dec_lock_ref(node, params)  # 尝试虚拟节点路径
        if result is not None:  # 如果成功
            return result  # 返回结果
        return self.inner.dec_lock_ref(node, params)  # 回退到内部缓存

    # -- Session lifecycle --  # 会话生命周期

    def release_session(self, session_id: str) -> None:  # 释放会话
        """释放会话的所有KV资源，减少锁引用并归还内存"""
        slot = self.slots.pop(session_id, None)  # 弹出会话槽
        if slot is None:  # 如果没有槽
            return  # 直接返回
        protected_len = slot.cache_protected_len  # 获取保护长度
        lock_node = slot.last_node  # 获取锁节点
        tokens_freed = (
            max(0, slot.kv_allocated_len - protected_len) if slot.is_holding_kv else 0  # 计算释放的token数
        )
        logger.info(
            "Session KV released: %s (%d tokens freed)", session_id, tokens_freed  # 记录释放信息
        )

        if lock_node is not None:  # 如果有锁节点
            if slot.swa_uuid_for_lock is not None:  # 如果有SWA锁UUID
                self.inner.dec_lock_ref(
                    lock_node,
                    DecLockRefParams(swa_uuid_for_lock=slot.swa_uuid_for_lock),  # 传递SWA参数减少锁引用
                )
            else:
                self.inner.dec_lock_ref(lock_node)  # 减少锁引用

        if slot.is_holding_kv:  # 如果槽持有KV
            start = protected_len  # 起始位置为保护长度
            end = slot.kv_allocated_len  # 结束位置为已分配长度
            if start < end:  # 如果有可释放的范围
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, start:end  # 获取KV索引
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)  # 释放KV索引
            self.req_to_token_pool.free_slots.append(slot.req_pool_idx)  # 归还请求池索引

        self._free_slot_mamba(slot)  # 释放Mamba状态

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:  # 计算会话持有的KV token总数
        """Total KV tokens held by session slots, not tracked by the tree.

        Excludes slots whose KV is currently owned by an owning request --
        those tokens are counted via uncached_size in the busy mem check.
        A slot's pool_idx being in active_pool_idxs indicates a req owns it.
        """  # 会话槽持有的KV token总数，不包含当前由请求拥有的槽
        total = 0  # 总计
        for slot in self.slots.values():  # 遍历所有槽
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs  # 检查槽是否在活跃批次中
            )
            if slot.is_holding_kv and not in_batch:  # 如果槽持有KV且不在活跃批次中
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)  # 对齐已分配长度
                total += allocated - slot.cache_protected_len  # 累加超出保护长度的部分
        return total  # 返回总数

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:  # 会话持有的完整token数
        """An alias to align the naming style of SWA"""  # 用于对齐SWA命名风格的别名
        return self.session_held_tokens(active_pool_idxs)  # 调用session_held_tokens

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:  # 会话持有的SWA token数
        """Total SWA tokens held by session slots, not tracked by the tree."""  # 会话槽持有的SWA token总数
        total = 0  # 总计
        for slot in self.slots.values():  # 遍历所有槽
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs  # 检查槽是否在活跃批次中
            )
            if slot.is_holding_kv and not in_batch:  # 如果槽持有KV且不在活跃批次中
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)  # 对齐已分配长度
                total += allocated - max(
                    slot.cache_protected_len, slot.swa_evicted_seqlen  # 减去保护长度和SWA已驱逐长度的较大值
                )
        return total  # 返回总数

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:  # 会话持有的请求池槽位数
        """Number of req pool slots held by session slots."""  # 会话槽持有的请求池槽位数

        def _owned(s):  # 检查槽是否拥有KV且不在活跃批次中
            in_batch = (
                active_pool_idxs is not None and s.req_pool_idx in active_pool_idxs  # 检查槽是否在活跃批次中
            )
            return s.is_holding_kv and not in_batch  # 持有KV且不在活跃批次

        return sum(_owned(s) for s in self.slots.values())  # 统计拥有的槽位数

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:  # 会话持有的Mamba池条目数
        """Total mamba_pool entries held by session slots (mamba_pool_idx +
        mamba_ping_pong_track_buffer). Excludes slots whose owning req is
        currently in the batch -- those slots are counted via the normal
        alloc/free paths (same convention as the sibling ``session_held_*``
        accessors).
        """  # 会话槽持有的mamba_pool条目总数，排除当前在批次中的请求拥有的槽
        total = 0  # 总计
        for slot in self.slots.values():  # 遍历所有槽
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs  # 检查槽是否在活跃批次中
            )
            if in_batch:  # 如果在活跃批次中
                continue  # 跳过
            if slot.mamba_pool_idx is not None:  # 如果有Mamba池索引
                total += slot.mamba_pool_idx.numel()  # 累加元素数
            if slot.mamba_ping_pong_track_buffer is not None:  # 如果有Mamba乒乓追踪缓冲区
                total += slot.mamba_ping_pong_track_buffer.numel()  # 累加元素数
        return total  # 返回总数

    def _free_slot_mamba(self, slot: SessionSlot) -> None:  # 释放槽的Mamba状态
        """Return a session slot's mamba pool state to the allocator."""  # 将会话槽的Mamba池状态归还给分配器
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)  # 获取Mamba池
        if mamba_pool is None:  # 如果没有Mamba池
            return  # 直接返回
        if slot.mamba_pool_idx is not None:  # 如果有Mamba池索引
            mamba_pool.free(slot.mamba_pool_idx.unsqueeze(0))  # 释放Mamba池索引
            slot.mamba_pool_idx = None  # 清空引用
        if slot.mamba_ping_pong_track_buffer is not None:  # 如果有Mamba乒乓追踪缓冲区
            mamba_pool.free(slot.mamba_ping_pong_track_buffer)  # 释放缓冲区
            slot.mamba_ping_pong_track_buffer = None  # 清空引用

    # -- Internal helpers (streaming body bits) --  # 内部辅助方法（流式体部分）

    def _free_tail(self, slot: SessionSlot, req: Req, prefix_len: int) -> None:  # 释放孤立的尾部KV
        """match_prefix path: free orphaned KV in [prefix_len, kv_allocated_len)
        before alloc_for_extend overwrites it. The gap appears when spec
        decoding pushes allocated above committed, or when retract retry's
        logit-reserve pulls prefix_len below committed.
        """  # match_prefix路径：在alloc_for_extend覆盖前释放[prefix_len, kv_allocated_len)范围内的孤立KV
        self._free_kv_aligned(slot.req_pool_idx, prefix_len, slot.kv_allocated_len)  # 释放对齐的KV
        slot.kv_allocated_len = prefix_len  # 更新槽的已分配长度
        slot.kv_committed_len = min(slot.kv_committed_len, prefix_len)  # 更新槽的已提交长度
        slot.swa_evicted_seqlen = min(slot.swa_evicted_seqlen, prefix_len)  # 更新槽的SWA已驱逐长度
        req.kv_allocated_len = prefix_len  # 更新请求的已分配长度
        req.kv_committed_len = min(req.kv_committed_len, prefix_len)  # 更新请求的已提交长度
        req.swa_evicted_seqlen = min(req.swa_evicted_seqlen, prefix_len)  # 更新请求的SWA已驱逐长度

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:  # 修剪超出的KV
        """Trim slot KV to finished_len boundary. Spec v2 may overshoot
        max_new_tokens (verify round commits M+1 at a time); next turn's
        input is output_ids[:finished_len], so positions past that must
        be released to avoid token/KV mismatch.
        """  # 将槽KV修剪到finished_len边界，避免token/KV不匹配
        target = len(req.origin_input_ids) + finished_len  # 计算目标长度
        self._free_kv_aligned(req.req_pool_idx, target, req.kv_allocated_len)  # 释放超出目标的KV
        req.kv_allocated_len = min(req.kv_allocated_len, target)  # 更新已分配长度
        req.kv_committed_len = min(req.kv_committed_len, target)  # 更新已提交长度
        req.swa_evicted_seqlen = min(req.swa_evicted_seqlen, target)  # 更新SWA已驱逐长度
        req.output_ids = req.output_ids[:finished_len]  # 截断输出ID

    def _free_kv_aligned(self, pool_idx: int, target: int, end: int) -> None:  # 按页对齐释放KV
        """Free req_to_token[pool_idx, ceil_align(target):end). Page-aligned
        because PagedTokenToKVPoolAllocator.free returns whole pages
        (free_index // page_size), so partial-page free would corrupt pages
        still holding committed tokens. The range [target, ceil_align(target))
        stays attached until release_session frees the whole page.
        """  # 释放req_to_token[pool_idx, ceil_align(target):end)，按页对齐释放以避免损坏仍持有已提交token的页
        if end <= target:  # 如果没有需要释放的范围
            return  # 直接返回
        start = target  # 起始位置
        if self.page_size > 1:  # 如果页大小大于1
            start = ceil_align(start, self.page_size)  # 向上对齐到页边界
        if start < end:  # 如果对齐后仍有可释放范围
            tail = self.req_to_token_pool.req_to_token[pool_idx, start:end]  # 获取尾部索引
            self.token_to_kv_pool_allocator.free(tail)  # 释放尾部KV

    @staticmethod
    def _mark_kv_freed(req: Req) -> None:  # 标记请求的KV已释放
        """Set bookkeeping flags so busy check skips this finished req."""  # 设置记账标志，使忙碌检查跳过此已完成请求
        if not req.kv_committed_freed:  # 如果已提交KV未释放
            req.pop_committed_kv_cache()  # 弹出已提交KV缓存
        if not req.kv_overallocated_freed:  # 如果过度分配KV未释放
            req.pop_overallocated_kv_cache()  # 弹出过度分配KV缓存

    # -- Pass-through methods --  # 透传方法

    def evictable_size(self):  # 可驱逐大小
        """获取内部缓存的可驱逐大小"""
        return self.inner.evictable_size()  # 透传到内部缓存

    def full_evictable_size(self):  # 完整可驱逐大小
        """获取内部缓存的完整可驱逐大小"""
        return self.inner.full_evictable_size()  # 透传到内部缓存

    def swa_evictable_size(self):  # SWA可驱逐大小
        """获取内部缓存的SWA可驱逐大小"""
        return self.inner.swa_evictable_size()  # 透传到内部缓存

    def protected_size(self):  # 保护大小
        """获取内部缓存的保护大小"""
        return self.inner.protected_size()  # 透传到内部缓存

    def full_protected_size(self):  # 完整保护大小
        """获取内部缓存的完整保护大小"""
        return self.inner.full_protected_size()  # 透传到内部缓存

    def swa_protected_size(self):  # SWA保护大小
        """获取内部缓存的SWA保护大小"""
        return self.inner.swa_protected_size()  # 透传到内部缓存

    def total_size(self):  # 总大小
        """获取内部缓存的总大小"""
        return self.inner.total_size()  # 透传到内部缓存

    def pretty_print(self):  # 美化打印
        """美化打印内部缓存内容"""
        return self.inner.pretty_print()  # 透传到内部缓存

    def init_load_back(self, params: InitLoadBackParams):  # 初始化加载回写
        """初始化加载回写到内部缓存"""
        return self.inner.init_load_back(params)  # 透传到内部缓存

    def ready_to_load_host_cache(self):  # 是否准备好加载主机缓存
        """检查内部缓存是否准备好加载主机缓存"""
        return self.inner.ready_to_load_host_cache()  # 透传到内部缓存

    def flush_write_through_acks(self) -> None:  # 刷新写透确认
        """刷新内部缓存的写透确认"""
        return self.inner.flush_write_through_acks()  # 透传到内部缓存

    def check_hicache_events(self):  # 检查HiCache事件
        """检查内部缓存的HiCache事件"""
        return self.inner.check_hicache_events()  # 透传到内部缓存

    def take_events(self):  # 获取事件
        """获取内部缓存的事件"""
        return self.inner.take_events()  # 透传到内部缓存

    def supports_swa(self):  # 是否支持SWA
        """检查内部缓存是否支持滑动窗口注意力"""
        return self.inner.supports_swa()  # 透传到内部缓存

    def supports_mamba(self):  # 是否支持Mamba
        """检查内部缓存是否支持Mamba"""
        return self.inner.supports_mamba()  # 透传到内部缓存

    def supports_streaming_session(self) -> bool:  # 是否支持流式会话
        """检查是否支持流式会话"""
        return True  # 始终返回True

    def is_chunk_cache(self):  # 是否为分块缓存
        """检查内部缓存是否为分块缓存"""
        return self.inner.is_chunk_cache()  # 透传到内部缓存

    def is_tree_cache(self):  # 是否为树缓存
        """检查内部缓存是否为树缓存"""
        return self.inner.is_tree_cache()  # 透传到内部缓存

    def available_and_evictable_str(self):  # 可用和可驱逐的字符串表示
        """获取内部缓存的可用和可驱逐的字符串表示"""
        return self.inner.available_and_evictable_str()  # 透传到内部缓存

    def init_metrics_collector(self):  # 初始化指标收集器
        """初始化内部缓存的指标收集器"""
        return self.inner.init_metrics_collector()  # 透传到内部缓存

    def sanity_check(self):  # 完整性检查
        # Skip inner sanity check when sessions hold tree locks, because  # 当会话持有树锁时跳过内部完整性检查，因为
        # the check asserts all nodes are unlocked during idle.  # 检查断言空闲时所有节点都是解锁的
        if self.any_holding_kv():  # 如果有任何槽持有KV
            return  # 跳过检查
        self.inner.sanity_check()  # 执行内部缓存完整性检查

    # Forward attribute access for cache-specific methods (e.g.  # 转发属性访问以支持缓存特定方法（例如
    # sliding_window_size, all_values_flatten, etc.)  # sliding_window_size, all_values_flatten等）
    def __getattr__(self, name):  # 属性访问代理
        """将未定义的属性访问转发到内部缓存"""
        return getattr(self.inner, name)  # 转发到内部缓存
